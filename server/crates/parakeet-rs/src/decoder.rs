use crate::boost::Boost;
use crate::model::Model;
use crate::runtime::{CpuRuntime, Graph};
use crate::{sys, Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32,
    pub frame: usize,
    pub duration_frames: usize,
    /// Softmax probability of this token over the vocabulary at its step.
    pub prob: f32,
}

struct StepGraph {
    graph: Graph,
    embedding: *mut sys::ggml_tensor,
    encoder: *mut sys::ggml_tensor,
    prev_h: Vec<*mut sys::ggml_tensor>,
    prev_c: Vec<*mut sys::ggml_tensor>,
    next_h: Vec<*mut sys::ggml_tensor>,
    next_c: Vec<*mut sys::ggml_tensor>,
    logits: *mut sys::ggml_tensor,
}

struct EmbeddingGraph {
    graph: Graph,
    id: *mut sys::ggml_tensor,
    output: *mut sys::ggml_tensor,
}

impl EmbeddingGraph {
    unsafe fn new(model: &Model) -> Result<Self> {
        let mut graph = Graph::new()?;
        let id = sys::ggml_new_tensor_1d(graph.ctx, sys::ggml_type_GGML_TYPE_F32, 1);
        let id_i32 = sys::ggml_cast(graph.ctx, id, sys::ggml_type_GGML_TYPE_I32);
        let output = sys::ggml_get_rows(graph.ctx, model.cpu_tensor("pred.embed.weight").unwrap(), id_i32);
        graph.output(output);
        Ok(Self { graph, id, output })
    }

    unsafe fn lookup(&mut self, runtime: &CpuRuntime, token: u32, hidden: usize) -> Result<Vec<f32>> {
        runtime.execute(&mut self.graph, &[(self.id, &[token as f32])])?;
        let mut result = vec![0.0f32; hidden];
        sys::ggml_backend_tensor_get(self.output, result.as_mut_ptr().cast(), 0, result.len() * 4);
        Ok(result)
    }
}

impl StepGraph {
    unsafe fn new(model: &Model) -> Result<Self> {
        let mut graph = Graph::new()?;
        let hidden = model.info().predictor_hidden as i64;
        let layers = model.info().predictor_layers as usize;
        let embedding = sys::ggml_new_tensor_1d(graph.ctx, sys::ggml_type_GGML_TYPE_F32, hidden);
        let encoder = sys::ggml_new_tensor_1d(graph.ctx, sys::ggml_type_GGML_TYPE_F32, model.info().encoder_dimension as i64);
        let mut prev_h = Vec::new();
        let mut prev_c = Vec::new();
        let mut next_h = Vec::new();
        let mut next_c = Vec::new();
        let mut input = embedding;
        for layer in 0..layers {
            let h = sys::ggml_new_tensor_1d(graph.ctx, sys::ggml_type_GGML_TYPE_F32, hidden);
            let c = sys::ggml_new_tensor_1d(graph.ctx, sys::ggml_type_GGML_TYPE_F32, hidden);
            prev_h.push(h);
            prev_c.push(c);
            let wx = model.cpu_tensor(&format!("pred.lstm.{layer}.Wx")).unwrap();
            let wh = model.cpu_tensor(&format!("pred.lstm.{layer}.Wh")).unwrap();
            let bias = model.cpu_tensor(&format!("pred.lstm.{layer}.bias"));
            let mut gates = sys::ggml_add(
                graph.ctx,
                sys::ggml_mul_mat(graph.ctx, wx, input),
                sys::ggml_mul_mat(graph.ctx, wh, h),
            );
            if let Some(bias) = bias {
                gates = sys::ggml_add(graph.ctx, gates, bias);
            } else {
                let b_ih = model.cpu_tensor(&format!("decoder.lstm.{layer}.b_ih")).unwrap();
                let b_hh = model.cpu_tensor(&format!("decoder.lstm.{layer}.b_hh")).unwrap();
                gates = sys::ggml_add(graph.ctx, gates, sys::ggml_add(graph.ctx, b_ih, b_hh));
            }
            let part = |index: usize| {
                sys::ggml_view_1d(
                    graph.ctx,
                    gates,
                    hidden,
                    index * hidden as usize * sys::ggml_element_size(gates),
                )
            };
            let i = sys::ggml_sigmoid(graph.ctx, part(0));
            let f = sys::ggml_sigmoid(graph.ctx, part(1));
            let g = sys::ggml_tanh(graph.ctx, part(2));
            let o = sys::ggml_sigmoid(graph.ctx, part(3));
            let nc = sys::ggml_add(graph.ctx, sys::ggml_mul(graph.ctx, f, c), sys::ggml_mul(graph.ctx, i, g));
            let nh = sys::ggml_mul(graph.ctx, o, sys::ggml_tanh(graph.ctx, nc));
            graph.output(nh);
            graph.output(nc);
            next_h.push(nh);
            next_c.push(nc);
            input = nh;
        }
        let enc = sys::ggml_add(
            graph.ctx,
            sys::ggml_mul_mat(graph.ctx, model.cpu_tensor("joint.enc.weight").unwrap(), encoder),
            model.cpu_tensor("joint.enc.bias").unwrap(),
        );
        let pred = sys::ggml_add(
            graph.ctx,
            sys::ggml_mul_mat(graph.ctx, model.cpu_tensor("joint.pred.weight").unwrap(), input),
            model.cpu_tensor("joint.pred.bias").unwrap(),
        );
        let joined = sys::ggml_relu(graph.ctx, sys::ggml_add(graph.ctx, enc, pred));
        let logits = sys::ggml_add(
            graph.ctx,
            sys::ggml_mul_mat(graph.ctx, model.cpu_tensor("joint.out.weight").unwrap(), joined),
            model.cpu_tensor("joint.out.bias").unwrap(),
        );
        graph.output(logits);
        Ok(Self {
            graph,
            embedding,
            encoder,
            prev_h,
            prev_c,
            next_h,
            next_c,
            logits,
        })
    }
}

/// What a decoding step reads and writes, kept whole so a boosted term that
/// doesn't get finished can be taken back.
#[derive(Clone)]
struct State {
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
    embedding: Vec<f32>,
    frame: usize,
    symbols: usize,
    /// Tokens output so far.
    emitted: usize,
    /// Terms being spelled; see `Boost::advance`.
    partial: Vec<(usize, usize)>,
}

impl Model {
    /// Greedy TDT decoding. Where `boost` turns the model's choice toward a term,
    /// the state before is kept, and if the term then isn't spelled to its end the
    /// decoder goes back there and takes the model's own choice: a lone "▁в" from
    /// Віктор is undone, a whole "Мунблум" is kept.
    pub fn decode(&self, encoded: &[f32], frames: usize, boost: &Boost) -> Result<Vec<Token>> {
        let dimension = self.info().encoder_dimension as usize;
        if encoded.len() != frames * dimension {
            return Err(Error::Ggml("encoder output shape"));
        }
        unsafe {
            let runtime = CpuRuntime::new()?;
            let mut step_graph = StepGraph::new(self)?;
            let mut embedding_graph = EmbeddingGraph::new(self)?;
            let hidden = self.info().predictor_hidden as usize;
            let layers = self.info().predictor_layers as usize;
            let mut state = State {
                h: vec![vec![0.0f32; hidden]; layers],
                c: vec![vec![0.0f32; hidden]; layers],
                // NeMo TDT uses the blank embedding as the start-of-sequence
                // predictor input (not an all-zero pseudo-token).
                embedding: embedding_graph.lookup(&runtime, self.tokenizer().blank_id(), hidden)?,
                frame: 0,
                symbols: 0,
                emitted: 0,
                partial: Vec::new(),
            };
            let mut output = Vec::new();
            let mut checkpoint: Option<State> = None;
            // The step right after going back runs without boost.
            let mut unboosted = false;
            let mut iterations = 0;
            let limit = frames * 16 + 1024;
            loop {
                if state.frame >= frames {
                    // A term still unfinished when the audio ends was never said.
                    let Some(saved) = checkpoint.take() else { break };
                    state = saved;
                    output.truncate(state.emitted);
                    unboosted = true;
                }
                if iterations >= limit {
                    break;
                }
                iterations += 1;
                let frame = state.frame;
                let enc = &encoded[frame * dimension..(frame + 1) * dimension];
                let mut inputs = vec![(step_graph.embedding, state.embedding.as_slice()), (step_graph.encoder, enc)];
                for layer in 0..layers {
                    inputs.push((step_graph.prev_h[layer], state.h[layer].as_slice()));
                    inputs.push((step_graph.prev_c[layer], state.c[layer].as_slice()));
                }
                runtime.execute(&mut step_graph.graph, &inputs)?;
                let token_classes = self.info().vocabulary_size as usize;
                let mut logits = vec![0.0f32; token_classes + self.info().joint_extra_outputs as usize];
                sys::ggml_backend_tensor_get(step_graph.logits, logits.as_mut_ptr().cast(), 0, logits.len() * 4);
                let best = |logits: &[f32]| {
                    logits[..token_classes]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|x| x.0)
                        .unwrap()
                };
                let own = best(&logits);
                if !unboosted {
                    boost.apply(&state.partial, &mut logits[..token_classes]);
                }
                unboosted = false;
                let token = best(&logits);
                let mut guess = false;
                if token != own && checkpoint.is_none() {
                    checkpoint = Some(state.clone());
                    guess = true;
                }
                let duration_index = logits[token_classes..]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|x| x.0)
                    .unwrap_or(0);
                let duration = self.info().tdt_durations[duration_index].max(0) as usize;
                let is_blank = token as u32 == self.tokenizer().blank_id();
                if iterations <= 3 {
                    let token_logit = logits[token];
                    let blank_logit = logits[self.tokenizer().blank_id() as usize];
                    eprintln!("[parakeet][decoder] iter={iterations} frame={frame} token={token} token_logit={token_logit:.6} blank={} blank_logit={blank_logit:.6} duration={duration}", self.tokenizer().blank_id());
                }
                if !is_blank {
                    for layer in 0..layers {
                        sys::ggml_backend_tensor_get(step_graph.next_h[layer], state.h[layer].as_mut_ptr().cast(), 0, hidden * 4);
                        sys::ggml_backend_tensor_get(step_graph.next_c[layer], state.c[layer].as_mut_ptr().cast(), 0, hidden * 4);
                    }
                    let token_logit = logits[token];
                    let sum = logits[..token_classes]
                        .iter()
                        .map(|logit| (logit - token_logit).exp())
                        .sum::<f32>();
                    output.push(Token {
                        id: token as u32,
                        frame,
                        duration_frames: duration,
                        prob: 1.0 / sum,
                    });
                    state.emitted = output.len();
                    let finished = boost.advance(&mut state.partial, token as u32);
                    if checkpoint.is_some() && (finished || state.partial.is_empty()) {
                        if finished {
                            checkpoint = None;
                        } else if !guess {
                            state = checkpoint.take().unwrap();
                            output.truncate(state.emitted);
                            unboosted = true;
                            continue;
                        }
                    }
                    state.embedding = embedding_graph.lookup(&runtime, token as u32, hidden)?;
                }
                state.frame += duration;
                state.symbols += 1;
                if duration != 0 {
                    state.symbols = 0;
                } else if self.info().tdt_max_symbols > 0 && state.symbols >= self.info().tdt_max_symbols as usize {
                    state.frame += 1;
                    state.symbols = 0;
                } else if is_blank && self.info().tdt_max_symbols > 0 {
                    iterations += self.info().tdt_max_symbols as usize - state.symbols;
                    state.frame += 1;
                    state.symbols = 0;
                }
            }
            if iterations >= limit {
                return Err(Error::Ggml("RNN-T iteration limit"));
            }
            Ok(output)
        }
    }
}
