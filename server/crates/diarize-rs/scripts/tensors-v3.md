# Nemotron-3-Diarization q8_0 GGUF: KV and tensors

Read from NVIDIA's `Nemotron-3-Diarization.q8_0.gguf` with `dump_gguf.py`. Shapes are GGUF order (reversed vs torch).
The f32 build from `convert_v3.sh` has identical names/shapes/KV except `general.file_type` (7 vs 0) and `general.name`.

## KV (57)

| key | value |
|---|---|
| `GGUF.version` | `3` |
| `GGUF.tensor_count` | `360` |
| `GGUF.kv_count` | `57` |
| `general.architecture` | `'sortformer'` |
| `general.name` | `'Nemotron-3-Diarization.q8_0'` |
| `sortformer.version` | `'v3'` |
| `general.file_type` | `7` |
| `sortformer.encoder.d_model` | `512` |
| `sortformer.encoder.n_layers` | `31` |
| `sortformer.encoder.n_heads` | `8` |
| `sortformer.encoder.d_ff` | `2048` |
| `sortformer.encoder.conv_kernel_size` | `0` |
| `sortformer.encoder.subsampling_factor` | `8` |
| `sortformer.encoder.subsampling_conv_channels` | `256` |
| `sortformer.encoder.feat_in` | `128` |
| `sortformer.encoder.xscaling` | `False` |
| `sortformer.encoder.use_bias` | `True` |
| `sortformer.encoder.pos_emb_max_len` | `5000` |
| `sortformer.encoder.conv_norm` | `'batch_norm'` |
| `sortformer.encoder.conv_context` | `'symmetric'` |
| `sortformer.encoder.att_context_style` | `'regular'` |
| `sortformer.encoder.type` | `'transformer_rope'` |
| `sortformer.encoder.subsampling_type` | `'feature_stacking'` |
| `sortformer.encoder.qkv_bias` | `False` |
| `sortformer.encoder.qk_norm` | `False` |
| `sortformer.encoder.pre_block_norm` | `True` |
| `sortformer.encoder.rope_base` | `10000.0` |
| `sortformer.encoder.rotary_fraction` | `1.0` |
| `sortformer.transformer.n_layers` | `0` |
| `sortformer.transformer.hidden_size` | `192` |
| `sortformer.transformer.inner_size` | `0` |
| `sortformer.transformer.n_heads` | `0` |
| `sortformer.transformer.pre_ln` | `True` |
| `sortformer.num_speakers` | `8` |
| `sortformer.high_resolution` | `True` |
| `sortformer.output_subsampling_factor` | `1` |
| `sortformer.upsample_factor` | `8` |
| `sortformer.learnable_silence` | `True` |
| `sortformer.preprocessor.sample_rate` | `16000` |
| `sortformer.preprocessor.window_size` | `0.02500000037252903` |
| `sortformer.preprocessor.window_stride` | `0.009999999776482582` |
| `sortformer.preprocessor.n_fft` | `512` |
| `sortformer.preprocessor.features` | `128` |
| `sortformer.preprocessor.normalize` | `'NA'` |
| `sortformer.preprocessor.preemph` | `0.9700000286102295` |
| `sortformer.preprocessor.dither` | `9.999999747378752e-06` |
| `sortformer.preprocessor.log_zero_guard` | `5.960464477539063e-08` |
| `sortformer.scoring.spkcache_sil_frames_per_spk` | `1` |
| `sortformer.scoring.pred_score_threshold` | `0.25` |
| `sortformer.scoring.scores_boost_latest` | `0.05000000074505806` |
| `sortformer.scoring.sil_threshold` | `0.20000000298023224` |
| `sortformer.scoring.strong_boost_rate` | `0.75` |
| `sortformer.scoring.weak_boost_rate` | `1.5` |
| `sortformer.scoring.min_pos_scores_rate` | `0.5` |
| `sortformer.streaming.spkcache_len` | `264` |
| `sortformer.streaming.fifo_len` | `0` |
| `sortformer.streaming.chunk_len` | `264` |
| `sortformer.streaming.spkcache_update_period` | `264` |
| `sortformer.streaming.chunk_left_context` | `0` |
| `sortformer.streaming.chunk_right_context` | `0` |

## Tensors (360)

| name | shape | type |
|---|---|---|
| `preprocessor.fb` | `[257, 128]` | F32 |
| `encoder.pre_encode.proj.weight` | `[1024, 512]` | Q8_0 |
| `encoder.embed_norm.weight` | `[512]` | F32 |
| `encoder.embed_norm.bias` | `[512]` | F32 |
| `encoder.layers.0.norm1.weight` | `[512]` | F32 |
| `encoder.layers.0.norm1.bias` | `[512]` | F32 |
| `encoder.layers.0.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.0.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.0.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.0.norm2.weight` | `[512]` | F32 |
| `encoder.layers.0.norm2.bias` | `[512]` | F32 |
| `encoder.layers.0.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.0.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.0.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.0.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.1.norm1.weight` | `[512]` | F32 |
| `encoder.layers.1.norm1.bias` | `[512]` | F32 |
| `encoder.layers.1.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.1.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.1.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.1.norm2.weight` | `[512]` | F32 |
| `encoder.layers.1.norm2.bias` | `[512]` | F32 |
| `encoder.layers.1.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.1.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.1.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.1.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.2.norm1.weight` | `[512]` | F32 |
| `encoder.layers.2.norm1.bias` | `[512]` | F32 |
| `encoder.layers.2.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.2.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.2.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.2.norm2.weight` | `[512]` | F32 |
| `encoder.layers.2.norm2.bias` | `[512]` | F32 |
| `encoder.layers.2.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.2.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.2.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.2.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.3.norm1.weight` | `[512]` | F32 |
| `encoder.layers.3.norm1.bias` | `[512]` | F32 |
| `encoder.layers.3.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.3.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.3.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.3.norm2.weight` | `[512]` | F32 |
| `encoder.layers.3.norm2.bias` | `[512]` | F32 |
| `encoder.layers.3.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.3.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.3.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.3.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.4.norm1.weight` | `[512]` | F32 |
| `encoder.layers.4.norm1.bias` | `[512]` | F32 |
| `encoder.layers.4.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.4.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.4.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.4.norm2.weight` | `[512]` | F32 |
| `encoder.layers.4.norm2.bias` | `[512]` | F32 |
| `encoder.layers.4.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.4.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.4.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.4.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.5.norm1.weight` | `[512]` | F32 |
| `encoder.layers.5.norm1.bias` | `[512]` | F32 |
| `encoder.layers.5.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.5.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.5.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.5.norm2.weight` | `[512]` | F32 |
| `encoder.layers.5.norm2.bias` | `[512]` | F32 |
| `encoder.layers.5.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.5.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.5.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.5.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.6.norm1.weight` | `[512]` | F32 |
| `encoder.layers.6.norm1.bias` | `[512]` | F32 |
| `encoder.layers.6.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.6.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.6.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.6.norm2.weight` | `[512]` | F32 |
| `encoder.layers.6.norm2.bias` | `[512]` | F32 |
| `encoder.layers.6.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.6.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.6.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.6.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.7.norm1.weight` | `[512]` | F32 |
| `encoder.layers.7.norm1.bias` | `[512]` | F32 |
| `encoder.layers.7.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.7.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.7.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.7.norm2.weight` | `[512]` | F32 |
| `encoder.layers.7.norm2.bias` | `[512]` | F32 |
| `encoder.layers.7.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.7.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.7.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.7.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.8.norm1.weight` | `[512]` | F32 |
| `encoder.layers.8.norm1.bias` | `[512]` | F32 |
| `encoder.layers.8.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.8.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.8.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.8.norm2.weight` | `[512]` | F32 |
| `encoder.layers.8.norm2.bias` | `[512]` | F32 |
| `encoder.layers.8.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.8.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.8.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.8.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.9.norm1.weight` | `[512]` | F32 |
| `encoder.layers.9.norm1.bias` | `[512]` | F32 |
| `encoder.layers.9.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.9.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.9.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.9.norm2.weight` | `[512]` | F32 |
| `encoder.layers.9.norm2.bias` | `[512]` | F32 |
| `encoder.layers.9.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.9.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.9.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.9.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.10.norm1.weight` | `[512]` | F32 |
| `encoder.layers.10.norm1.bias` | `[512]` | F32 |
| `encoder.layers.10.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.10.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.10.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.10.norm2.weight` | `[512]` | F32 |
| `encoder.layers.10.norm2.bias` | `[512]` | F32 |
| `encoder.layers.10.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.10.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.10.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.10.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.11.norm1.weight` | `[512]` | F32 |
| `encoder.layers.11.norm1.bias` | `[512]` | F32 |
| `encoder.layers.11.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.11.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.11.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.11.norm2.weight` | `[512]` | F32 |
| `encoder.layers.11.norm2.bias` | `[512]` | F32 |
| `encoder.layers.11.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.11.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.11.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.11.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.12.norm1.weight` | `[512]` | F32 |
| `encoder.layers.12.norm1.bias` | `[512]` | F32 |
| `encoder.layers.12.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.12.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.12.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.12.norm2.weight` | `[512]` | F32 |
| `encoder.layers.12.norm2.bias` | `[512]` | F32 |
| `encoder.layers.12.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.12.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.12.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.12.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.13.norm1.weight` | `[512]` | F32 |
| `encoder.layers.13.norm1.bias` | `[512]` | F32 |
| `encoder.layers.13.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.13.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.13.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.13.norm2.weight` | `[512]` | F32 |
| `encoder.layers.13.norm2.bias` | `[512]` | F32 |
| `encoder.layers.13.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.13.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.13.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.13.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.14.norm1.weight` | `[512]` | F32 |
| `encoder.layers.14.norm1.bias` | `[512]` | F32 |
| `encoder.layers.14.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.14.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.14.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.14.norm2.weight` | `[512]` | F32 |
| `encoder.layers.14.norm2.bias` | `[512]` | F32 |
| `encoder.layers.14.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.14.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.14.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.14.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.15.norm1.weight` | `[512]` | F32 |
| `encoder.layers.15.norm1.bias` | `[512]` | F32 |
| `encoder.layers.15.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.15.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.15.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.15.norm2.weight` | `[512]` | F32 |
| `encoder.layers.15.norm2.bias` | `[512]` | F32 |
| `encoder.layers.15.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.15.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.15.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.15.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.16.norm1.weight` | `[512]` | F32 |
| `encoder.layers.16.norm1.bias` | `[512]` | F32 |
| `encoder.layers.16.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.16.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.16.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.16.norm2.weight` | `[512]` | F32 |
| `encoder.layers.16.norm2.bias` | `[512]` | F32 |
| `encoder.layers.16.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.16.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.16.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.16.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.17.norm1.weight` | `[512]` | F32 |
| `encoder.layers.17.norm1.bias` | `[512]` | F32 |
| `encoder.layers.17.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.17.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.17.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.17.norm2.weight` | `[512]` | F32 |
| `encoder.layers.17.norm2.bias` | `[512]` | F32 |
| `encoder.layers.17.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.17.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.17.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.17.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.18.norm1.weight` | `[512]` | F32 |
| `encoder.layers.18.norm1.bias` | `[512]` | F32 |
| `encoder.layers.18.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.18.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.18.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.18.norm2.weight` | `[512]` | F32 |
| `encoder.layers.18.norm2.bias` | `[512]` | F32 |
| `encoder.layers.18.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.18.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.18.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.18.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.19.norm1.weight` | `[512]` | F32 |
| `encoder.layers.19.norm1.bias` | `[512]` | F32 |
| `encoder.layers.19.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.19.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.19.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.19.norm2.weight` | `[512]` | F32 |
| `encoder.layers.19.norm2.bias` | `[512]` | F32 |
| `encoder.layers.19.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.19.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.19.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.19.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.20.norm1.weight` | `[512]` | F32 |
| `encoder.layers.20.norm1.bias` | `[512]` | F32 |
| `encoder.layers.20.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.20.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.20.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.20.norm2.weight` | `[512]` | F32 |
| `encoder.layers.20.norm2.bias` | `[512]` | F32 |
| `encoder.layers.20.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.20.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.20.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.20.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.21.norm1.weight` | `[512]` | F32 |
| `encoder.layers.21.norm1.bias` | `[512]` | F32 |
| `encoder.layers.21.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.21.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.21.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.21.norm2.weight` | `[512]` | F32 |
| `encoder.layers.21.norm2.bias` | `[512]` | F32 |
| `encoder.layers.21.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.21.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.21.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.21.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.22.norm1.weight` | `[512]` | F32 |
| `encoder.layers.22.norm1.bias` | `[512]` | F32 |
| `encoder.layers.22.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.22.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.22.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.22.norm2.weight` | `[512]` | F32 |
| `encoder.layers.22.norm2.bias` | `[512]` | F32 |
| `encoder.layers.22.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.22.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.22.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.22.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.23.norm1.weight` | `[512]` | F32 |
| `encoder.layers.23.norm1.bias` | `[512]` | F32 |
| `encoder.layers.23.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.23.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.23.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.23.norm2.weight` | `[512]` | F32 |
| `encoder.layers.23.norm2.bias` | `[512]` | F32 |
| `encoder.layers.23.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.23.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.23.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.23.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.24.norm1.weight` | `[512]` | F32 |
| `encoder.layers.24.norm1.bias` | `[512]` | F32 |
| `encoder.layers.24.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.24.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.24.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.24.norm2.weight` | `[512]` | F32 |
| `encoder.layers.24.norm2.bias` | `[512]` | F32 |
| `encoder.layers.24.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.24.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.24.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.24.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.25.norm1.weight` | `[512]` | F32 |
| `encoder.layers.25.norm1.bias` | `[512]` | F32 |
| `encoder.layers.25.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.25.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.25.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.25.norm2.weight` | `[512]` | F32 |
| `encoder.layers.25.norm2.bias` | `[512]` | F32 |
| `encoder.layers.25.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.25.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.25.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.25.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.26.norm1.weight` | `[512]` | F32 |
| `encoder.layers.26.norm1.bias` | `[512]` | F32 |
| `encoder.layers.26.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.26.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.26.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.26.norm2.weight` | `[512]` | F32 |
| `encoder.layers.26.norm2.bias` | `[512]` | F32 |
| `encoder.layers.26.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.26.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.26.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.26.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.27.norm1.weight` | `[512]` | F32 |
| `encoder.layers.27.norm1.bias` | `[512]` | F32 |
| `encoder.layers.27.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.27.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.27.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.27.norm2.weight` | `[512]` | F32 |
| `encoder.layers.27.norm2.bias` | `[512]` | F32 |
| `encoder.layers.27.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.27.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.27.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.27.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.28.norm1.weight` | `[512]` | F32 |
| `encoder.layers.28.norm1.bias` | `[512]` | F32 |
| `encoder.layers.28.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.28.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.28.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.28.norm2.weight` | `[512]` | F32 |
| `encoder.layers.28.norm2.bias` | `[512]` | F32 |
| `encoder.layers.28.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.28.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.28.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.28.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.29.norm1.weight` | `[512]` | F32 |
| `encoder.layers.29.norm1.bias` | `[512]` | F32 |
| `encoder.layers.29.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.29.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.29.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.29.norm2.weight` | `[512]` | F32 |
| `encoder.layers.29.norm2.bias` | `[512]` | F32 |
| `encoder.layers.29.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.29.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.29.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.29.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.layers.30.norm1.weight` | `[512]` | F32 |
| `encoder.layers.30.norm1.bias` | `[512]` | F32 |
| `encoder.layers.30.attn.w_qkv.weight` | `[512, 1536]` | Q8_0 |
| `encoder.layers.30.attn.out_proj.weight` | `[512, 512]` | Q8_0 |
| `encoder.layers.30.attn.out_proj.bias` | `[512]` | F32 |
| `encoder.layers.30.norm2.weight` | `[512]` | F32 |
| `encoder.layers.30.norm2.bias` | `[512]` | F32 |
| `encoder.layers.30.ffn.net.0.weight` | `[512, 2048]` | Q8_0 |
| `encoder.layers.30.ffn.net.0.bias` | `[2048]` | F32 |
| `encoder.layers.30.ffn.net.3.weight` | `[2048, 512]` | Q8_0 |
| `encoder.layers.30.ffn.net.3.bias` | `[512]` | F32 |
| `encoder.final_norm.weight` | `[512]` | F32 |
| `encoder.final_norm.bias` | `[512]` | F32 |
| `learnable_sil_emb` | `[512]` | F32 |
| `head.first_hidden_to_hidden.weight` | `[192, 192]` | F32 |
| `head.first_hidden_to_hidden.bias` | `[192]` | F32 |
| `head.single_hidden_to_spks.weight` | `[192, 8]` | F32 |
| `head.single_hidden_to_spks.bias` | `[8]` | F32 |
| `head.activity_head.0.weight` | `[192]` | F32 |
| `head.activity_head.0.bias` | `[192]` | F32 |
| `head.activity_head.1.weight` | `[192, 3]` | F32 |
| `head.activity_head.1.bias` | `[3]` | F32 |
| `encoder_proj.weight` | `[512, 192]` | Q8_0 |
| `encoder_proj.bias` | `[192]` | F32 |
| `subpixel_upsample.weight` | `[3, 192, 1536]` | F16 |
| `subpixel_upsample.bias` | `[1536]` | F32 |
