import { invoke } from '@tauri-apps/api/core'
import type { AiConnection } from '~/lib/ai'
import type { CallSpeakerTurn } from '~/lib/call-speakers'
import * as config from '~/lib/config'
import { dueForTuning, refineDraft, tuneVocabulary, type HintsSettings } from '~/lib/hints'
import { detectedLanguages, type LanguageMode } from '~/lib/languages'
import type { ModelMetadata } from '~/lib/model'
import type { Segment, Transcript } from '~/lib/transcript'
import { dropBlocked, knownFamilies, learn, loadVocabulary, saveVocabulary, type Vocabulary } from '~/lib/vocabulary'

export interface HintsRun {
	settings: HintsSettings
	connection: AiConnection
	mainModelPath: string
	/** Where the learned vocabulary lives, when `settings.learn` is on. */
	vocabularyPath?: string
	/** Load a model on the server the way the queue loads the main one. */
	loadModel: (path: string) => Promise<void>
	isAborted: () => boolean
	onWarning: (message: string) => void
	/** The main pass's language, so a draft or extra pass doesn't fall back to English. */
	lang?: string
	/** Language codes auto-detection may choose between; empty is any. */
	languages?: string[]
}

// Measured 13.3% vs 13.8% WER on the Oct 27 grid; within noise, kept by choice.
export const TURN_LOUDNESS_DBFS = -20

/** The speed pass `p` decodes at (0 is the main pass): half at normal speed, a quarter each slower and faster. */
export function passSpeed(p: number): number {
	return [1.0, 0.95, 1.05, 1.0][p % 4]
}

/** Participants, most talkative first. */
export function callNames(turns: CallSpeakerTurn[] | undefined): string[] {
	const seconds = new Map<string, number>()
	for (const turn of turns ?? []) seconds.set(turn.name, (seconds.get(turn.name) ?? 0) + turn.end - turn.start)
	return [...seconds].sort((a, b) => b[1] - a[1]).map(([name]) => name)
}

/**
 * One fast pass with the draft model, decoded `runs` times over altered audio, leaning toward the
 * `hotwords` it already knows. Leaves the draft model loaded.
 */
async function draftSegments(path: string, draftModel: string, swap: boolean, run: HintsRun, runs = 1, hotwords: string[] = [], speed = 1): Promise<Segment[]> {
	if (swap) await run.loadModel(draftModel)
	const metadata = await invoke<ModelMetadata>('get_model_metadata', { modelPath: draftModel })
	const vad = metadata.capabilities.requires_vad ? { vad_model: `${await invoke<string>('get_models_folder')}/${config.vadModelFilename}` } : {}
	const draft = await invoke<Transcript>('transcribe', {
		options: {
			path,
			turn_loudness: TURN_LOUDNESS_DBFS,
			...vad,
			...(run.lang ? { lang: run.lang } : {}),
			...(run.languages?.length ? { languages: run.languages } : {}),
			...(runs > 1 ? { draft_runs: runs } : {}),
			...(hotwords.length > 0 ? { hotwords } : {}),
			...(speed !== 1 ? { speed } : {}),
		},
	})
	return draft.segments.filter((segment) => segment.text.trim())
}

export interface Hints {
	/** What the draft pass heard, to merge with the main pass; see `mergeRuns`. */
	draft: Segment[]
	/** The known terms as families, main spelling first; see `knownFamilies`. */
	families: string[][]
}

/**
 * One more transcription of the file per model, to merge with the main pass. The draft pass counts
 * when its model is listed, so a model never runs twice. Pass `p` (1 for the first listed) decodes at
 * `passSpeed(p)`, so the main model may run again unless that is its normal speed. A failed pass only
 * warns. Leaves the main model loaded.
 */
export async function extraPasses(
	path: string,
	models: string[],
	draft: { model: string | null; segments: Segment[] },
	run: Pick<HintsRun, 'mainModelPath' | 'loadModel' | 'isAborted' | 'onWarning' | 'lang' | 'languages'>,
): Promise<Segment[][]> {
	const runs: Segment[][] = draft.segments.length > 0 ? [draft.segments] : []
	let loaded = run.mainModelPath
	for (const [index, model] of [...new Set(models)].entries()) {
		const speed = passSpeed(index + 1)
		if ((model === run.mainModelPath && speed === 1) || model === draft.model || run.isAborted()) continue
		try {
			const swap = model !== loaded
			loaded = model
			runs.push(await draftSegments(path, model, swap, run as HintsRun, 1, [], speed))
		} catch (error) {
			if (!run.isAborted()) run.onWarning(message(error))
		}
	}
	if (loaded !== run.mainModelPath) await run.loadModel(run.mainModelPath)
	return runs
}

/**
 * The languages every pass of a file is kept to, and each speech window as the detector heard it
 * (`language` unset where it was unsure). Detecting runs the loaded main model over each speech
 * window without decoding, so it costs no model swap; a model that can't detect, a failed detection,
 * or one that heard nothing falls back to the list. With `turns` it detects even for a fixed list,
 * over the call's turns only, so each window is one person's. Never throws.
 */
export async function spokenLanguages(
	path: string,
	mode: LanguageMode,
	list: string[],
	canDetect: boolean,
	onWarning: (message: string) => void,
	turns?: CallSpeakerTurn[],
): Promise<{ languages: string[]; heard: Segment[] }> {
	if ((mode === 'list' && !turns?.length) || !canDetect) return { languages: list, heard: [] }
	try {
		const among = mode !== 'detect' ? list : []
		const detection = await invoke<Transcript>('transcribe', {
			options: {
				path,
				detect_language: true,
				raw_languages: true,
				vad_model: `${await invoke<string>('get_models_folder')}/${config.vadModelFilename}`,
				...(among.length ? { languages: among } : {}),
				...(turns?.length ? { windows: turns.filter((turn) => turn.end > turn.start).map((turn) => [turn.start, turn.end]) } : {}),
			},
		})
		const detected = mode === 'list' ? [] : detectedLanguages(detection.segments)
		return { languages: detected.length > 0 ? detected : list, heard: detection.segments }
	} catch (error) {
		onWarning(message(error))
	}
	return { languages: list, heard: [] }
}

/** A missing or broken vocabulary only warns: a broken one is left alone for the user to fix. */
async function readVocabulary(run: HintsRun): Promise<Vocabulary | null> {
	try {
		return await loadVocabulary(run.vocabularyPath!)
	} catch (error) {
		run.onWarning(message(error))
		return null
	}
}

async function writeVocabulary(run: HintsRun, vocabulary: Vocabulary) {
	try {
		await saveVocabulary(run.vocabularyPath!, vocabulary)
	} catch (error) {
		run.onWarning(message(error))
	}
}

function message(error: unknown) {
	return String(error instanceof Error ? error.message : ((error as { message?: string })?.message ?? error))
}

/**
 * The hints for one file. A failed draft or AI step only warns: the file is still transcribed, with
 * fewer runs to vote with and fewer terms to fix spelling by. Failing to load the main model back does throw, since the
 * transcription can't run without it.
 */
// ponytail: swaps models twice per file; draft the whole queue first if batches of calls get slow.
export async function hintOptions(path: string, turns: CallSpeakerTurn[] | undefined, run: HintsRun): Promise<Hints> {
	const names = run.settings.callNames ? callNames(turns) : []
	let vocabulary = run.settings.learn && run.vocabularyPath ? await readVocabulary(run) : null
	let tuned = false
	// Known terms steer the draft's recognition in every spelling; they are learned only from what the AI finds below.
	const families = vocabulary ? knownFamilies(vocabulary) : []
	let terms: string[] = []
	let draft: Segment[] = []
	const draftModel = run.settings.draftModelPath
	if (draftModel) {
		const swap = draftModel !== run.mainModelPath
		try {
			draft = await draftSegments(path, draftModel, swap, run, run.settings.draftRuns, families.flat())
		} catch (error) {
			if (!run.isAborted()) run.onWarning(message(error))
		}
		// Refined while the small draft model holds the GPU: a local LLM and Whisper large together would not fit 8 GB.
		if (draft.length > 0 && !run.isAborted()) {
			try {
				terms = await refineDraft(
					draft.map((segment) => segment.text.trim()),
					names,
					run.settings,
					run.connection,
					families.map((family) => family.join(' / ')),
				)
			} catch (error) {
				run.onWarning(message(error))
			}
		}
		// Also while the draft model holds the GPU, and only with the AI refiner: the words one learns nothing.
		if (vocabulary && run.settings.refiner !== 'words' && dueForTuning(vocabulary, run.settings) && !run.isAborted()) {
			try {
				vocabulary = await tuneVocabulary(vocabulary, run.settings, run.connection)
				tuned = true
			} catch (error) {
				run.onWarning(message(error))
			}
		}
		if (swap) await run.loadModel(run.mainModelPath)
	}
	if (vocabulary) {
		terms = dropBlocked(vocabulary, terms)
		// Only the AI's terms are learned: the capitalized-words refiner keeps too many mishearings.
		if (run.settings.refiner !== 'words' && (terms.length > 0 || tuned))
			await writeVocabulary(run, terms.length > 0 ? learn(vocabulary, terms) : vocabulary)
	}
	// No Whisper prompt: listed names and terms made it hear them where they weren't said ("все, що" as
	// "Сашо"; Oct 15 pieces 32% WER against 27% without). The terms fix spelling after the fact instead.
	return { draft, families }
}
