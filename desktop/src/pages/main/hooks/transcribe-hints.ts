import { invoke } from '@tauri-apps/api/core'
import type { AiConnection } from '~/lib/ai'
import type { CallSpeakerTurn } from '~/lib/call-speakers'
import * as config from '~/lib/config'
import { buildPrompt, refineDraft, type HintsSettings } from '~/lib/hints'
import type { ModelMetadata } from '~/lib/model'
import type { Transcript } from '~/lib/transcript'

export interface HintsRun {
	settings: HintsSettings
	connection: AiConnection
	/** The prompt the user typed in settings; the glossary goes after it. */
	userPrompt?: string
	mainModelPath: string
	/** Load a model on the server the way the queue loads the main one. */
	loadModel: (path: string) => Promise<void>
	isAborted: () => boolean
	onWarning: (message: string) => void
}

/** Participants, most talkative first. */
function callNames(turns: CallSpeakerTurn[] | undefined): string[] {
	const seconds = new Map<string, number>()
	for (const turn of turns ?? []) seconds.set(turn.name, (seconds.get(turn.name) ?? 0) + turn.end - turn.start)
	return [...seconds].sort((a, b) => b[1] - a[1]).map(([name]) => name)
}

/** One fast pass with the draft model, its lines. Leaves the draft model loaded. */
async function draftLines(path: string, draftModel: string, swap: boolean, run: HintsRun): Promise<string[]> {
	if (swap) await run.loadModel(draftModel)
	const metadata = await invoke<ModelMetadata>('get_model_metadata', { modelPath: draftModel })
	const vad = metadata.capabilities.requires_vad ? { vad_model: `${await invoke<string>('get_models_folder')}/${config.vadModelFilename}` } : {}
	const draft = await invoke<Transcript>('transcribe', { options: { path, ...vad } })
	return draft.segments.map((segment) => segment.text.trim()).filter(Boolean)
}

function message(error: unknown) {
	return String(error instanceof Error ? error.message : ((error as { message?: string })?.message ?? error))
}

/**
 * The prompt options for one file, or null when there is nothing to hint. A failed draft or AI step
 * only warns: the file is still transcribed, with whatever names the call gave. Failing to load the
 * main model back does throw, since the transcription can't run without it.
 */
// ponytail: swaps models twice per file; draft the whole queue first if batches of calls get slow.
export async function hintOptions(path: string, turns: CallSpeakerTurn[] | undefined, run: HintsRun) {
	const names = run.settings.callNames ? callNames(turns) : []
	let terms: string[] = []
	const draftModel = run.settings.draftModelPath
	if (draftModel) {
		const swap = draftModel !== run.mainModelPath
		let lines: string[] = []
		try {
			lines = await draftLines(path, draftModel, swap, run)
		} catch (error) {
			if (!run.isAborted()) run.onWarning(message(error))
		}
		// Refined while the small draft model holds the GPU: a local LLM and Whisper large together would not fit 8 GB.
		if (lines.length > 0 && !run.isAborted()) {
			try {
				terms = await refineDraft(lines, names, run.settings, {
					connection: run.connection,
					askCli: (command, prompt) => invoke<string>('ask_cli', { command, prompt }),
				})
			} catch (error) {
				run.onWarning(message(error))
			}
		}
		if (swap) await run.loadModel(run.mainModelPath)
	}
	const prompt = buildPrompt([...names, ...terms], run.userPrompt)
	if (!prompt || prompt === run.userPrompt?.trim()) return null
	return { init_prompt: prompt, carry_prompt: true }
}
