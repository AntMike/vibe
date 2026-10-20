import { invoke } from '@tauri-apps/api/core'
import type { CallSpeakerTurn } from './call-speakers'
import type { Segment, SpeakerNames, Transcript } from './transcript'

/**
 * Whisper cannot tell Ukrainian from Russian turn by turn, so a call settles on one of them for
 * everyone (see `CONFUSED_FAMILIES` on the server). A speaker who talks the other one is decoded
 * again in the language picked for them, here: their lines' stretches of audio, nothing else.
 */

/** Lines are cut tight to the words; this much either side keeps the first and last sound. */
const PAD_SECONDS = 0.3

/** The language most of `speaker`'s talk was decoded in, by time; undefined when none says. */
export function speakerLanguage(segments: Segment[], speaker: number): string | undefined {
	const seconds = new Map<string, number>()
	for (const segment of segments) {
		if (segment.speaker !== speaker || !segment.language) continue
		seconds.set(segment.language, (seconds.get(segment.language) ?? 0) + segment.stop - segment.start)
	}
	return [...seconds].sort((a, b) => b[1] - a[1])[0]?.[0]
}

/**
 * The stretches `speaker` talks, as `[start, end]` seconds: each run of their lines unbroken by
 * anyone else's, padded a little but never into a neighbour's line.
 */
export function speakerWindows(segments: Segment[], speaker: number): [number, number][] {
	const sorted = [...segments].sort((a, b) => a.start - b.start)
	const windows: [number, number][] = []
	let run: Segment[] = []
	const close = (next?: Segment) => {
		if (run.length === 0) return
		const before = sorted[sorted.indexOf(run[0]) - 1]
		const first = run[0].start / 100
		const last = Math.max(...run.map((segment) => segment.stop)) / 100
		const start = Math.min(first, Math.max(first - PAD_SECONDS, before ? before.stop / 100 : 0, 0))
		const end = Math.max(last, Math.min(last + PAD_SECONDS, next ? next.start / 100 : Infinity))
		windows.push([start, end])
		run = []
	}
	for (const segment of sorted) {
		if (segment.speaker === speaker) run.push(segment)
		else close(segment)
	}
	close()
	return windows
}

/**
 * The transcript with `speaker`'s lines swapped for `heard`, the same stretches decoded again. A
 * stretch the new decode came back empty for keeps its old lines rather than losing them.
 */
export function withSpeakerLines(segments: Segment[], speaker: number, heard: Segment[]): Segment[] {
	const windows = speakerWindows(segments, speaker)
	const inside = (segment: Segment, [start, end]: [number, number]) => segment.start / 100 >= start - 0.01 && segment.start / 100 < end
	const lines = heard.filter((segment) => segment.text.trim())
	const kept = windows.flatMap((window) => {
		const fresh = lines.filter((segment) => inside(segment, window))
		const old = segments.filter((segment) => segment.speaker === speaker && inside(segment, window))
		return fresh.length > 0 ? fresh.map((segment) => ({ ...segment, speaker })) : old
	})
	return [...segments.filter((segment) => segment.speaker !== speaker), ...kept].sort((a, b) => a.start - b.start)
}

/** The named speakers with a language remembered for them whose lines were decoded in another. */
export function speakersToRedo(segments: Segment[], names: SpeakerNames | undefined, remembered: Record<string, string>) {
	return Object.entries(names ?? {}).flatMap(([index, name]) => {
		const speaker = Number(index)
		const language = remembered[name.trim()]
		const lines = segments.filter((segment) => segment.speaker === speaker)
		return language && lines.some((segment) => segment.language !== language) ? [{ speaker, language }] : []
	})
}

/**
 * `speaker`'s stretches of `path` decoded again in `language` with the loaded model; merge them in
 * with `withSpeakerLines`. `options` are the run's own (model options, VAD); diarization is left out,
 * the stretches stand in for its turns.
 */
export async function decodeSpeaker(path: string, segments: Segment[], speaker: number, language: string, options: Record<string, unknown>) {
	const windows = speakerWindows(segments, speaker)
	if (windows.length === 0) return []
	const transcript = await invoke<Transcript>('transcribe', {
		options: { ...options, diarize_model: undefined, languages: undefined, path, lang: language, windows },
	})
	return transcript.segments
}

/** The languages Whisper mixes up turn by turn; the server's `CONFUSED_FAMILIES`. */
const CONFUSED = ['uk', 'ru', 'be']
/** Someone heard this much in another of them is worth asking about; less is the detector misjudging. */
const DOUBT_SHARE = 0.25
/** Too little talk to judge by. */
const MIN_DOUBT_SECONDS = 5

/** Whether the spoken languages leave room for the mix-up: two of `CONFUSED`, or any language at all. */
export function canConfuse(spoken: string[]) {
	return spoken.length === 0 || spoken.filter((language) => CONFUSED.includes(language)).length > 1
}

export interface LanguageDoubt {
	name: string
	/** Seconds the detector heard them in each of the confusable languages. */
	seconds: Record<string, number>
}

export interface LanguageCheck {
	/** The one the recording settles on: what everyone is decoded in unless picked otherwise. */
	language: string
	doubts: LanguageDoubt[]
}

/**
 * The people a raw detection pass heard in the recording's language too seldom to trust it for them,
 * leaving out those with a language already `remembered`. Null when nobody is in doubt.
 */
export function languageDoubts(heard: Segment[], turns: CallSpeakerTurn[], remembered: Record<string, string>): LanguageCheck | null {
	const byName = new Map<string, Record<string, number>>()
	const overall: Record<string, number> = {}
	for (const window of heard) {
		if (!window.language || !CONFUSED.includes(window.language)) continue
		const [start, stop] = [window.start / 100, window.stop / 100]
		const overlap = (turn: CallSpeakerTurn) => Math.min(stop, turn.end) - Math.max(start, turn.start)
		const turn = turns.reduce<CallSpeakerTurn | undefined>((best, turn) => (overlap(turn) > (best ? overlap(best) : 0) ? turn : best), undefined)
		if (!turn) continue
		const seconds = byName.get(turn.name) ?? {}
		seconds[window.language] = (seconds[window.language] ?? 0) + stop - start
		overall[window.language] = (overall[window.language] ?? 0) + stop - start
		byName.set(turn.name, seconds)
	}
	const language = Object.entries(overall).sort((a, b) => b[1] - a[1])[0]?.[0]
	if (!language) return null
	const doubts = [...byName]
		.filter(([name, seconds]) => {
			const total = Object.values(seconds).reduce((sum, value) => sum + value, 0)
			return !remembered[name.trim()] && total >= MIN_DOUBT_SECONDS && 1 - (seconds[language] ?? 0) / total >= DOUBT_SHARE
		})
		.map(([name, seconds]) => ({ name, seconds }))
	return doubts.length > 0 ? { language, doubts } : null
}
