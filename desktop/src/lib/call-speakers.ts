import type { Segment, SpeakerNames } from '~/lib/transcript'

/** A stretch of a recording during which the call app (a Slack huddle, for now) showed `name` talking, in seconds. */
export interface CallSpeakerTurn {
	start: number
	end: number
	name: string
}

export function parseCallSpeakers(value: unknown): CallSpeakerTurn[] | undefined {
	if (!Array.isArray(value)) return undefined
	const turns = value.filter(
		(turn): turn is CallSpeakerTurn =>
			typeof turn === 'object' &&
			turn !== null &&
			typeof turn.start === 'number' &&
			typeof turn.end === 'number' &&
			typeof turn.name === 'string' &&
			turn.name.trim() !== '',
	)
	return turns.length > 0 ? turns : undefined
}

/** A sentence the call can place needs its best match to cover at least this share of it. */
const MIN_COVERAGE = 0.2
/** Seconds the call app's active-speaker ring can trail the voice: Slack huddles ran 0.2-2.2 s behind. */
// ponytail: calibration knobs, measured on the Oct 2026 huddles; tune here if voices and the call disagree.
const MAX_LAG = 2
/** Share of a sentence the call must give its pick away from the lagging end of a turn, to overrule the voice. */
const SURE = 0.75
/** How many times more of the diarized speaker's talk the voice's person must hold than the call's pick. */
const VOICE_ODDS = 5

/** Seconds each name was shown talking during a segment (segment times are centiseconds). */
function overlapByName(segment: Segment, turns: CallSpeakerTurn[], lag = 0): Map<string, number> {
	const start = segment.start / 100
	const stop = segment.stop / 100 + lag
	const byName = new Map<string, number>()
	for (const turn of turns) {
		const shared = Math.min(stop, turn.end) - Math.max(start, turn.start)
		if (shared > 0) byName.set(turn.name, (byName.get(turn.name) ?? 0) + shared)
	}
	return byName
}

/** Seconds `name` was shown talking during a segment outside the last `MAX_LAG` of each turn, where the ring may still show them after they stopped. */
function sureSeconds(segment: Segment, turns: CallSpeakerTurn[], name: string): number {
	return turns
		.filter((turn) => turn.name === name)
		.reduce((sum, turn) => sum + Math.max(0, Math.min(segment.stop / 100, turn.end - MAX_LAG) - Math.max(segment.start / 100, turn.start)), 0)
}

/** The name holding over half of the talk, if it also covers `minSeconds`. */
function majority(byName: Map<string, number>, minSeconds = 0): string | undefined {
	if (byName.size === 0) return undefined
	const total = [...byName.values()].reduce((sum, seconds) => sum + seconds, 0)
	const [name, seconds] = [...byName].reduce((best, entry) => (entry[1] > best[1] ? entry : best))
	return seconds > total / 2 && seconds >= minSeconds ? name : undefined
}

/**
 * Give every sentence the participant the call app showed talking over most of it, each person
 * their own speaker — so a call tells apart more people than diarization can, and works without
 * it. A sentence the call can't place keeps its diarized speaker, named after whoever that speaker
 * matched most overall. The ring lags the voice, so at a handover the voice wins: a sentence the
 * call is unsure of goes to its diarized speaker's person, when that person is there within the
 * lag and far likelier for that voice than the call's pick.
 */
// ponytail: O(segments × turns) scan; a sweep over both sorted lists if hour-long calls get slow.
export function speakersFromCall(segments: Segment[], turns: CallSpeakerTurn[]): { segments: Segment[]; speakerNames: SpeakerNames } {
	const overlaps = segments.map((segment) => overlapByName(segment, turns))
	const bySpeaker = new Map<number, Map<string, number>>()
	// Seconds each diarized speaker talked: a name must cover half of it, not just beat the other
	// names — a Teams call shows only you, and would otherwise name everyone who ever talked over you.
	const spoken = new Map<number, number>()
	segments.forEach((segment, i) => {
		if (segment.speaker == null) return
		const total = bySpeaker.get(segment.speaker) ?? new Map<string, number>()
		for (const [name, seconds] of overlaps[i]) total.set(name, (total.get(name) ?? 0) + seconds)
		bySpeaker.set(segment.speaker, total)
		spoken.set(segment.speaker, (spoken.get(segment.speaker) ?? 0) + (segment.stop - segment.start) / 100)
	})

	// One index per person and per diarized speaker nobody could be matched to, in order of appearance.
	const keys: string[] = []
	const speakerNames: SpeakerNames = {}
	const indexOf = (key: string, name?: string) => {
		let index = keys.indexOf(key)
		if (index < 0) {
			index = keys.push(key) - 1
			if (name) speakerNames[index] = name
		}
		return index
	}
	const voices = new Map([...bySpeaker].map(([speaker, byName]) => [speaker, majority(byName, spoken.get(speaker)! / 2)]))
	const named = segments.map((segment, i) => {
		const seconds = (segment.stop - segment.start) / 100
		const fromCall = majority(overlaps[i], seconds * MIN_COVERAGE)
		const voice = segment.speaker != null ? voices.get(segment.speaker) : undefined
		const voiceWins =
			fromCall != null &&
			voice != null &&
			voice !== fromCall &&
			sureSeconds(segment, turns, fromCall) < seconds * SURE &&
			overlapByName(segment, turns, MAX_LAG).has(voice) &&
			bySpeaker.get(segment.speaker!)!.get(voice)! >= VOICE_ODDS * (bySpeaker.get(segment.speaker!)!.get(fromCall) ?? 0)
		const name = voiceWins ? voice : (fromCall ?? voice)
		if (name) return { ...segment, speaker: indexOf(`name:${name}`, name) }
		if (segment.speaker != null) return { ...segment, speaker: indexOf(`speaker:${segment.speaker}`) }
		return segment
	})
	return { segments: named, speakerNames }
}

/**
 * The names in a transcript that the user typed: those its call's own turns would not have given.
 * A name the call gave is the call's to give again, so it must not be carried onto a re-transcription,
 * where the earlier speaker it sat on may have swallowed someone else's turn.
 */
export function typedSpeakerNames(segments: Segment[], names: SpeakerNames, turns: CallSpeakerTurn[] | undefined): SpeakerNames {
	if (!turns?.length) return names
	const call = speakersFromCall(segments, turns)
	const typed: SpeakerNames = {}
	for (const [speaker, name] of Object.entries(names)) {
		const fromCall = segments.some((segment, i) => segment.speaker === Number(speaker) && call.speakerNames[call.segments[i].speaker!] === name)
		if (!fromCall) typed[Number(speaker)] = name
	}
	return typed
}

/**
 * Names from an earlier transcript of the same audio, moved onto the speakers of a new one: each new
 * speaker takes the name of the earlier speaker who said most of its lines, when that one had a name.
 */
export function carrySpeakerNames(before: Segment[], names: SpeakerNames, after: Segment[]): SpeakerNames {
	// Unnamed earlier speakers count too (as ''), so they can outvote a name instead of handing it out.
	const turns = before.flatMap((segment) =>
		segment.speaker == null ? [] : [{ start: segment.start / 100, end: segment.stop / 100, name: names[segment.speaker]?.trim() ?? '' }],
	)
	const bySpeaker = new Map<number, Map<string, number>>()
	for (const segment of after) {
		if (segment.speaker == null) continue
		const total = bySpeaker.get(segment.speaker) ?? new Map<string, number>()
		for (const [name, seconds] of overlapByName(segment, turns)) total.set(name, (total.get(name) ?? 0) + seconds)
		bySpeaker.set(segment.speaker, total)
	}
	const carried: SpeakerNames = {}
	for (const [speaker, byName] of bySpeaker) {
		const name = majority(byName)
		if (name) carried[speaker] = name
	}
	return carried
}
