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

/** Seconds each name was shown talking during a segment (segment times are centiseconds). */
function overlapByName(segment: Segment, turns: CallSpeakerTurn[]): Map<string, number> {
	const start = segment.start / 100
	const stop = segment.stop / 100
	const byName = new Map<string, number>()
	for (const turn of turns) {
		const shared = Math.min(stop, turn.end) - Math.max(start, turn.start)
		if (shared > 0) byName.set(turn.name, (byName.get(turn.name) ?? 0) + shared)
	}
	return byName
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
 * matched most overall.
 */
// ponytail: O(segments × turns) scan; a sweep over both sorted lists if hour-long calls get slow.
export function speakersFromCall(segments: Segment[], turns: CallSpeakerTurn[]): { segments: Segment[]; speakerNames: SpeakerNames } {
	const overlaps = segments.map((segment) => overlapByName(segment, turns))
	const bySpeaker = new Map<number, Map<string, number>>()
	segments.forEach((segment, i) => {
		if (segment.speaker == null) return
		const total = bySpeaker.get(segment.speaker) ?? new Map<string, number>()
		for (const [name, seconds] of overlaps[i]) total.set(name, (total.get(name) ?? 0) + seconds)
		bySpeaker.set(segment.speaker, total)
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
	const named = segments.map((segment, i) => {
		const minSeconds = ((segment.stop - segment.start) / 100) * MIN_COVERAGE
		const name = majority(overlaps[i], minSeconds) ?? (segment.speaker != null ? majority(bySpeaker.get(segment.speaker)!) : undefined)
		if (name) return { ...segment, speaker: indexOf(`name:${name}`, name) }
		if (segment.speaker != null) return { ...segment, speaker: indexOf(`speaker:${segment.speaker}`) }
		return segment
	})
	return { segments: named, speakerNames }
}
