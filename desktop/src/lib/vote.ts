import type { Segment } from './transcript'

/**
 * Merging several transcriptions of one file. The main pass keeps its lines, timing and speakers;
 * the other runs only correct its words: one other run replaces the lines Whisper got stuck on,
 * and from two on every word is put to a vote.
 */

/** Whisper, once it starts repeating a line, keeps repeating it for the rest of its window. */
const LOOP_MIN_REPEATS = 3

const key = (text: string) => text.replace(/[^\p{L}\p{N}]+/gu, '').toLowerCase()

/**
 * The main pass with its loops replaced from another run: a line repeated three or more times in
 * a row is Whisper stuck, not someone speaking, so that stretch is taken from what the other run
 * heard there instead — nothing, when it heard nothing. The lines get the speaker they replace.
 */
export function swapLoops(segments: Segment[], other: Segment[]): Segment[] {
	const merged: Segment[] = []
	for (let i = 0; i < segments.length; ) {
		let j = i + 1
		while (j < segments.length && key(segments[j].text) === key(segments[i].text)) j += 1
		if (j - i < LOOP_MIN_REPEATS) {
			merged.push(...segments.slice(i, j))
		} else {
			const { start, speaker } = segments[i]
			const stop = segments[j - 1].stop
			for (const line of other) {
				const middle = (line.start + line.stop) / 2
				if (middle >= start && middle < stop) merged.push(speaker === undefined ? line : { ...line, speaker })
			}
		}
		i = j
	}
	return merged
}

const words = (text: string) => text.trim().split(/\s+/).filter(Boolean)

/** Every word of a run with a time: spread evenly over its line. */
function timedWords(segments: Segment[]): { at: number; word: string }[] {
	return segments.flatMap((segment) => {
		const list = words(segment.text)
		const step = (segment.stop - segment.start) / list.length
		return list.map((word, index) => ({ at: segment.start + step * (index + 0.5), word }))
	})
}

/** One run's words laid against the main line: what it has for each main word (null: nothing), and what it adds after each. */
interface Alignment {
	lead: string[]
	at: (string | null)[]
	after: string[][]
}

/** Longest common subsequence of the two word lists, as the matched index pairs in order. */
function matches(a: string[], b: string[]): [number, number][] {
	const length: number[][] = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0))
	for (let i = a.length - 1; i >= 0; i -= 1) {
		for (let j = b.length - 1; j >= 0; j -= 1) {
			length[i][j] = key(a[i]) === key(b[j]) ? length[i + 1][j + 1] + 1 : Math.max(length[i + 1][j], length[i][j + 1])
		}
	}
	const pairs: [number, number][] = []
	for (let i = 0, j = 0; i < a.length && j < b.length; ) {
		if (key(a[i]) === key(b[j])) pairs.push([i++, j++])
		else if (length[i + 1][j] >= length[i][j + 1]) i += 1
		else j += 1
	}
	return pairs
}

/** The unmatched words between matches are paired up in order; what one side has over is a deletion or an insertion. */
function align(main: string[], other: string[]): Alignment {
	const alignment: Alignment = { lead: [], at: new Array<string | null>(main.length).fill(null), after: main.map(() => []) }
	let [i, j] = [0, 0]
	for (const [mi, oj] of [...matches(main, other), [main.length, other.length] as [number, number]]) {
		const gap = Math.min(mi - i, oj - j)
		for (let k = 0; k < gap; k += 1) alignment.at[i + k] = other[j + k]
		const extra = other.slice(j + gap, oj)
		if (extra.length > 0) {
			if (i + gap === 0) alignment.lead.push(...extra)
			else alignment.after[i + gap - 1].push(...extra)
		}
		if (mi < main.length) alignment.at[mi] = other[oj]
		;[i, j] = [mi + 1, oj + 1]
	}
	return alignment
}

/** The choice most runs made, when that is a strict majority of all of them; otherwise the main's. */
function majority<T>(main: T, others: T[], id: (choice: T) => string): T {
	const needed = Math.floor((others.length + 1) / 2) + 1
	const counts = new Map<string, { choice: T; votes: number }>()
	for (const choice of [main, ...others]) {
		const entry = counts.get(id(choice))
		if (entry) entry.votes += 1
		else counts.set(id(choice), { choice, votes: 1 })
	}
	const winner = [...counts.values()].find((entry) => entry.votes >= needed)
	return winner ? winner.choice : main
}

const wordId = (word: string | null) => (word === null ? '\0' : key(word))
const listId = (list: string[]) => list.map(key).join(' ')

/**
 * Every word of the main pass put to a vote with the other runs: a word most runs replace is
 * replaced, one most runs leave out is left out, and what most runs add is added. Each main line
 * is voted on its own, against the other runs' words timed inside it.
 */
// ponytail: words are timed by spreading a line's words evenly, so a vote near a line boundary can
// miss by a word; word timestamps from the engines would fix that.
export function voteWords(main: Segment[], runs: Segment[][]): Segment[] {
	const timed = runs.map(timedWords)
	return main.map((segment, index) => {
		const from = index === 0 ? -Infinity : (main[index - 1].stop + segment.start) / 2
		const to = index === main.length - 1 ? Infinity : (segment.stop + main[index + 1].start) / 2
		const line = words(segment.text)
		if (line.length === 0) return segment
		const alignments = timed.map((run) =>
			align(
				line,
				run.filter(({ at }) => at >= from && at < to).map(({ word }) => word),
			),
		)
		const voted: string[] = [
			...majority(
				[],
				alignments.map((a) => a.lead),
				listId,
			),
		]
		line.forEach((word, k) => {
			const choice = majority<string | null>(
				word,
				alignments.map((a) => a.at[k]),
				wordId,
			)
			if (choice !== null) voted.push(choice)
			voted.push(
				...majority(
					[],
					alignments.map((a) => a.after[k]),
					listId,
				),
			)
		})
		const text = voted.join(' ')
		return text === segment.text.trim() ? segment : { ...segment, text }
	})
}

/** The main pass corrected by the other runs; see the module note. */
export function mergeRuns(main: Segment[], runs: Segment[][]): Segment[] {
	const others = runs.filter((run) => run.length > 0)
	if (others.length === 0) return main
	const unstuck = swapLoops(main, others[0])
	return others.length >= 2 ? voteWords(unstuck, others) : unstuck
}
