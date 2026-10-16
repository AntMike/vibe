import type { Segment } from './transcript'
import { key } from './vocabulary'

/** A misheard word as the transcript has it, and its right spelling. */
export interface Replacement {
	from: string
	to: string
}

/** Fixes taken from the AI per transcript; a longer list is more likely rewriting than fixing. */
export const MAX_REPLACEMENTS = 80

const WORD = /\p{L}[\p{L}'’]*/gu

/** Edits between `a` and `b`: letters inserted, deleted or swapped for another. */
function distance(a: string, b: string): number {
	let row = Array.from({ length: b.length + 1 }, (_, j) => j)
	for (let i = 1; i <= a.length; i++) {
		const next = [i]
		for (let j = 1; j <= b.length; j++) next[j] = Math.min(row[j] + 1, next[j - 1] + 1, row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1))
		row = next
	}
	return row[b.length]
}

/** `segments` with each text run through `fix`; a line it leaves alone stays the same object. */
function mapText(segments: Segment[], fix: (text: string) => string): Segment[] {
	return segments.map((segment) => {
		const text = fix(segment.text)
		return text === segment.text ? segment : { ...segment, text }
	})
}

/**
 * Words one sound away from exactly one known term, spelled as that term: стімейти as естімейти,
 * плікап as Клікап. Of the term's forms, the one closest to the word as written wins, so its script
 * and ending stay.
 */
// ponytail: 6 letters and distance 1 are the guard against everyday words; tighten if one gets replaced.
export function fixSpelling(segments: Segment[], families: string[][]): Segment[] {
	const known = families.map((family) => ({ family, keys: [...new Set(family.map(key))] }))
	if (known.length === 0) return segments
	const fix = (word: string) => {
		if (word.replace(/['’]/g, '').length < 6) return word
		const sound = key(word)
		if (known.some(({ keys }) => keys.includes(sound))) return word
		const near = known.filter(({ keys }) => keys.some((other) => distance(other, sound) === 1))
		if (near.length !== 1) return word
		return near[0].family.reduce((best, form) => (distance(form, word) < distance(best, word) ? form : best))
	}
	return mapText(segments, (text) => text.replace(WORD, fix))
}

/** The AI's fixes, as whole words in any case; a pair that changes nothing is skipped. */
export function applyReplacements(segments: Segment[], pairs: Replacement[]): Segment[] {
	const spelling = new Map(
		pairs
			.filter(({ from, to }) => from.trim() && to.trim() && from.toLowerCase() !== to.toLowerCase())
			.slice(0, MAX_REPLACEMENTS)
			.map(({ from, to }) => [from.toLowerCase(), to]),
	)
	if (spelling.size === 0) return segments
	// Longest first, so a phrase wins over a word inside it. `\b` knows no Cyrillic, hence the lookarounds.
	const words = [...spelling.keys()].sort((a, b) => b.length - a.length).map((from) => from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
	const pattern = new RegExp(`(?<![\\p{L}\\p{N}])(?:${words.join('|')})(?![\\p{L}\\p{N}])`, 'giu')
	return mapText(segments, (text) => text.replace(pattern, (word) => spelling.get(word.toLowerCase()) ?? word))
}
