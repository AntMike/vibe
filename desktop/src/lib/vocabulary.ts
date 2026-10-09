import * as pathApi from '@tauri-apps/api/path'
import * as fs from '@tauri-apps/plugin-fs'
import { transcriptsFolder } from './transcripts-store'

/**
 * Terms learned from recordings. Every term the AI picks out of a draft is counted; one picked in
 * two recordings is known, and the AI is told its spelling from then on. A term that stops coming
 * up is forgotten — a mishearing seldom comes back. The file sits next to the transcripts so it
 * can be edited by hand: `pinned` terms are always used, `blocked` ones never.
 */
export interface LearnedTerm {
	/** Recordings it was picked in. */
	seen: number
	/** The `recordings` count when it was last picked. */
	last: number
}

export interface Vocabulary {
	version: 1
	/** Recordings learned from so far. */
	recordings: number
	pinned: string[]
	blocked: string[]
	terms: Record<string, LearnedTerm>
}

export const VOCABULARY_FILENAME = 'vocabulary.json'
const KNOWN_AFTER = 2
/** Recordings without a term before it is forgotten: a one-off goes soon, a known term only once out of use. */
const FORGET_CANDIDATE_AFTER = 10
const FORGET_KNOWN_AFTER = 40
/** Enough for a team's names and projects without the prompt growing with every meeting. */
const MAX_KNOWN = 80

export function emptyVocabulary(): Vocabulary {
	return { version: 1, recordings: 0, pinned: [], blocked: [], terms: {} }
}

/** One term however it is cased or joined: В'ячеслав and Вячеслав, Coin Flip and Coinflip. */
const key = (term: string) => term.toLowerCase().replace(/['’ʼ\s-]/g, '')

/** The vocabulary after one more recording that turned up `terms`. */
export function learn(vocabulary: Vocabulary, terms: string[]): Vocabulary {
	const recordings = vocabulary.recordings + 1
	const blocked = new Set(vocabulary.blocked.map(key))
	const spelling = new Map(Object.keys(vocabulary.terms).map((term) => [key(term), term]))
	const next = { ...vocabulary.terms }
	const counted = new Set<string>()
	for (const term of terms.map((term) => term.trim())) {
		const id = key(term)
		if (!term || blocked.has(id) || counted.has(id)) continue
		counted.add(id)
		const known = spelling.get(id)
		if (known) next[known] = { seen: next[known].seen + 1, last: recordings }
		else next[term] = { seen: 1, last: recordings }
	}
	for (const [term, { seen, last }] of Object.entries(next)) {
		if (recordings - last >= (seen >= KNOWN_AFTER ? FORGET_KNOWN_AFTER : FORGET_CANDIDATE_AFTER)) delete next[term]
	}
	return { ...vocabulary, recordings, terms: next }
}

/** The pinned terms, then learned ones seen often enough to trust, most used and most recent first. */
export function knownTerms(vocabulary: Vocabulary): string[] {
	const blocked = new Set(vocabulary.blocked.map(key))
	const pinned = new Set(vocabulary.pinned.map(key))
	const learned = Object.entries(vocabulary.terms)
		.filter(([term, { seen }]) => seen >= KNOWN_AFTER && !blocked.has(key(term)) && !pinned.has(key(term)))
		.sort((a, b) => b[1].seen - a[1].seen || b[1].last - a[1].last)
		.map(([term]) => term)
	return [...vocabulary.pinned, ...learned].slice(0, MAX_KNOWN)
}

export function dropBlocked(vocabulary: Vocabulary, terms: string[]): string[] {
	const blocked = new Set(vocabulary.blocked.map(key))
	return terms.filter((term) => !blocked.has(key(term)))
}

/** Throws on a file that isn't a vocabulary, so a hand edit gone wrong is reported, not overwritten. */
export function parseVocabulary(text: string): Vocabulary {
	const value = JSON.parse(text) as Partial<Vocabulary>
	if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error(`${VOCABULARY_FILENAME} is not a vocabulary`)
	const strings = (list: unknown) => (Array.isArray(list) ? list.filter((item): item is string => typeof item === 'string') : [])
	const terms: Record<string, LearnedTerm> = {}
	for (const [term, entry] of Object.entries(value.terms ?? {})) {
		if (Number.isFinite(entry?.seen) && Number.isFinite(entry?.last)) terms[term] = { seen: entry.seen, last: entry.last }
	}
	return {
		version: 1,
		recordings: Number.isFinite(value.recordings) ? value.recordings! : 0,
		pinned: strings(value.pinned),
		blocked: strings(value.blocked),
		terms,
	}
}

export async function vocabularyPath(projectsPath?: string | null) {
	return pathApi.join(await transcriptsFolder(projectsPath), VOCABULARY_FILENAME)
}

/** The file's path, created empty when missing so it can be opened and edited before anything is learned. */
export async function vocabularyFile(projectsPath?: string | null) {
	const path = await vocabularyPath(projectsPath)
	if (!(await fs.exists(path))) await saveVocabulary(path, emptyVocabulary())
	return path
}

export async function loadVocabulary(path: string): Promise<Vocabulary> {
	return (await fs.exists(path)) ? parseVocabulary(await fs.readTextFile(path)) : emptyVocabulary()
}

/** Written beside and renamed over, so a crash never leaves half a file. */
export async function saveVocabulary(path: string, vocabulary: Vocabulary) {
	const temporary = `${path}.tmp-${Date.now()}`
	await fs.writeTextFile(temporary, JSON.stringify(vocabulary, null, '\t'))
	await fs.rename(temporary, path)
}
