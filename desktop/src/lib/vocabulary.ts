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
	/** The `recordings` count when the AI last tidied the terms; see `tune`. */
	tuned?: number
}

/** A tidy-up the AI suggests: other spellings and mishearings to fold into one term, and junk to drop. */
export interface Tuning {
	merge: Record<string, string>
	drop: string[]
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

/**
 * The vocabulary with `tuning` applied; the pinned and blocked lists are the user's and stay as they are.
 * Only terms not known yet are dropped: a known one was picked from two transcripts, which the AI saw
 * and the tidy-up doesn't ("футбол" there is a project, not a sport).
 */
export function tune(vocabulary: Vocabulary, tuning: Tuning): Vocabulary {
	const blocked = new Set(vocabulary.blocked.map(key))
	const pinned = new Set(vocabulary.pinned.map(key))
	const spelling = new Map(Object.keys(vocabulary.terms).map((term) => [key(term), term]))
	const merges = new Map(Object.entries(tuning.merge).map(([variant, term]) => [key(variant), term.trim()]))
	// A → B and B → C both end at C, in whatever order the AI listed them.
	const target = (term: string) => {
		for (let i = 0; i < 5; i++) {
			const next = merges.get(key(term))
			if (!next || key(next) === key(term)) break
			term = next
		}
		return term
	}
	const terms = { ...vocabulary.terms }
	for (const variant of merges.keys()) {
		const from = spelling.get(variant)
		const to = target(variant)
		if (!from || !terms[from] || !to || to === from) continue
		const entry = terms[from]
		delete terms[from]
		if (blocked.has(key(to)) || pinned.has(key(to))) continue
		const existing = spelling.get(key(to))
		const into = existing && terms[existing] ? existing : to
		const prior = terms[into]
		terms[into] = prior ? { seen: Math.min(vocabulary.recordings, prior.seen + entry.seen), last: Math.max(prior.last, entry.last) } : entry
		spelling.set(key(to), into)
	}
	for (const term of tuning.drop) {
		const known = spelling.get(key(term))
		if (known && terms[known]?.seen < KNOWN_AFTER) delete terms[known]
	}
	return { ...vocabulary, terms, tuned: vocabulary.recordings }
}

/** The AI's tidy-up out of its reply, whether or not it came wrapped in a code fence. */
export function parseTuning(reply: string): Tuning {
	let value: { merge?: unknown; drop?: unknown }
	try {
		value = JSON.parse(reply.slice(reply.indexOf('{'), reply.lastIndexOf('}') + 1))
	} catch {
		throw new Error(`The AI did not answer with a vocabulary tidy-up: ${reply.slice(0, 200)}`)
	}
	const merge: Record<string, string> = {}
	for (const [variant, term] of Object.entries(value?.merge ?? {})) if (typeof term === 'string') merge[variant] = term
	const drop = Array.isArray(value?.drop) ? value.drop.filter((term): term is string => typeof term === 'string') : []
	return { merge, drop }
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
		...(Number.isFinite(value.tuned) ? { tuned: value.tuned } : {}),
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
