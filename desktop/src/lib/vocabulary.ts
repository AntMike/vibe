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

// prettier-ignore
const LATIN: Record<string, string> = {
	а: 'a', б: 'b', в: 'v', г: 'g', ґ: 'g', д: 'd', е: 'e', є: 'e', ё: 'e', э: 'e', ж: 'zh', з: 'z', и: 'i', і: 'i', ї: 'i',
	й: 'i', ы: 'i', к: 'k', л: 'l', м: 'm', н: 'n', о: 'o', п: 'p', р: 'r', с: 's', т: 't', у: 'u', ф: 'f', х: 'h',
	ц: 'ts', ч: 'ch', ш: 'sh', щ: 'shch', ь: '', ъ: '', ю: 'u', я: 'a',
}

/**
 * One term however it is cased, joined, written or inflected: В'ячеслав and Вячеслав, Coin Flip and Coinflip,
 * Фігма and Figma, варруму and warroom, префабом and префаб. Spelled out in Latin by sound, then the case
 * ending cut off.
 */
// ponytail: rough sound-alike and ending cut; folds Олександра the person into Олександр, a per-spelling exception list would split them.
export function key(term: string) {
	const latin = [...term.toLowerCase().replace(/['’ʼ\s-]/g, '')]
		.map((letter) => LATIN[letter] ?? letter)
		.join('')
		.replace(/ph/g, 'f')
		.replace(/oo/g, 'u')
		.replace(/ee/g, 'i')
		.replace(/ck|q|c(?!h)/g, 'k')
		.replace(/w/g, 'v')
		.replace(/x/g, 'ks')
		.replace(/y/g, 'i')
		.replace(/(.)\1+/g, '$1')
	const stem = latin.replace(/(ami|ovi|evi|om|em|am|ah|iv)$/, '').replace(/[aeiou]+$/, '')
	return stem.length >= 3 ? stem : latin
}

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

/** Spellings of one term, from before they were matched as one or typed in by hand, as the most seen of them. */
function fold(terms: Record<string, LearnedTerm>, recordings: number): Record<string, LearnedTerm> {
	const folded: Record<string, LearnedTerm> = {}
	const spelling = new Map<string, string>()
	for (const [term, entry] of Object.entries(terms)) {
		const other = spelling.get(key(term))
		const prior = other ? folded[other] : undefined
		if (other) delete folded[other]
		const into = prior && prior.seen >= entry.seen ? other! : term
		folded[into] = prior ? { seen: Math.min(recordings, prior.seen + entry.seen), last: Math.max(prior.last, entry.last) } : entry
		spelling.set(key(term), into)
	}
	return folded
}

/**
 * The vocabulary with `tuning` applied; the pinned and blocked lists are the user's and stay as they are.
 * Only terms not known yet are dropped: a known one was picked from two transcripts, which the AI saw
 * and the tidy-up doesn't ("футбол" there is a project, not a sport).
 */
export function tune(vocabulary: Vocabulary, tuning: Tuning): Vocabulary {
	const blocked = new Set(vocabulary.blocked.map(key))
	const pinned = new Set(vocabulary.pinned.map(key))
	const terms = fold(vocabulary.terms, vocabulary.recordings)
	const spelling = new Map(Object.keys(terms).map((term) => [key(term), term]))
	const merges = new Map(Object.entries(tuning.merge).map(([variant, term]) => [key(variant), term.trim()]))
	// A → B and B → C both end at C, in whatever order the AI listed them.
	const target = (term: string) => {
		for (let i = 0; i < 5; i++) {
			const next = merges.get(key(term))
			// Same key is a respelling (Фігма → Figma), so the spelling, not the key, ends the chain.
			if (!next || next === term) break
			term = next
		}
		return term
	}
	for (const variant of Object.keys(tuning.merge)) {
		const from = spelling.get(key(variant))
		let to = target(variant)
		if (!from || !terms[from] || !to || to.toLowerCase() === from.toLowerCase()) continue
		// The AI may write "вітя" for a name: it keeps its capital.
		if (/^\p{Lu}/u.test(from)) to = to[0].toUpperCase() + to.slice(1)
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

/** `terms` without the blocked ones, each spelled as the vocabulary has it (a pinned spelling first): префабу as префаб. */
export function spellAsKnown(vocabulary: Vocabulary, terms: string[]): string[] {
	const blocked = new Set(vocabulary.blocked.map(key))
	const spelling = new Map([...Object.keys(vocabulary.terms), ...vocabulary.pinned].map((term) => [key(term), term]))
	return terms.filter((term) => !blocked.has(key(term))).map((term) => spelling.get(key(term)) ?? term)
}

/** Throws on a file that isn't a vocabulary, so a hand edit gone wrong is reported, not overwritten. */
export function parseVocabulary(text: string): Vocabulary {
	const value = JSON.parse(text) as Partial<Vocabulary>
	if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new Error(`${VOCABULARY_FILENAME} is not a vocabulary`)
	const strings = (list: unknown) => (Array.isArray(list) ? list.filter((item): item is string => typeof item === 'string') : [])
	const recordings = Number.isFinite(value.recordings) ? value.recordings! : 0
	const terms: Record<string, LearnedTerm> = {}
	for (const [term, entry] of Object.entries(value.terms ?? {})) {
		if (Number.isFinite(entry?.seen) && Number.isFinite(entry?.last)) terms[term] = { seen: entry.seen, last: entry.last }
	}
	return {
		version: 1,
		recordings,
		pinned: strings(value.pinned),
		blocked: strings(value.blocked),
		terms: fold(terms, recordings),
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
