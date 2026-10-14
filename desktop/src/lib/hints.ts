import { chunkLines, createClient, fillPrompt, type AiConnection } from './ai'
import { parseTuning, tune, type Vocabulary } from './vocabulary'

/**
 * Recognition hints: a glossary of the names and terms in a recording, handed to Whisper as its
 * prompt so it spells them right. The terms come from the call's participant names and, when a draft
 * model is set, from a fast first pass (Parakeet) that is boiled down to its rare words — by a free
 * heuristic or by the AI connection: an API, local Ollama, or a command-line AI on the user's subscription.
 */
export type HintsRefiner = 'words' | 'ai'

export interface HintsSettings {
	enabled: boolean
	/** Use the names of the people the call app showed talking. */
	callNames: boolean
	/** Model for the draft pass; null skips it. */
	draftModelPath: string | null
	refiner: HintsRefiner
	/** The AI refiner's own model, or its own command when the AI connection is a custom CLI; empty uses the summary's. */
	aiModel?: string
	/** Draft passes to decode, each over slightly altered audio; the server keeps the surest word. */
	draftRuns: number
	/** Keep `vocabulary.json` beside the transcripts: learn terms from each recording, tell the AI the known ones. */
	learn: boolean
	/** Recordings between the AI's tidy-ups of the learned vocabulary; 0 never tidies it. */
	tuneEvery: number
}

export const DEFAULT_HINTS: HintsSettings = {
	enabled: false,
	callNames: true,
	draftModelPath: null,
	refiner: 'words',
	draftRuns: 1,
	learn: false,
	tuneEvery: 0,
}

/**
 * Whisper reads at most half its 448-token context as prompt and drops the front of anything longer.
 * Cyrillic runs about two characters a token, so this keeps a glossary whole in any script.
 */
const MAX_PROMPT_CHARS = 400
const MAX_TERMS = 40
/** CLIs have big contexts; this only keeps an hours-long draft from becoming a huge stdin. */
const CLI_CONTEXT_TOKENS = 100_000

export const GLOSSARY_PROMPT = `Output only the requested content. No introductions, explanations, or commentary.

Below is a rough automatic transcript of a recording; it has recognition mistakes. List the names of people, companies, products and places, and the specialized terms, that occur in it, spelled correctly and fixing obvious mishearings. Write each term the way it belongs in a transcript of the spoken language, never translated or transliterated: people's names and borrowed words that take the language's endings in its own script (Діма, префаб, спайн, колбек); only brand and product names in Latin (Figma, Unity, Claude). Skip everyday words. Include a term only when you can tell which real name, word or term it is; leave out garbled words you cannot resolve to one (a wrong term does more harm than a missing one). A person's name counts only when it is used as a name: someone addressed, or said to do something. Known participants: {speakers}. Terms known from earlier recordings, a term written several ways as A / B: {known}. Spell a term one of its known ways when it is the same one, but list it only when it occurs in this transcript.

Answer with one comma-separated line of at most ${MAX_TERMS} terms, most important first.

"""
{transcript}
"""`

export const TUNE_PROMPT = `Output only the requested content. No introductions, explanations, or commentary.

Below are names and terms picked out of rough automatic transcripts of one group's recordings, each with the number of recordings it came up in and the other ways it was written, which stay with it. Some are one name or term written several ways: other inflected forms, other cases or scripts, or mishearings of it. Some are everyday words, or garbled words that are no real name or term.

Answer with JSON only: {"merge": {"variant": "term", ...}, "drop": ["term", ...]}
- merge: each variant to the one spelling its term should have, usually its most frequent entry. Write the term the way it belongs in a transcript of the spoken language, never translated or transliterated: people's names and borrowed words that take the language's endings in its own script and in their base form (Вітя, not Віті; префаб); only brand and product names in Latin (Figma, Unity, Claude).
- drop: everyday words, and garbled words you cannot tell the real name or term of. Keep a word that may be the name of a project, game, product, feature or person even when it is also an everyday word (a game called Taxi).
Leave out every term that is fine as it is. When unsure, leave a term alone: a wrong merge does more harm than a duplicate.

Always right as spelled: {pinned}

"""
{terms}
"""`

const WORD = /[\p{L}\p{N}][\p{L}\p{N}'’.+-]*[\p{L}\p{N}+]|[\p{L}\p{N}]/gu

/**
 * The free refiner: words a draft capitalizes mid-sentence (names) or writes with inner capitals or
 * digits (iOS, GPT-5), most frequent first. A word the draft also writes in lowercase is an ordinary
 * word that started a clause, so it is left out.
 */
// ponytail: capitalization heuristic; misses lowercase jargon, which is what the AI refiners are for.
export function rareWords(text: string, limit = MAX_TERMS): string[] {
	const lowercase = new Set<string>()
	const counts = new Map<string, number>()
	for (const sentence of text.split(/(?<=[.!?…])\s+|\n+/)) {
		const words = sentence.match(WORD) ?? []
		words.forEach((word, i) => {
			if (word === word.toLowerCase()) {
				lowercase.add(word)
				return
			}
			const special = /\p{Ll}\p{Lu}/u.test(word) || (/\p{N}/u.test(word) && /\p{L}/u.test(word))
			const capitalized = i > 0 && /^\p{Lu}/u.test(word) && word.length > 1
			if (special || capitalized) counts.set(word, (counts.get(word) ?? 0) + 1)
		})
	}
	return [...counts]
		.filter(([word]) => !lowercase.has(word.toLowerCase()))
		.sort((a, b) => b[1] - a[1])
		.slice(0, limit)
		.map(([word]) => word)
}

/** Terms out of an AI reply, whether it came back as a line, a list, or wrapped in a code fence. */
export function parseGlossary(reply: string): string[] {
	return reply
		.replace(/```\w*/g, '')
		.replace(/^\s*(glossary|terms)\s*:/i, '')
		.split(/[,;\n]/)
		.map((term) =>
			term
				.trim()
				.replace(/^([-*•]|\d+[.)])\s*/, '')
				.replace(/[.]$/, '')
				.trim(),
		)
		.filter((term) => term.length > 0 && term.length <= 40)
		.slice(0, MAX_TERMS)
}

/**
 * The Whisper prompt: the user's own prompt, then the terms as a plain list. No "Glossary:" label —
 * an English word there would nudge a Ukrainian call toward English. Names go first: when the list
 * runs long, it is the last, least important terms that don't fit.
 */
export function buildPrompt(terms: string[], userPrompt?: string): string {
	const seen = new Set<string>()
	const kept: string[] = []
	let length = 0
	for (const term of terms.map((term) => term.trim()).filter(Boolean)) {
		const key = term.toLowerCase()
		if (seen.has(key)) continue
		if (length + term.length + 2 > MAX_PROMPT_CHARS) break
		seen.add(key)
		kept.push(term)
		length += term.length + 2
	}
	const glossary = kept.length > 0 ? `${kept.join(', ')}.` : ''
	return [userPrompt?.trim(), glossary].filter(Boolean).join(' ')
}

/** The AI connection with the hints' own model, or own command for a custom CLI, when one is set. */
function hintsConnection(settings: HintsSettings, connection: AiConnection): AiConnection {
	const own = settings.aiModel?.trim()
	if (!own) return connection
	return connection.platform === 'cli' && connection.cli === 'custom' ? { ...connection, cliCommand: own } : { ...connection, model: own }
}

/** True once `tuneEvery` recordings were learned since the AI last tidied the vocabulary. */
export function dueForTuning(vocabulary: Vocabulary, settings: HintsSettings) {
	return settings.tuneEvery > 0 && vocabulary.recordings - (vocabulary.tuned ?? 0) >= settings.tuneEvery
}

/** Let the AI fold spellings and mishearings of one term together and drop junk; see `TUNE_PROMPT`. */
export async function tuneVocabulary(vocabulary: Vocabulary, settings: HintsSettings, connection: AiConnection): Promise<Vocabulary> {
	const terms = Object.entries(vocabulary.terms)
		.sort((a, b) => b[1].seen - a[1].seen)
		.map(([term, { seen, forms }]) => `${term} (${seen}${forms ? `; also ${forms.join(', ')}` : ''})`)
		.join('\n')
	const prompt = TUNE_PROMPT.replace('{pinned}', () => vocabulary.pinned.join(', ') || 'none').replace('{terms}', () => terms)
	return tune(vocabulary, parseTuning(await createClient(hintsConnection(settings, connection), { unloadAfter: true }).ask(prompt)))
}

/** Boil a draft transcript down to its glossary with the chosen refiner. */
export async function refineDraft(
	lines: string[],
	names: string[],
	settings: HintsSettings,
	connection: AiConnection,
	known: string[] = [],
): Promise<string[]> {
	if (settings.refiner === 'words') return rareWords(lines.join('\n'))
	connection = hintsConnection(settings, connection)
	const contextTokens = connection.platform === 'cli' ? CLI_CONTEXT_TOKENS : connection.contextTokens
	// ponytail: only the first context-full of a long draft; names past it rarely matter more than the opening's.
	// The known terms go in before chunking so a long list is counted against the draft's room.
	const template = GLOSSARY_PROMPT.replace('{known}', () => known.join(', ') || 'none')
	const transcript = chunkLines(lines, template, contextTokens)[0]
	const prompt = fillPrompt(template, { transcript, speakers: names.join(', ') || 'unknown' })
	// Unloaded right after when local: Whisper needs the GPU memory next.
	return parseGlossary(await createClient(connection, { unloadAfter: true }).ask(prompt))
}
