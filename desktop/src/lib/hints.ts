import { chunkLines, createClient, fillPrompt, type AiConnection } from './ai'

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
}

export const DEFAULT_HINTS: HintsSettings = {
	enabled: false,
	callNames: true,
	draftModelPath: null,
	refiner: 'words',
}

/**
 * Whisper reads at most half its 448-token context as prompt and drops the front of anything longer.
 * Cyrillic runs about two characters a token, so this keeps a glossary whole in any script.
 */
const MAX_PROMPT_CHARS = 400
const MAX_TERMS = 50
/** CLIs have big contexts; this only keeps an hours-long draft from becoming a huge stdin. */
const CLI_CONTEXT_TOKENS = 100_000

export const GLOSSARY_PROMPT = `Output only the requested content. No introductions, explanations, or commentary.

Below is a rough automatic transcript of a recording; it has recognition mistakes. List the names of people, companies, products and places, and the specialized terms, that occur in it, spelled correctly and fixing obvious mishearings. Keep each term in the language and script it is spoken in. Known participants: {speakers}.

Answer with one comma-separated line of at most ${MAX_TERMS} terms, most important first.

"""
{transcript}
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

/** Boil a draft transcript down to its glossary with the chosen refiner. */
export async function refineDraft(lines: string[], names: string[], settings: HintsSettings, connection: AiConnection): Promise<string[]> {
	if (settings.refiner === 'words') return rareWords(lines.join('\n'))
	const contextTokens = connection.platform === 'cli' ? CLI_CONTEXT_TOKENS : connection.contextTokens
	// ponytail: only the first context-full of a long draft; names past it rarely matter more than the opening's.
	const transcript = chunkLines(lines, GLOSSARY_PROMPT, contextTokens)[0]
	const prompt = fillPrompt(GLOSSARY_PROMPT, { transcript, speakers: names.join(', ') || 'unknown' })
	// Unloaded right after when local: Whisper needs the GPU memory next.
	return parseGlossary(await createClient(connection, { unloadAfter: true }).ask(prompt))
}
