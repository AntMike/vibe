import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_AI } from './ai'
import { buildPrompt, DEFAULT_HINTS, parseGlossary, rareWords, refineDraft } from './hints'

vi.mock('@tauri-apps/plugin-http', () => ({ fetch: vi.fn() }))

describe('rareWords', () => {
	it('keeps names and mixed-case terms, most frequent first, and skips sentence starts', () => {
		const text = 'So we deploy to Kubernetes today. Then Oleksii checks iOS builds. Then Oleksii said GPT-5 is fine.\nSo it ships.'
		expect(rareWords(text)).toEqual(['Oleksii', 'Kubernetes', 'iOS', 'GPT-5'])
	})

	it('leaves out a capitalized word the draft also writes in lowercase', () => {
		expect(rareWords('We met. And then Monday came, and we left.')).toEqual(['Monday'])
	})

	it('works in Cyrillic', () => {
		expect(rareWords('Вчора ми говорили з Олексієм про Kubernetes.')).toEqual(['Олексієм', 'Kubernetes'])
	})
})

describe('parseGlossary', () => {
	it('reads a line, a list, or a fenced reply', () => {
		expect(parseGlossary('Oleksii, Kubernetes, Vibe.')).toEqual(['Oleksii', 'Kubernetes', 'Vibe'])
		expect(parseGlossary('Glossary:\n- Oleksii\n- Kubernetes\n2. Vibe')).toEqual(['Oleksii', 'Kubernetes', 'Vibe'])
		expect(parseGlossary('```\nOleksii; Kubernetes\n```')).toEqual(['Oleksii', 'Kubernetes'])
	})
})

describe('buildPrompt', () => {
	it('puts the glossary after the user prompt, deduplicated', () => {
		expect(buildPrompt(['Oleksii', 'oleksii', ' Vibe '], 'A call about releases.')).toBe('A call about releases. Oleksii, Vibe.')
	})

	it('drops the last terms once the prompt is full', () => {
		const terms = Array.from({ length: 100 }, (_, i) => `Term${i}`)
		const prompt = buildPrompt(terms)
		expect(prompt.length).toBeLessThanOrEqual(400)
		expect(prompt.startsWith('Term0, Term1,')).toBe(true)
		expect(prompt).not.toContain('Term99')
	})

	it('is empty with nothing to say', () => {
		expect(buildPrompt([])).toBe('')
		expect(buildPrompt([], '  ')).toBe('')
	})
})

describe('refineDraft', () => {
	it('sends the draft and names to the chosen CLI on stdin and parses its answer', async () => {
		const askCli = vi.fn().mockResolvedValue('Oleksii, Kubernetes')
		const terms = await refineDraft(
			['we use kubernetes'],
			['Oleksii'],
			{ ...DEFAULT_HINTS, refiner: 'cli', cli: 'codex' },
			{ connection: DEFAULT_AI.connection, askCli },
		)
		expect(terms).toEqual(['Oleksii', 'Kubernetes'])
		const [command, prompt] = askCli.mock.calls[0]
		expect(command).toMatch(/^codex exec /)
		expect(prompt).toContain('we use kubernetes')
		expect(prompt).toContain('Known participants: Oleksii.')
	})

	it('needs no AI for the free refiner', async () => {
		const askCli = vi.fn()
		expect(await refineDraft(['then Oleksii spoke'], [], DEFAULT_HINTS, { connection: DEFAULT_AI.connection, askCli })).toEqual(['Oleksii'])
		expect(askCli).not.toHaveBeenCalled()
	})
})
