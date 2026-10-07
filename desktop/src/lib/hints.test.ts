import { describe, expect, it, vi } from 'vitest'
import { cliCommand, DEFAULT_AI } from './ai'
import { buildPrompt, DEFAULT_HINTS, parseGlossary, rareWords, refineDraft } from './hints'

const invokeMock = vi.fn()
vi.mock('@tauri-apps/plugin-http', () => ({ fetch: vi.fn() }))
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...args: unknown[]) => invokeMock(...args) }))

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
	it('sends the draft and names to a CLI connection on stdin and parses its answer', async () => {
		invokeMock.mockResolvedValue('Oleksii, Kubernetes')
		const connection = { ...DEFAULT_AI.connection, platform: 'cli' as const, cli: 'codex' as const, model: '' }
		const terms = await refineDraft(['we use kubernetes'], ['Oleksii'], { ...DEFAULT_HINTS, refiner: 'ai' }, connection)
		expect(terms).toEqual(['Oleksii', 'Kubernetes'])
		const [name, { command, prompt }] = invokeMock.mock.calls[0]
		expect(name).toBe('ask_cli')
		expect(command).toBe('codex exec --skip-git-repo-check -s read-only --color never -')
		expect(prompt).toContain('we use kubernetes')
		expect(prompt).toContain('Known participants: Oleksii.')
	})

	it('needs no AI for the free refiner', async () => {
		invokeMock.mockReset()
		expect(await refineDraft(['then Oleksii spoke'], [], DEFAULT_HINTS, DEFAULT_AI.connection)).toEqual(['Oleksii'])
		expect(invokeMock).not.toHaveBeenCalled()
	})
})

describe('cliCommand', () => {
	it('names the model only when it is a plain identifier', () => {
		const connection = { ...DEFAULT_AI.connection, platform: 'cli' as const, cli: 'claude' as const }
		expect(cliCommand({ ...connection, model: '' })).toBe('claude -p')
		expect(cliCommand({ ...connection, model: 'haiku' })).toBe('claude -p --model haiku')
		expect(cliCommand({ ...connection, model: 'haiku && rm -rf ~' })).toBe('claude -p')
		expect(cliCommand({ ...connection, cli: 'custom', cliCommand: ' my-ai --stdin ' })).toBe('my-ai --stdin')
	})
})
