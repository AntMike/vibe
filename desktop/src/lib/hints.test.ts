import { describe, expect, it, vi } from 'vitest'
import { cliCommand, DEFAULT_AI } from './ai'
import { DEFAULT_HINTS, parseGlossary, parsePostEdits, parseReplacements, rareWords, refineDraft, splitFixes, withoutFixes } from './hints'

const LEAN = '--tools "" --strict-mcp-config --setting-sources "" --no-session-persistence'

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

describe('splitFixes', () => {
	it('splits the trailing fixes block off the summary', () => {
		const reply = '# Stand-up\n\nAll good.\n\n```fixes\n[{"from": "зайнувати", "to": "заінвайтити"}, {"from": "Боти фулл-паті", "to": "Або ти фул-паті"}]\n```\n'
		expect(splitFixes(reply)).toEqual({
			summary: '# Stand-up\n\nAll good.',
			fixes: [
				{ from: 'зайнувати', to: 'заінвайтити' },
				{ from: 'Боти фулл-паті', to: 'Або ти фул-паті' },
			],
		})
	})

	it('keeps the summary when the block is missing, unclosed or broken', () => {
		expect(splitFixes('Just a summary.')).toEqual({ summary: 'Just a summary.', fixes: [] })
		expect(splitFixes('Sum.\n```fixes\n[{"from": "a", "to": "b"}]')).toEqual({ summary: 'Sum.', fixes: [{ from: 'a', to: 'b' }] })
		expect(splitFixes('Sum.\n```fixes\nnot json\n```')).toEqual({ summary: 'Sum.', fixes: [] })
	})

	it('hides the block while it streams in, a half-written fence too', () => {
		expect(withoutFixes('Sum.\n```fixes\n[{"fr')).toBe('Sum.\n')
		expect(withoutFixes('Sum.\n``')).toBe('Sum.\n')
		expect(withoutFixes('Sum.')).toBe('Sum.')
	})
})

describe('parseReplacements', () => {
	it('reads fenced JSON and drops anything but a pair of words', () => {
		const reply = '```json\n[{"from": " стімейти ", "to": "естімейти"}, {"from": "x"}, {"from": "a", "to": ""}, 3]\n```'
		expect(parseReplacements(reply)).toEqual([{ from: 'стімейти', to: 'естімейти' }])
		expect(parseReplacements('{}')).toEqual([])
		expect(() => parseReplacements('nothing to fix')).toThrow()
	})
})

describe('parseGlossary', () => {
	it('reads a line, a list, or a fenced reply', () => {
		expect(parseGlossary('Oleksii, Kubernetes, Vibe.')).toEqual(['Oleksii', 'Kubernetes', 'Vibe'])
		expect(parseGlossary('Glossary:\n- Oleksii\n- Kubernetes\n2. Vibe')).toEqual(['Oleksii', 'Kubernetes', 'Vibe'])
		expect(parseGlossary('```\nOleksii; Kubernetes\n```')).toEqual(['Oleksii', 'Kubernetes'])
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

	it('uses its own model when set', async () => {
		invokeMock.mockReset()
		invokeMock.mockResolvedValue('Oleksii')
		const connection = { ...DEFAULT_AI.connection, platform: 'cli' as const, cli: 'claude' as const, model: 'opus' }
		await refineDraft(['hi'], [], { ...DEFAULT_HINTS, refiner: 'ai', aiModel: ' haiku ' }, connection)
		expect(invokeMock.mock.calls[0][1].command).toBe(`claude -p --model haiku ${LEAN}`)
	})

	it('runs its own command over a custom CLI connection', async () => {
		invokeMock.mockReset()
		invokeMock.mockResolvedValue('Oleksii')
		const connection = { ...DEFAULT_AI.connection, platform: 'cli' as const, cli: 'custom' as const, cliCommand: 'claude -p --model sonnet' }
		await refineDraft(['hi'], [], { ...DEFAULT_HINTS, refiner: 'ai', aiModel: 'claude -p --model haiku' }, connection)
		expect(invokeMock.mock.calls[0][1].command).toBe('claude -p --model haiku')
	})

	it('tells the AI the known terms', async () => {
		invokeMock.mockReset()
		invokeMock.mockResolvedValue('Mumblum')
		const connection = { ...DEFAULT_AI.connection, platform: 'cli' as const, cli: 'claude' as const }
		await refineDraft(['по мумблуму'], [], { ...DEFAULT_HINTS, refiner: 'ai' }, connection, ['Mumblum / Мумблум', 'ШІ'])
		expect(invokeMock.mock.calls[0][1].prompt).toContain('a term written several ways as A / B: Mumblum / Мумблум, ШІ.')
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
		expect(cliCommand({ ...connection, model: '' })).toBe(`claude -p ${LEAN}`)
		expect(cliCommand({ ...connection, model: 'haiku' })).toBe(`claude -p --model haiku ${LEAN}`)
		expect(cliCommand({ ...connection, model: 'haiku && rm -rf ~' })).toBe(`claude -p ${LEAN}`)
		expect(cliCommand({ ...connection, cli: 'custom', cliCommand: ' my-ai --stdin ' })).toBe('my-ai --stdin')
	})
})

describe('parsePostEdits', () => {
	it('reads the changed lines by number, fenced or not', () => {
		expect(parsePostEdits('```json\n[{"n": 2, "text": " Клікап готовий. "}, {"n": 5, "text": "ok"}]\n```')).toEqual(
			new Map([
				[2, 'Клікап готовий.'],
				[5, 'ok'],
			]),
		)
	})

	it('drops entries that are not a number with a text', () => {
		expect(parsePostEdits('[{"n": "1", "text": "a"}, {"n": 2}, {"n": 3, "text": " "}, null]').size).toBe(0)
	})

	it('gives no edits for a garbage reply', () => {
		expect(parsePostEdits('Sure! Here are the fixes: line 2 was wrong.').size).toBe(0)
		expect(parsePostEdits('{"n": 1, "text": "a"}').size).toBe(0)
	})
})

describe('DEFAULT_HINTS', () => {
	it('corrects terms by default and leaves the sentence edit off', () => {
		expect(DEFAULT_HINTS.correct).toBe(true)
		expect(DEFAULT_HINTS.postEdit).toBe(false)
	})
})
