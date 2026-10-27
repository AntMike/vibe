import { beforeEach, describe, expect, it, vi } from 'vitest'

const invoke = vi.fn()
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...args: unknown[]) => invoke(...args) }))

import { extraPasses, TURN_LOUDNESS_DBFS } from './transcribe-hints'

describe('extraPasses', () => {
	const main = '/models/main.bin'
	const other = '/models/other.bin'
	const loaded: string[] = []
	const run = {
		mainModelPath: main,
		loadModel: async (path: string) => void loaded.push(path),
		isAborted: () => false,
		onWarning: vi.fn(),
	}
	const decoded: Record<string, unknown>[] = []

	beforeEach(() => {
		loaded.length = 0
		decoded.length = 0
		invoke.mockImplementation(async (command: string, args: { options?: Record<string, unknown> }) => {
			if (command === 'get_model_metadata') return { capabilities: { requires_vad: false } }
			if (command === 'transcribe') {
				decoded.push(args.options!)
				return { segments: [{ start: 0, stop: 1, text: 'hi' }] }
			}
			return ''
		})
	})

	it('decodes main-model passes at their own speeds and other models at normal speed, levelled', async () => {
		const { runs, mainPasses } = await extraPasses('a.wav', [other], { model: null, segments: [] }, run, 2)
		expect(runs).toHaveLength(3)
		expect(mainPasses).toBe(2)
		expect(decoded).toEqual([
			{ path: 'a.wav', turn_loudness: TURN_LOUDNESS_DBFS },
			{ path: 'a.wav', turn_loudness: TURN_LOUDNESS_DBFS, speed: 0.95 },
			{ path: 'a.wav', turn_loudness: TURN_LOUDNESS_DBFS, speed: 1.05 },
		])
		expect(TURN_LOUDNESS_DBFS).toBe(-20)
		expect(loaded).toEqual([other, main])
	})

	it('ignores the main model in the list and needs no swap when only it runs', async () => {
		const { runs, mainPasses } = await extraPasses('a.wav', [main, main], { model: null, segments: [] }, run)
		expect(runs).toEqual([])
		expect(mainPasses).toBe(0)
		expect(decoded).toEqual([])
		const again = await extraPasses('a.wav', [main], { model: null, segments: [] }, run, 1)
		expect(again.mainPasses).toBe(1)
		expect(decoded.map((options) => options.speed)).toEqual([0.95])
		expect(loaded).toEqual([])
	})

	it('does not count a failed main pass', async () => {
		let calls = 0
		invoke.mockImplementation(async (command: string) => {
			if (command === 'get_model_metadata') return { capabilities: { requires_vad: false } }
			if (command === 'transcribe' && ++calls === 1) throw new Error('boom')
			if (command === 'transcribe') return { segments: [{ start: 0, stop: 1, text: 'hi' }] }
			return ''
		})
		const { runs, mainPasses } = await extraPasses('a.wav', [], { model: null, segments: [] }, run, 2)
		expect(runs).toHaveLength(1)
		expect(mainPasses).toBe(1)
		expect(run.onWarning).toHaveBeenCalledOnce()
	})

	it('still skips the draft model', async () => {
		await extraPasses('a.wav', [other], { model: other, segments: [] }, run)
		expect(decoded).toEqual([])
	})

	describe('language passes', () => {
		const lines = [
			{ start: 100, stop: 200, text: 'a', language: 'ru' },
			{ start: 250, stop: 400, text: 'b', language: 'ru' },
			{ start: 450, stop: 600, text: 'c', language: 'uk' },
			{ start: 1000, stop: 1100, text: 'd', language: 'ru' },
		]

		it('decodes only the lines in the language, joined when close, forced to it', async () => {
			const { runs, mainPasses } = await extraPasses('a.wav', [], { model: null, segments: [] }, run, 0, { segments: lines, models: { ru: other } })
			expect(decoded).toEqual([
				{
					path: 'a.wav',
					turn_loudness: TURN_LOUDNESS_DBFS,
					lang: 'ru',
					windows: [
						[1, 4],
						[10, 11],
					],
				},
			])
			expect(runs).toHaveLength(1)
			expect(mainPasses).toBe(0)
			expect(loaded).toEqual([other, main])
		})

		it('makes no request when the main pass heard none in the language', async () => {
			await extraPasses('a.wav', [], { model: null, segments: [] }, run, 0, { segments: lines, models: { en: other, ru: '' } })
			expect(decoded).toEqual([])
			expect(loaded).toEqual([])
		})

		it('only warns when it fails, and still loads the main model back', async () => {
			run.onWarning.mockClear()
			invoke.mockImplementation(async (command: string) => {
				if (command === 'get_model_metadata') return { capabilities: { requires_vad: false } }
				throw new Error('boom')
			})
			const { runs } = await extraPasses('a.wav', [], { model: null, segments: [] }, run, 0, { segments: lines, models: { ru: other } })
			expect(runs).toEqual([])
			expect(run.onWarning).toHaveBeenCalledOnce()
			expect(loaded).toEqual([other, main])
		})
	})
})
