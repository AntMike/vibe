import { beforeEach, describe, expect, it, vi } from 'vitest'

const invoke = vi.fn()
vi.mock('@tauri-apps/api/core', () => ({ invoke: (...args: unknown[]) => invoke(...args) }))

import { extraPasses, passSpeed, TURN_LOUDNESS_DBFS } from './transcribe-hints'

describe('passSpeed', () => {
	it('is half at normal speed, a quarter each slower and faster', () => {
		expect([0, 1, 2, 3, 4, 5, 6, 7].map(passSpeed)).toEqual([1, 0.95, 1.05, 1, 1, 0.95, 1.05, 1])
	})
})

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

	it('decodes the main model as an extra pass at that pass speed, levelled', async () => {
		const runs = await extraPasses('a.wav', [main], { model: null, segments: [] }, run)
		expect(runs).toHaveLength(1)
		expect(decoded).toEqual([{ path: 'a.wav', turn_loudness: TURN_LOUDNESS_DBFS, speed: 0.95 }])
		expect(TURN_LOUDNESS_DBFS).toBe(-20)
		expect(loaded).toEqual([])
	})

	it('numbers passes by list order, skips the main model at normal speed and swaps back', async () => {
		await extraPasses('a.wav', [other, main, '/models/third.bin', main], { model: null, segments: [] }, run)
		// Dedup leaves [other, main, third]: speeds 0.95, 1.05, 1.
		expect(decoded.map((options) => options.speed)).toEqual([0.95, 1.05, undefined])
		expect(loaded).toEqual([other, main, '/models/third.bin', main])
	})

	it('still skips the draft model', async () => {
		await extraPasses('a.wav', [other], { model: other, segments: [] }, run)
		expect(decoded).toEqual([])
	})
})
