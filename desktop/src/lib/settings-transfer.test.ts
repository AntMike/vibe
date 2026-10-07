import { describe, expect, it } from 'vitest'
import { CONFIG_KEYS } from './config-keys'
import { exportSettings, importSettings } from './settings-transfer'

const mine: Record<string, unknown> = {
	[CONFIG_KEYS.theme]: 'dark',
	[CONFIG_KEYS.modelPath]: 'C:/models/turbo.bin',
	[CONFIG_KEYS.ai]: { connection: { platform: 'cli', claudeApiKey: 'sk-mine', openaiApiKey: '' }, tasks: { summary: { prompt: 'P' } } },
	[CONFIG_KEYS.hints]: { enabled: true, draftModelPath: 'C:/models/parakeet.gguf' },
}

describe('settings transfer', () => {
	it('leaves keys, paths and machine state out of the file', () => {
		const file = exportSettings((key) => mine[key], new Date('2026-10-07T00:00:00Z'))
		expect(file.vibeSettings).toBe(1)
		expect(file.settings[CONFIG_KEYS.theme]).toBe('dark')
		expect(file.settings).not.toHaveProperty(CONFIG_KEYS.modelPath)
		expect(JSON.stringify(file)).not.toContain('sk-mine')
		expect(file.settings[CONFIG_KEYS.hints]).toEqual({ enabled: true })
		expect(file.settings[CONFIG_KEYS.ai]).toEqual({ connection: { platform: 'cli' }, tasks: { summary: { prompt: 'P' } } })
	})

	it('imports shared settings but keeps the receiver’s own keys and paths', () => {
		const theirs: Record<string, unknown> = {
			[CONFIG_KEYS.ai]: { connection: { claudeApiKey: 'sk-theirs', openaiApiKey: 'ok-theirs' } },
			[CONFIG_KEYS.hints]: { draftModelPath: 'D:/parakeet.gguf' },
		}
		const text = JSON.stringify(exportSettings((key) => mine[key]))
		const writes = importSettings(text, (key) => theirs[key])
		expect(writes[CONFIG_KEYS.theme]).toBe('dark')
		expect(writes).not.toHaveProperty(CONFIG_KEYS.modelPath)
		expect(writes[CONFIG_KEYS.ai]).toEqual({
			connection: { platform: 'cli', claudeApiKey: 'sk-theirs', openaiApiKey: 'ok-theirs' },
			tasks: { summary: { prompt: 'P' } },
		})
		expect(writes[CONFIG_KEYS.hints]).toEqual({ enabled: true, draftModelPath: 'D:/parakeet.gguf' })
	})

	it('ignores local keys slipped into a file and rejects files that aren’t settings', () => {
		const text = JSON.stringify({ vibeSettings: 1, settings: { [CONFIG_KEYS.modelPath]: 'X', unknown: 1 } })
		expect(importSettings(text, () => undefined)).toEqual({})
		expect(() => importSettings('{"a":1}', () => undefined)).toThrow('not a Vibe settings file')
		expect(() => importSettings('nope', () => undefined)).toThrow('not a JSON file')
		expect(() => importSettings('{"vibeSettings":99,"settings":{}}', () => undefined)).toThrow('newer Vibe')
	})
})
