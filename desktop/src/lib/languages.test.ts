import { describe, expect, it } from 'vitest'
import { detectedLanguages } from './languages'

const window = (from: number, to: number, language?: string) => ({ start: from * 100, stop: to * 100, language })

describe('detectedLanguages', () => {
	it('keeps every language spoken long enough, most spoken first', () => {
		expect(detectedLanguages([window(0, 20, 'en'), window(20, 50, 'uk'), window(50, 80, 'uk')])).toEqual(['uk', 'en'])
	})

	it('drops a language heard for a moment', () => {
		expect(detectedLanguages([window(0, 60, 'uk'), window(60, 62, 'pl'), window(62, 90, 'en')])).toEqual(['uk', 'en'])
	})

	it('is empty when no window has a language', () => {
		expect(detectedLanguages([window(0, 30)])).toEqual([])
	})
})
