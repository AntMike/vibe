import { describe, expect, it } from 'vitest'
import { canConfuse, languageDoubts, speakerLanguage, speakersToRedo, speakerWindows, withSpeakerLines } from './speaker-language'

const line = (start: number, stop: number, speaker: number, text = 'x', language = 'uk') => ({ start: start * 100, stop: stop * 100, text, speaker, language })

describe('speakerWindows', () => {
	it('takes each run of one speaker, padded but not into a neighbour', () => {
		const segments = [line(0, 5, 0), line(5.1, 9, 1), line(9.5, 12, 1), line(12.1, 20, 0), line(30, 31, 1)]
		const windows = speakerWindows(segments, 1)
		expect(windows.map(([start, end]) => [+start.toFixed(2), +end.toFixed(2)])).toEqual([
			[5, 12.1],
			[29.7, 31.3],
		])
	})
})

describe('withSpeakerLines', () => {
	it("swaps the speaker's lines for the new ones and keeps a stretch that came back empty", () => {
		const segments = [line(0, 5, 0), line(6, 9, 1, 'old a'), line(10, 12, 0), line(20, 22, 1, 'old b')]
		const heard = [{ start: 600, stop: 900, text: 'new a', language: 'ru', speaker: 0 }]
		const next = withSpeakerLines(segments, 1, heard)
		expect(next.map((segment) => [segment.text, segment.speaker])).toEqual([
			['x', 0],
			['new a', 1],
			['x', 0],
			['old b', 1],
		])
	})
})

describe('speakerLanguage and speakersToRedo', () => {
	const segments = [line(0, 10, 0), line(10, 12, 0, 'x', 'ru'), line(12, 20, 1)]

	it("finds a speaker's language by time", () => {
		expect(speakerLanguage(segments, 0)).toBe('uk')
		expect(speakerLanguage(segments, 3)).toBeUndefined()
	})

	it('redoes only named speakers whose lines are not all in their language', () => {
		const names = { 0: 'Anton', 1: 'Dima' }
		expect(speakersToRedo(segments, names, { Anton: 'ru', Dima: 'uk' })).toEqual([{ speaker: 0, language: 'ru' }])
		expect(speakersToRedo(segments, names, {})).toEqual([])
	})
})

describe('languageDoubts', () => {
	// The Oct 20 stand-up as the detector heard it: Ukrainian, with Anton's Russian split down the middle.
	const turns = [
		{ start: 0, end: 60, name: 'Tetiana' },
		{ start: 60, end: 107, name: 'Anton' },
		{ start: 107, end: 250, name: 'Volodymyr' },
		{ start: 250, end: 252, name: 'Oleh' },
	]
	const heard = [
		line(0, 60, 0, '', 'uk'),
		line(61, 83, 0, '', 'ru'),
		line(83, 107, 0, '', 'uk'),
		line(107, 241, 0, '', 'uk'),
		line(241, 250, 0, '', 'ru'), // 6% of Volodymyr: the detector, not him
		{ start: 25000, stop: 25200, text: '', language: 'ru' }, // too little talk to judge
		{ start: 0, stop: 6000, text: '', language: 'en' }, // English is told apart: no doubt
		{ start: 6000, stop: 6100, text: '' }, // unsure window
	]

	it('asks about people heard in the other language too often', () => {
		expect(languageDoubts(heard, turns, {})).toEqual({ language: 'uk', doubts: [{ name: 'Anton', seconds: { ru: 22, uk: 24 } }] })
	})

	it('does not ask again about someone whose language was picked', () => {
		expect(languageDoubts(heard, turns, { Anton: 'ru' })).toBeNull()
	})

	it('only checks when the languages can be confused', () => {
		expect(canConfuse(['uk', 'ru', 'en'])).toBe(true)
		expect(canConfuse([])).toBe(true)
		expect(canConfuse(['uk', 'en'])).toBe(false)
	})
})
