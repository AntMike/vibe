import { describe, expect, it } from 'vitest'
import { carrySpeakerNames, parseCallSpeakers, speakersFromCall } from './call-speakers'

const segment = (start: number, stop: number, speaker?: number) => ({ start: start * 100, stop: stop * 100, text: 'x', speaker })

describe('speakersFromCall', () => {
	const turns = [
		{ start: 0, end: 10, name: 'Alex' },
		{ start: 10, end: 20, name: 'Serhii' },
		{ start: 20, end: 30, name: 'Olena' },
		{ start: 40, end: 41, name: 'Alex' },
	]

	it('gives each sentence whoever the call showed talking, one speaker per person', () => {
		// Diarization lumped three people into speaker 0.
		const { segments, speakerNames } = speakersFromCall([segment(0, 9, 0), segment(11, 19, 0), segment(21, 29, 0), segment(1, 8)], turns)
		expect(segments.map((s) => s.speaker)).toEqual([0, 1, 2, 0])
		expect(speakerNames).toEqual({ 0: 'Alex', 1: 'Serhii', 2: 'Olena' })
	})

	it('falls back to the diarized speaker, and its best name, when the call cannot place a sentence', () => {
		// 40-50 has Alex for only a tenth of it; 60-70 has nobody.
		const { segments, speakerNames } = speakersFromCall([segment(0, 9, 3), segment(40, 50, 3), segment(60, 70, 5), segment(80, 90)], turns)
		expect(segments.map((s) => s.speaker)).toEqual([0, 0, 1, undefined])
		expect(speakerNames).toEqual({ 0: 'Alex' })
	})
})

describe('parseCallSpeakers', () => {
	it('keeps only well-formed turns', () => {
		expect(
			parseCallSpeakers([
				{ start: 0, end: 1, name: 'Alex' },
				{ start: 0, name: 'x' },
				{ start: 0, end: 1, name: ' ' },
			]),
		).toEqual([{ start: 0, end: 1, name: 'Alex' }])
		expect(parseCallSpeakers([])).toBeUndefined()
		expect(parseCallSpeakers('x')).toBeUndefined()
	})
})

describe('carrySpeakerNames', () => {
	it('names each new speaker after the earlier speaker who said most of its lines', () => {
		const before = [segment(0, 10, 0), segment(10, 20, 1), segment(20, 30, 2)]
		// The new run numbered them differently and split speaker 0's time across two lines.
		const after = [segment(0, 4, 5), segment(4, 10, 5), segment(10, 20, 3), segment(20, 30, 4), segment(30, 40)]
		expect(carrySpeakerNames(before, { 0: 'Alex', 1: ' Serhii ' }, after)).toEqual({ 5: 'Alex', 3: 'Serhii' })
	})

	it('leaves a speaker unnamed when it mostly talks over an unnamed earlier one', () => {
		const before = [segment(0, 3, 0), segment(3, 10, 1)]
		expect(carrySpeakerNames(before, { 0: 'Alex' }, [segment(0, 10, 0)])).toEqual({})
	})
})
