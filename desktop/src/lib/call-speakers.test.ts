import { describe, expect, it } from 'vitest'
import { carrySpeakerNames, parseCallSpeakers, speakersFromCall, typedSpeakerNames } from './call-speakers'

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

	it('gives a handover to the voice heard, though the call ring still shows the one before', () => {
		// Olena answers at 18.8, but the ring moves to her only at 20.4: her "Hi." sits in Alex's turn.
		const lagging = [
			{ start: 0, end: 20.4, name: 'Alex' },
			{ start: 20.4, end: 30, name: 'Olena' },
			{ start: 30, end: 40, name: 'Alex' },
			{ start: 40, end: 50, name: 'Olena' },
		]
		const lines = [segment(0, 10, 0), segment(11, 18.6, 0), segment(18.8, 19.1, 1), segment(19.1, 29.5, 1), segment(28.5, 38, 0), segment(40.5, 49, 1)]
		const { segments, speakerNames } = speakersFromCall(lines, lagging)
		expect(segments.map((s) => speakerNames[s.speaker!])).toEqual(['Alex', 'Alex', 'Olena', 'Olena', 'Alex', 'Olena'])
	})

	it('names a diarized speaker only when the call covers most of its talk', () => {
		// Teams shows only you: speaker 1 crosses your turn for 1 of its 20 s, and stays unnamed.
		const mine = [{ start: 0, end: 10, name: 'Mike' }]
		const { segments, speakerNames } = speakersFromCall([segment(0, 9, 0), segment(9, 19, 1), segment(30, 40, 1)], mine)
		expect(segments.map((s) => s.speaker)).toEqual([0, 1, 1])
		expect(speakerNames).toEqual({ 0: 'Mike' })
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

describe('typedSpeakerNames', () => {
	// Serhii (10-14) handed over to Mike (14-30), but an earlier run heard it all as one Mike line.
	const turns = [
		{ start: 0, end: 10, name: 'Anton' },
		{ start: 10, end: 14, name: 'Serhii' },
		{ start: 14, end: 30, name: 'Mike' },
	]
	const before = [segment(0, 10, 0), segment(10, 30, 1), segment(30, 40, 2)]

	it('keeps only the names the call would not have given, so a re-run can name Serhii', () => {
		const typed = typedSpeakerNames(before, { 0: 'Anton', 1: 'Mike', 2: 'Olena' }, turns)
		expect(typed).toEqual({ 2: 'Olena' })
		const after = [segment(0, 10, 0), segment(10, 14, 1), segment(14, 30, 2), segment(30, 40, 3)]
		expect(carrySpeakerNames(before, typed, after)).toEqual({ 3: 'Olena' })
	})

	it('keeps every name when the recording has no call turns', () => {
		expect(typedSpeakerNames(before, { 1: 'Mike' }, undefined)).toEqual({ 1: 'Mike' })
	})

	it('keeps a name the user typed over the call one', () => {
		expect(typedSpeakerNames(before, { 0: 'Anton B.' }, turns)).toEqual({ 0: 'Anton B.' })
	})
})
