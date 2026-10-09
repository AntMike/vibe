import { describe, expect, it } from 'vitest'
import { mergeRuns, swapLoops, voteWords } from './vote'

const line = (start: number, stop: number, text: string, speaker?: number) => ({ start, stop, text, ...(speaker === undefined ? {} : { speaker }) })

describe('swapLoops', () => {
	it('replaces a repeated line with what the other run heard there, keeping the speaker', () => {
		const main = [
			line(0, 3000, 'Thanks for watching!', 0),
			line(3000, 6000, 'Thanks for watching', 0),
			line(6000, 9000, ' thanks for watching! ', 0),
			line(9000, 12000, 'Then Dima.', 0),
		]
		const other = [line(1000, 4000, 'Let me start.'), line(5000, 8000, 'Worked on cricket.'), line(9500, 11500, 'Then Dima.')]
		expect(swapLoops(main, other)).toEqual([
			line(1000, 4000, 'Let me start.', 0),
			line(5000, 8000, 'Worked on cricket.', 0),
			line(9000, 12000, 'Then Dima.', 0),
		])
	})

	it('drops a loop the other run heard nothing under, and keeps a line said twice', () => {
		const main = [line(0, 1000, 'Yes.'), line(1000, 2000, 'Yes.'), line(2000, 3000, 'No.'), line(3000, 4000, 'No.'), line(4000, 5000, 'No.')]
		expect(swapLoops(main, [line(0, 1500, 'Yes, yes.')])).toEqual(main.slice(0, 2))
	})
})

describe('voteWords', () => {
	it('replaces, drops and adds the words most runs agree on, keeping the main spelling otherwise', () => {
		const main = [line(0, 5000, 'Um the fat cat sat on mat.', 1), line(5000, 8000, 'It Slept.', 1)]
		const a = [line(0, 5000, 'the flat cat sat on the mat'), line(5000, 8000, 'it slept')]
		const b = [line(0, 2500, 'the flat cat'), line(2500, 5000, 'sat on the mat!'), line(5000, 8000, 'it wept')]
		expect(voteWords(main, [a, b])).toEqual([line(0, 5000, 'the flat cat sat on the mat.', 1), line(5000, 8000, 'It Slept.', 1)])
	})

	it('needs a strict majority, so one other run changes nothing', () => {
		const main = [line(0, 1000, 'fat cat')]
		expect(voteWords(main, [[line(0, 1000, 'flat cat')]])).toEqual(main)
	})
})

describe('mergeRuns', () => {
	it('swaps loops with one run and votes from two', () => {
		const main = [line(0, 1000, 'Hi.'), line(1000, 2000, 'Hi.'), line(2000, 3000, 'Hi.'), line(3000, 4000, 'Bye now.')]
		const a = [line(500, 2500, 'Hello there.'), line(3000, 4000, 'Bye for now.')]
		const b = [line(3000, 4000, 'Bye for now.')]
		expect(mergeRuns(main, [])).toEqual(main)
		expect(mergeRuns(main, [a])).toEqual([line(500, 2500, 'Hello there.'), line(3000, 4000, 'Bye now.')])
		expect(mergeRuns(main, [a, b])).toEqual([line(500, 2500, 'Hello there.'), line(3000, 4000, 'Bye for now.')])
	})
})
