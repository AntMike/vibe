import { describe, expect, it, vi } from 'vitest'

vi.mock('@tauri-apps/plugin-fs', () => ({}))
vi.mock('@tauri-apps/api/path', () => ({}))

import { applyReplacements, fixSpelling, guardedEdits } from './spelling'

const line = (text: string) => ({ start: 0, stop: 1, text })
const texts = (segments: { text: string }[]) => segments.map((segment) => segment.text)

describe('fixSpelling', () => {
	const families = [
		['естімейт', 'естімейти'],
		['ClickUp', 'Клікап'],
	]

	it('respells a word one sound off a known term, in its closest form', () => {
		expect(texts(fixSpelling([line('Дай стімейти, будь ласка.')], families))).toEqual(['Дай естімейти, будь ласка.'])
		expect(texts(fixSpelling([line('Задача в плікап?')], families))).toEqual(['Задача в Клікап?'])
	})

	it('leaves known forms, short words and words two off alone, as the same line', () => {
		for (const text of ['Дай естімейти.', 'Мітинг.', 'Дай тімейти.', 'Ставка тайм.']) {
			const segment = line(text)
			expect(fixSpelling([segment], families)[0]).toBe(segment)
		}
		// One off, but five letters is too short to trust.
		expect(texts(fixSpelling([line('лікап')], [['ClickUp', 'Клікап']]))).toEqual(['лікап'])
		// Two sounds off.
		expect(texts(fixSpelling([line('плікоп')], [['ClickUp', 'Клікап']]))).toEqual(['плікоп'])
	})

	it('leaves a word one off two terms alone', () => {
		expect(texts(fixSpelling([line('плікап')], [['Клікап'], ['Плікав']]))).toEqual(['плікап'])
	})
})

describe('applyReplacements', () => {
	it('replaces whole words in any case, Cyrillic included', () => {
		const pairs = [
			{ from: 'стімейти', to: 'естімейти' },
			{ from: 'фігма', to: 'Figma' },
		]
		expect(texts(applyReplacements([line('Стімейти у Фігма, не фігмастер.')], pairs))).toEqual(['естімейти у Figma, не фігмастер.'])
	})

	it('skips empty and no-op pairs and keeps untouched lines', () => {
		const segment = line('Фігма тут.')
		const out = applyReplacements(
			[segment, line('плікап')],
			[
				{ from: 'фігма', to: 'Фігма' },
				{ from: 'плікап', to: '' },
				{ from: '', to: 'x' },
			],
		)
		expect(out[0]).toBe(segment)
		expect(texts(out)).toEqual(['Фігма тут.', 'плікап'])
	})

	it('takes at most 80 pairs', () => {
		const pairs = Array.from({ length: 81 }, (_, i) => ({ from: `w${i}`, to: `x${i}` }))
		expect(texts(applyReplacements([line('w0 w80')], pairs))).toEqual(['x0 w80'])
	})
})

describe('guardedEdits', () => {
	const segments = [line('We use Click Up for tasks.'), line('Yes, that is fine.')]

	it('applies an edit within the cap', () => {
		expect(texts(guardedEdits(segments, new Map([[1, 'We use ClickUp for tasks.']])))).toEqual(['We use ClickUp for tasks.', 'Yes, that is fine.'])
	})

	it('keeps the original when the edit is a rewrite', () => {
		expect(texts(guardedEdits(segments, new Map([[2, 'Absolutely, I agree with everything you said.']])))).toEqual(texts(segments))
	})

	it('ignores line numbers outside the transcript', () => {
		expect(guardedEdits(segments, new Map([[0, 'x'], [3, 'y']]))).toEqual(segments)
	})
})
