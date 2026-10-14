import { describe, expect, it, vi } from 'vitest'

vi.mock('@tauri-apps/plugin-fs', () => ({}))
vi.mock('@tauri-apps/api/path', () => ({}))

import { emptyVocabulary, key, knownTerms, learn, parseTuning, parseVocabulary, spellAsKnown, tune } from './vocabulary'

describe('vocabulary', () => {
	it('knows a term once two recordings turned it up, in its first spelling', () => {
		const once = learn(emptyVocabulary(), ['Mumblum', 'Дімослав', 'mumblum'])
		expect(once.terms.Mumblum.seen).toBe(1)
		expect(knownTerms(once)).toEqual([])
		const twice = learn(once, ['MUMBLUM', 'Coinflip'])
		expect(knownTerms(twice)).toEqual(['Mumblum'])
		expect(knownTerms(learn(twice, ['Coin Flip', "В'ячеслав"]))).toEqual(['Coinflip', 'Mumblum'])
	})

	it('forgets a one-off soon and a known term only once it is out of use', () => {
		let vocabulary = learn(learn(emptyVocabulary(), ['Mumblum', 'Дімослав']), ['Mumblum'])
		for (let i = 0; i < 9; i++) vocabulary = learn(vocabulary, ['Figma'])
		expect(vocabulary.terms['Дімослав']).toBeUndefined()
		expect(knownTerms(vocabulary)).toContain('Mumblum')
		for (let i = 0; i < 40; i++) vocabulary = learn(vocabulary, ['Figma'])
		expect(vocabulary.terms.Mumblum).toBeUndefined()
		expect(knownTerms(vocabulary)).toEqual(['Figma'])
	})

	it('puts pinned terms first and never learns or uses blocked ones', () => {
		const vocabulary = { ...emptyVocabulary(), pinned: ['ШІ'], blocked: ['дімослав'] }
		const learned = learn(learn(vocabulary, ['Дімослав', 'Figma']), ['Дімослав', 'Figma'])
		expect(learned.terms['Дімослав']).toBeUndefined()
		expect(knownTerms(learned)).toEqual(['ШІ', 'Figma'])
		expect(spellAsKnown(learned, ['Дімослав', 'Діма', 'figma'])).toEqual(['Діма', 'Figma'])
	})

	it('matches a term across scripts and endings, and only that', () => {
		const same = [
			['Фігма', 'Figma', 'Фігмою'],
			['Крикет', 'Cricket', 'крікету'],
			['префаб', 'префабу', 'Prefab', 'префабами'],
			['варрум', 'варруму', 'warroom'],
			['Коїнфліп', 'Coin Flip'],
			['Коннект', 'Connect'],
			['Вітя', 'Віті', 'Вітею'],
			['Олексій', 'Олексія', 'Олексієм'],
			['Мунблум', 'Мунблуму'],
		]
		for (const group of same) expect(new Set(group.map(key)).size).toBe(1)
		const apart = ['Діма', 'Дмитро', 'Макс', 'Максим', 'ШІ', 'Шиішка', 'Таня', 'Тамара', 'Ян', 'Яна']
		expect(new Set(apart.map(key)).size).toBe(apart.length)
		const learned = learn(learn(emptyVocabulary(), ['префаб']), ['префабу', 'Prefab'])
		expect(learned.terms).toEqual({ префаб: { seen: 2, last: 2 } })
		expect(spellAsKnown({ ...learned, pinned: ['Prefab'] }, ['префабом'])).toEqual(['Prefab'])
	})

	it('folds spellings of one term in an older file into the most seen', () => {
		const parsed = parseVocabulary('{"recordings":10,"terms":{"Figma":{"seen":1,"last":2},"Фігма":{"seen":9,"last":10}}}')
		expect(parsed.terms).toEqual({ Фігма: { seen: 10, last: 10 } })
	})

	it("renames a term to the AI's spelling of it", () => {
		const vocabulary = { ...emptyVocabulary(), recordings: 3, terms: { Фігма: { seen: 3, last: 3 } } }
		expect(tune(vocabulary, { merge: { Фігма: 'Figma' }, drop: [] }).terms).toEqual({ Figma: { seen: 3, last: 3 } })
	})

	it('reads a hand-edited file leniently and refuses one that is not a vocabulary', () => {
		const parsed = parseVocabulary('{"pinned":["ШІ",3],"terms":{"Figma":{"seen":2,"last":5},"bad":{}}}')
		expect(parsed).toEqual({ version: 1, recordings: 0, pinned: ['ШІ'], blocked: [], terms: { Figma: { seen: 2, last: 5 } } })
		expect(() => parseVocabulary('[]')).toThrow()
		expect(() => parseVocabulary('{"pinned": [')).toThrow()
		expect(parseVocabulary('{"recordings":12,"tuned":10}').tuned).toBe(10)
	})

	it("folds the AI's variants into one term, counts and all, and drops junk", () => {
		const vocabulary = {
			...emptyVocabulary(),
			recordings: 10,
			pinned: ['ШІ'],
			terms: {
				Вітя: { seen: 6, last: 10 },
				Віті: { seen: 7, last: 9 },
				Moonblu: { seen: 1, last: 4 },
				Монблум: { seen: 3, last: 8 },
				Шi: { seen: 2, last: 7 },
				сорян: { seen: 1, last: 6 },
				футбол: { seen: 7, last: 10 },
			},
		}
		// Monblum → Мунблум → listed after its own variant; "Шi" is the pinned term.
		const tuned = tune(vocabulary, {
			merge: { Віті: 'вітя', Moonblu: 'Монблум', Монблум: 'Мунблум', Шi: 'ШІ', Ghost: 'Figma' },
			drop: ['Сорян', 'футбол', 'nothing'],
		})
		// A known term is never dropped, only merged.
		expect(tuned.terms).toEqual({ Вітя: { seen: 10, last: 10 }, Мунблум: { seen: 4, last: 8 }, футбол: { seen: 7, last: 10 } })
		expect(tuned.tuned).toBe(10)
		expect(tuned.pinned).toEqual(['ШІ'])
	})

	it("reads the AI's tidy-up out of a fenced reply and refuses prose", () => {
		const reply = '```json\n{"merge": {"Віті": "Вітя", "bad": 3}, "drop": ["сорян", 1]}\n```'
		expect(parseTuning(reply)).toEqual({ merge: { Віті: 'Вітя' }, drop: ['сорян'] })
		expect(() => parseTuning('Nothing to change.')).toThrow(/tidy-up/)
	})
})
