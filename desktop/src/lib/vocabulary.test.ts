import { describe, expect, it, vi } from 'vitest'

vi.mock('@tauri-apps/plugin-fs', () => ({}))
vi.mock('@tauri-apps/api/path', () => ({}))

import { dropBlocked, emptyVocabulary, knownTerms, learn, parseVocabulary } from './vocabulary'

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
		expect(dropBlocked(learned, ['Дімослав', 'Діма'])).toEqual(['Діма'])
	})

	it('reads a hand-edited file leniently and refuses one that is not a vocabulary', () => {
		const parsed = parseVocabulary('{"pinned":["ШІ",3],"terms":{"Figma":{"seen":2,"last":5},"bad":{}}}')
		expect(parsed).toEqual({ version: 1, recordings: 0, pinned: ['ШІ'], blocked: [], terms: { Figma: { seen: 2, last: 5 } } })
		expect(() => parseVocabulary('[]')).toThrow()
		expect(() => parseVocabulary('{"pinned": [')).toThrow()
	})
})
