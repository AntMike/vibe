/** How a recording's languages are chosen: the list as it is, detected, or detected among the list. */
export type LanguageMode = 'list' | 'detect' | 'detect-list'

/** Below either, a language is more likely the detector misjudging a short window than someone speaking it. */
const MIN_SECONDS = 5
const MIN_SHARE = 0.05

/** The languages a detection pass heard, most spoken first; `start`/`stop` are centiseconds. */
export function detectedLanguages(segments: { start: number; stop: number; language?: string }[]): string[] {
	const seconds = new Map<string, number>()
	for (const { start, stop, language } of segments) {
		if (language) seconds.set(language, (seconds.get(language) ?? 0) + (stop - start) / 100)
	}
	const total = [...seconds.values()].reduce((sum, heard) => sum + heard, 0)
	return [...seconds]
		.filter(([, heard]) => heard >= MIN_SECONDS && heard >= total * MIN_SHARE)
		.sort((a, b) => b[1] - a[1])
		.map(([language]) => language)
}
