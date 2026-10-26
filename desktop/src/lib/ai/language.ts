import { detectedLanguages } from '~/lib/languages'
import { getLocale } from '~/paraglide/runtime.js'
import { PLACEHOLDERS } from './config'

/** The app language as a word the model understands: "Hebrew", "Français". */
export function appLanguageName() {
	try {
		return new Intl.DisplayNames(['en'], { type: 'language' }).of(getLocale()) ?? 'English'
	} catch {
		return 'English'
	}
}

/**
 * A prompt whose template is written in one language pulls the model into it, whatever the prompt
 * says about the transcript's language, so the summary is told what was actually spoken. A prompt
 * that asks for `{language}` picked its own output language and is left alone.
 */
export function spokenLanguageRule(template: string, segments: { start: number; stop: number; language?: string }[]) {
	const spoken = detectedLanguages(segments)[0]
	if (!spoken || template.includes(PLACEHOLDERS.language)) return ''
	let name = spoken
	try {
		name = new Intl.DisplayNames(['en'], { type: 'language' }).of(spoken) ?? spoken
	} catch {
		// an unknown code is still better than nothing to the model
	}
	return `The speakers mostly speak ${name}. Write the whole summary in ${name}, headings and labels included, even where the instructions or templates above are in another language.`
}
