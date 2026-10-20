import type { TranscriptExportContent } from './transcript-export'
import type { ProjectSource } from './types'
import { audioExtensions, videoExtensions } from './config'

// ponytail: ASCII like the "record" label, so names stay stable across UI languages
const contentSuffixes: Record<TranscriptExportContent, string> = {
	transcript: '-transcript',
	summary: '-summary',
	both: '-transcript-summary',
}

const mediaExtensions = new Set([...audioExtensions, ...videoExtensions])

function withoutMediaExtension(name: string): string {
	const dot = name.lastIndexOf('.')
	const extension = dot > 0 ? name.slice(dot + 1).toLowerCase() : ''
	return mediaExtensions.has(extension) ? name.slice(0, dot) : name
}

const pad = (value: number) => String(value).padStart(2, '0')

function datePrefix(date: Date) {
	return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}-${pad(date.getHours())}-${pad(date.getMinutes())}`
}

/** The stem the recorder falls back to when the user gave no name: "2026-10-20 14-30-05 (1)". */
const recorderTimeStem = /^\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}(?: \(\d+\))?$/

/**
 * Build the default saved-project name: `<date>-<time>-<label>`. The label is the file name or link
 * title, a recording's own name, or "record". User-entered names never pass through this helper.
 */
export function autoProjectName(name: string, source: ProjectSource = 'file', date = new Date()): string {
	const stem = withoutMediaExtension(name.trim()).trim()
	const label = !stem || (source === 'record' && recorderTimeStem.test(stem)) ? 'record' : stem
	return `${datePrefix(date)}-${label}`
}

const autoNamePrefix = /^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-/

/** Whether a title still has the shape `autoProjectName` gives it. */
export const isAutoProjectName = (name: string) => autoNamePrefix.test(name)

/** Swap the label of an auto-built name for a title taken from its summary; null when the name was the user's own. */
export function summaryProjectName(name: string, title: string): string | null {
	const prefix = name.match(autoNamePrefix)?.[0]
	const label = title
		.split('\n')[0]
		.replace(/[#*_`"'«»“”]/g, '')
		.replace(/^title:\s*/i, '')
		.replace(/[.\s]+$/, '')
		.trim()
		.slice(0, 60)
		.trim()
	return prefix && label ? `${prefix}${label}` : null
}

/** Turn the visible project title into a safe, recognizable export filename; `content` tags what is inside. */
export function projectExportFilename(name: string, extension: string, content?: TranscriptExportContent): string {
	const cleaned = withoutMediaExtension(name)
		.replace(/[/\\?%*:|"<>]/g, '-')
		// eslint-disable-next-line no-control-regex
		.replace(/[\u0000-\u001f]/g, '')
		.replace(/\s+/g, ' ')
		.trim()
		.replace(/^[. ]+|[. ]+$/g, '')
		.slice(0, 120)
	const safeStem = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(cleaned) ? `_${cleaned}` : cleaned || 'transcript'
	const safeExtension =
		extension
			.replace(/^\.+/, '')
			.replace(/[^a-z0-9]/gi, '')
			.toLowerCase() || 'txt'
	return `${safeStem}${content ? contentSuffixes[content] : ''}.${safeExtension}`
}
