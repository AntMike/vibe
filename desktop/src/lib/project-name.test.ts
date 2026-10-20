import { describe, expect, it } from 'vitest'
import { autoProjectName, isAutoProjectName, projectExportFilename, summaryProjectName } from './project-name'

describe('autoProjectName', () => {
	const at = new Date(2026, 9, 20, 9, 5, 30)

	it.each([
		['record', '2026-10-20 09-05-12.wav', '2026-10-20-09-05-record'],
		['record', '2026-10-20 09-05-12 (2).wav', '2026-10-20-09-05-record'],
		['record', 'Standup.wav', '2026-10-20-09-05-Standup'],
		['record', '', '2026-10-20-09-05-record'],
		['url', 'video-title.m4a', '2026-10-20-09-05-video-title'],
		['file', 'interview-final.mp3', '2026-10-20-09-05-interview-final'],
	] as const)('names %s projects from %s', (source, name, expected) => {
		expect(autoProjectName(name, source, at)).toBe(expected)
	})

	it('preserves dots that are not supported media extensions', () => {
		expect(autoProjectName('meeting.notes', 'file', at)).toBe('2026-10-20-09-05-meeting.notes')
	})
})

describe('summaryProjectName', () => {
	it('swaps the label for the summary title', () => {
		expect(summaryProjectName('2026-10-20-09-05-record', '**"Q3 roadmap review."**\n')).toBe('2026-10-20-09-05-Q3 roadmap review')
	})

	it('leaves names the user chose', () => {
		expect(isAutoProjectName('Team sync')).toBe(false)
		expect(summaryProjectName('Team sync', 'Q3 roadmap')).toBeNull()
	})
})

describe('projectExportFilename', () => {
	it.each([
		['File-interview-final.mp3', 'pdf', 'File-interview-final.pdf'],
		['Record-capture.WAV', '.docx', 'Record-capture.docx'],
		['team-standup', 'txt', 'team-standup.txt'],
		['Quarterly / review: final', 'md', 'Quarterly - review- final.md'],
		['NUL', 'json', '_NUL.json'],
		['   ', 'csv', 'transcript.csv'],
	] as const)('builds a safe export name from %s', (name, extension, expected) => {
		expect(projectExportFilename(name, extension)).toBe(expected)
	})

	it.each([
		['transcript', 'File-call-transcript.pdf'],
		['summary', 'File-call-summary.pdf'],
		['both', 'File-call-transcript-summary.pdf'],
	] as const)('tags %s exports', (content, expected) => {
		expect(projectExportFilename('File-call.mp3', 'pdf', content)).toBe(expected)
	})
})
