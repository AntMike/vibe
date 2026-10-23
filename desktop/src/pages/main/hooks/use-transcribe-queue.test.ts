// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest'
import { findJobTarget, reconcileProjectName, type Job } from './use-transcribe-queue'

describe('project name reconciliation', () => {
	it('keeps reconciling when the live title changes during a slow rename', async () => {
		let liveName = 'Second draft'
		const rename = vi.fn(async (_path: string, name: string) => {
			if (name === 'Second draft') liveName = 'Final title'
			return {
				path: `/projects/${name}/transcript.vibe.json`,
				mediaPath: `/projects/${name}/audio.wav`,
				name,
				createdAt: new Date(0),
			}
		})

		const result = await reconcileProjectName(
			'Original',
			{ recordPath: '/projects/Original/transcript.vibe.json', mediaPath: '/projects/Original/audio.wav' },
			() => liveName,
			rename,
		)

		expect(rename.mock.calls.map(([, name]) => name)).toEqual(['Second draft', 'Final title'])
		expect(result).toEqual({
			recordPath: '/projects/Final title/transcript.vibe.json',
			mediaPath: '/projects/Final title/audio.wav',
		})
	})
})

describe('late work target', () => {
	const job = (id: string, savedPath?: string) => ({ id, savedPath }) as Job
	const saved = new Map([['old', '/projects/A/transcript.vibe.json']])

	it('prefers the open job, then the reopened copy of its project, then the file alone', () => {
		expect(findJobTarget([job('old', '/x')], saved, 'old').job?.id).toBe('old')
		expect(findJobTarget([job('new', '/projects/A/transcript.vibe.json')], saved, 'old').job?.id).toBe('new')
		expect(findJobTarget([job('other', '/projects/B/transcript.vibe.json')], saved, 'old')).toEqual({
			job: undefined,
			savedPath: '/projects/A/transcript.vibe.json',
		})
		expect(findJobTarget([], saved, 'unknown')).toEqual({ job: undefined, savedPath: undefined })
	})
})
