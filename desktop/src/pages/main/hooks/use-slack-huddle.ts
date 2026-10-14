import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { useCallback, useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { formatExtensions } from '~/components/format-select'
import { renderExport } from '~/lib/auto-export'
import { CONFIG_KEYS } from '~/lib/config-keys'
import { usePersisted } from '~/lib/config-store'
import { projectExportFilename } from '~/lib/project-name'
import { findHuddleThread, postToThread, type HuddleThread } from '~/lib/slack'
import { m } from '~/paraglide/messages.js'
import { usePreferenceProvider } from '~/providers/preference'
import type { Summaries } from './use-summaries'
import type { Job, TranscribeQueue } from './use-transcribe-queue'

/**
 * Offer each Slack huddle recorded in this session to its huddle thread once the transcript is
 * done, and its summary too when one is made on finish. Send posts the summary as a reply and
 * attaches the transcript in the user's export format.
 */
export function useSlackHuddle(queue: TranscribeQueue, summaries: Summaries) {
	const preference = usePreferenceProvider()
	const [token] = usePersisted(CONFIG_KEYS.slackToken, '')
	const [withoutAsking] = usePersisted(CONFIG_KEYS.slackSendWithoutAsking, false)
	// Recordings waiting on their transcript, with the thread search already under way.
	const calls = useRef(new Map<string, Promise<HuddleThread | null>>())
	// Offers shown in a notification, waiting on its Send button.
	const offers = useRef(new Map<string, () => void>())
	const jobsRef = useRef(queue.jobs)
	useEffect(() => {
		jobsRef.current = queue.jobs
	}, [queue.jobs])

	/** A Slack call just became a project: start looking for its thread while it is transcribed. */
	const track = useCallback(
		(jobId: string, startedAt: number) => {
			const key = token.trim()
			if (!key) return
			const thread = findHuddleThread(key, startedAt).catch((error) => {
				console.error('slack huddle search failed:', error)
				return null
			})
			calls.current.set(jobId, thread)
		},
		[token],
	)

	const send = useCallback(
		async (jobId: string, thread: HuddleThread, where: string) => {
			const job = jobsRef.current.find((candidate) => candidate.id === jobId)
			if (!job) return
			try {
				const format = preference.textFormatTranscript
				const rendered = await renderExport(
					format,
					{ name: job.name, path: job.path, segments: job.segments, speakerNames: job.speakerNames },
					{
						direction: preference.textAreaDirection,
						theme: preference.exportOptions.theme,
						showTimestamps: preference.exportOptions.showTimestamps,
						showSpeakers: preference.exportOptions.showSpeakers,
						speakerLabel: m.speakerPrefix(),
						labels: { transcript: m.exportTranscript(), summary: m.exportSummary() },
					},
				)
				const bytes = typeof rendered === 'string' ? new TextEncoder().encode(rendered) : rendered
				const name = projectExportFilename(job.name, formatExtensions[format].slice(1), 'transcript')
				await postToThread(token.trim(), thread, job.summary, { name, bytes })
				toast.success(m.slackSent({ where }), { position: 'bottom-center' })
			} catch (error) {
				console.error('slack send failed:', error)
				toast.error(m.slackSendFailed(), { description: String(error), position: 'bottom-center' })
			}
		},
		[preference.exportOptions, preference.textAreaDirection, preference.textFormatTranscript, token],
	)

	const offer = useCallback(
		async (job: Job, search: Promise<HuddleThread | null>) => {
			const thread = await search
			if (!thread) {
				toast.error(m.slackNoHuddle(), { position: 'bottom-center' })
				return
			}
			const where = thread.label || m.slackDirectMessage()
			const go = () => void send(job.id, thread, where)
			if (withoutAsking) return go()
			// Only Windows records a call app on its own, so only Windows gets here.
			offers.current.set(job.id, go)
			void invoke('show_slack_send_toast', {
				title: m.slackOfferTitle(),
				body: m.slackOfferBody({ name: job.name, where }),
				send: m.slackSend(),
				dismiss: m.meetingPromptDismiss(),
				jobId: job.id,
			})
		},
		[send, withoutAsking],
	)

	useEffect(() => {
		const unlisten = listen<string>('slack_send', ({ payload }) => {
			offers.current.get(payload)?.()
			offers.current.delete(payload)
		})
		return () => {
			unlisten.then((fn) => fn())
		}
	}, [])

	const autoSummary = summaries.enabled && preference.ai.tasks.summary.autoOnFinish
	useEffect(() => {
		for (const [jobId, search] of calls.current) {
			const job = queue.jobs.find((candidate) => candidate.id === jobId)
			if (!job || job.status === 'error' || job.status === 'cancelled') {
				calls.current.delete(jobId)
				continue
			}
			if (job.status !== 'done' || job.segments.length === 0) continue
			// ponytail: waits on the auto summary's result; one that neither lands nor fails holds the offer.
			if (autoSummary && !job.summary && !summaries.errors[jobId]) continue
			calls.current.delete(jobId)
			void offer(job, search)
		}
	}, [autoSummary, offer, queue.jobs, summaries.errors])

	return { track }
}
