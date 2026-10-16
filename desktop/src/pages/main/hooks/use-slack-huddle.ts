import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { useCallback, useEffect, useRef, useState } from 'react'
import { toast } from 'sonner'
import { formatExtensions } from '~/components/format-select'
import { renderExport } from '~/lib/auto-export'
import { CONFIG_KEYS } from '~/lib/config-keys'
import { usePersisted } from '~/lib/config-store'
import { projectExportFilename } from '~/lib/project-name'
import { findHuddleThread, postWithConnector, postWithToken, type SlackFile, type SlackMethod, type SummaryAs } from '~/lib/slack'
import type { TranscriptExportContent } from '~/lib/transcript-export'
import { m } from '~/paraglide/messages.js'
import { usePreferenceProvider } from '~/providers/preference'
import type { Summaries } from './use-summaries'
import type { Job, TranscribeQueue } from './use-transcribe-queue'

export interface SlackHuddle {
	/** Sending is set up in Settings. */
	enabled: boolean
	/** A Slack call just became a project: offer it once it is transcribed. */
	track: (jobId: string) => void
	/** Post the job's summary, transcript or both to its huddle thread. */
	send: (jobId: string, content: TranscriptExportContent) => Promise<void>
	/** Jobs being sent right now. */
	sending: Record<string, boolean>
}

/**
 * Offer each Slack huddle recorded in this session to its huddle thread once the transcript is
 * done, and its summary too when one is made on finish. The notification can send, open the
 * project to edit the summary first, or be dismissed; the toolbar's Slack menu sends any time.
 */
export function useSlackHuddle(queue: TranscribeQueue, summaries: Summaries): SlackHuddle {
	const preference = usePreferenceProvider()
	const [method] = usePersisted<SlackMethod>(CONFIG_KEYS.slackMethod, 'off')
	const [token] = usePersisted(CONFIG_KEYS.slackToken, '')
	const [summaryAs] = usePersisted<SummaryAs>(CONFIG_KEYS.slackSummaryAs, 'message')
	const [defaultContent] = usePersisted<TranscriptExportContent>(CONFIG_KEYS.slackContent, 'both')
	const [withoutAsking] = usePersisted(CONFIG_KEYS.slackSendWithoutAsking, false)
	const [sending, setSending] = useState<Record<string, boolean>>({})
	// Recordings waiting on their transcript.
	const calls = useRef(new Set<string>())
	// Jobs whose notification is up, waiting on its Send button.
	const offers = useRef(new Set<string>())
	const jobsRef = useRef(queue.jobs)
	useEffect(() => {
		jobsRef.current = queue.jobs
	}, [queue.jobs])

	const track = useCallback(
		(jobId: string) => {
			if (method !== 'off') calls.current.add(jobId)
		},
		[method],
	)

	const send = useCallback(
		async (jobId: string, content: TranscriptExportContent) => {
			// The job as it is now, so a summary edited after the notification goes as edited.
			const job = jobsRef.current.find((candidate) => candidate.id === jobId)
			const call = job?.slackCall
			if (!job || !call) return
			const where = call.channel || m.slackDirectMessage()
			const summary = content !== 'transcript' ? job.summary?.trim() : undefined
			const withTranscript = content !== 'summary' && job.segments.length > 0
			if (!summary && !withTranscript) {
				toast.error(m.slackNothingToSend(), { position: 'bottom-center' })
				return
			}
			setSending((current) => ({ ...current, [jobId]: true }))
			const progress = toast.loading(m.slackSending({ where }), { position: 'bottom-center' })
			try {
				const encode = (text: string) => new TextEncoder().encode(text)
				const files: SlackFile[] = []
				if (withTranscript) {
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
					files.push({
						name: projectExportFilename(job.name, formatExtensions[format].slice(1), 'transcript'),
						bytes: typeof rendered === 'string' ? encode(rendered) : rendered,
					})
				}
				if (summary && summaryAs === 'file') files.push({ name: projectExportFilename(job.name, 'md', 'summary'), bytes: encode(summary) })
				const message = summaryAs === 'message' ? summary : undefined

				if (method === 'token') {
					const thread = await findHuddleThread(token.trim(), call.startedAt, call.channel)
					if (!thread) throw new Error(m.slackNoHuddle())
					await postWithToken(token.trim(), thread, message, files)
				} else {
					if (!call.channel) throw new Error(m.slackNoHuddle())
					await postWithConnector(call.channel, call.startedAt, message, files)
				}
				toast.success(m.slackSent({ where }), { id: progress, position: 'bottom-center' })
			} catch (error) {
				console.error('slack send failed:', error)
				toast.error(m.slackSendFailed(), { id: progress, description: String(error), position: 'bottom-center' })
			} finally {
				setSending(({ [jobId]: _, ...rest }) => rest)
			}
		},
		[method, preference.exportOptions, preference.textAreaDirection, preference.textFormatTranscript, summaryAs, token],
	)

	const offer = useCallback(
		(job: Job) => {
			if (withoutAsking) return void send(job.id, defaultContent)
			// Only Windows records a call app on its own, so only Windows gets here.
			offers.current.add(job.id)
			void invoke('show_slack_send_toast', {
				title: m.slackOfferTitle(),
				body: m.slackOfferBody({ name: job.name, where: job.slackCall?.channel || m.slackDirectMessage() }),
				send: m.slackSend(),
				review: m.slackReview(),
				dismiss: m.meetingPromptDismiss(),
				jobId: job.id,
			})
		},
		[defaultContent, send, withoutAsking],
	)

	const sendRef = useRef({ send, defaultContent, selectJob: queue.selectJob })
	useEffect(() => {
		sendRef.current = { send, defaultContent, selectJob: queue.selectJob }
	}, [defaultContent, queue.selectJob, send])

	useEffect(() => {
		const sent = listen<string>('slack_send', ({ payload }) => {
			if (offers.current.delete(payload)) void sendRef.current.send(payload, sendRef.current.defaultContent)
		})
		// Review: the window is up; show the project so its summary can be edited, then sent from the toolbar.
		const review = listen<string>('slack_review', ({ payload }) => {
			offers.current.delete(payload)
			if (jobsRef.current.some((job) => job.id === payload)) sendRef.current.selectJob(payload)
		})
		return () => {
			sent.then((fn) => fn())
			review.then((fn) => fn())
		}
	}, [])

	const autoSummary = summaries.enabled && preference.ai.tasks.summary.autoOnFinish
	useEffect(() => {
		for (const jobId of calls.current) {
			const job = queue.jobs.find((candidate) => candidate.id === jobId)
			if (!job || job.status === 'error' || job.status === 'cancelled') {
				calls.current.delete(jobId)
				continue
			}
			if (job.status !== 'done' || job.segments.length === 0) continue
			// ponytail: waits on the auto summary's result; one that neither lands nor fails holds the offer.
			if (autoSummary && !job.summary && !summaries.errors[jobId]) continue
			calls.current.delete(jobId)
			offer(job)
		}
	}, [autoSummary, offer, queue.jobs, summaries.errors])

	return { enabled: method !== 'off', track, send, sending }
}
