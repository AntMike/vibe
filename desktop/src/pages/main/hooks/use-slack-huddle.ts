import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { useCallback, useEffect, useRef } from 'react'
import { toast } from 'sonner'
import { formatExtensions } from '~/components/format-select'
import { renderExport } from '~/lib/auto-export'
import { CONFIG_KEYS } from '~/lib/config-keys'
import { usePersisted } from '~/lib/config-store'
import { projectExportFilename } from '~/lib/project-name'
import { findHuddleThread, postWithConnector, postWithToken, type SlackFile, type SlackMethod, type SummaryAs } from '~/lib/slack'
import { m } from '~/paraglide/messages.js'
import { usePreferenceProvider } from '~/providers/preference'
import type { Summaries } from './use-summaries'
import type { Job, TranscribeQueue } from './use-transcribe-queue'

/** A recorded huddle: when the recording started (unix seconds) and the channel Slack showed. */
interface Call {
	startedAt: number
	channel?: string
}

/**
 * Offer each Slack huddle recorded in this session to its huddle thread once the transcript is
 * done, and its summary too when one is made on finish. Send posts the summary, as a reply or a
 * file, and attaches the transcript in the user's export format.
 */
export function useSlackHuddle(queue: TranscribeQueue, summaries: Summaries) {
	const preference = usePreferenceProvider()
	const [method] = usePersisted<SlackMethod>(CONFIG_KEYS.slackMethod, 'off')
	const [token] = usePersisted(CONFIG_KEYS.slackToken, '')
	const [summaryAs] = usePersisted<SummaryAs>(CONFIG_KEYS.slackSummaryAs, 'message')
	const [withoutAsking] = usePersisted(CONFIG_KEYS.slackSendWithoutAsking, false)
	// Recordings waiting on their transcript.
	const calls = useRef(new Map<string, Call>())
	// Offers shown in a notification, waiting on its Send button.
	const offers = useRef(new Map<string, () => void>())
	const jobsRef = useRef(queue.jobs)
	useEffect(() => {
		jobsRef.current = queue.jobs
	}, [queue.jobs])

	/** A Slack call just became a project: offer it once it is transcribed. */
	const track = useCallback(
		(jobId: string, startedAt: number, channel?: string | null) => {
			if (method !== 'off') calls.current.set(jobId, { startedAt, channel: channel || undefined })
		},
		[method],
	)

	const send = useCallback(
		async (jobId: string, call: Call, where: string) => {
			const job = jobsRef.current.find((candidate) => candidate.id === jobId)
			if (!job) return
			const sending = toast.loading(m.slackSending({ where }), { position: 'bottom-center' })
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
				const encode = (text: string) => new TextEncoder().encode(text)
				const files: SlackFile[] = [
					{
						name: projectExportFilename(job.name, formatExtensions[format].slice(1), 'transcript'),
						bytes: typeof rendered === 'string' ? encode(rendered) : rendered,
					},
				]
				const summary = job.summary?.trim()
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
				toast.success(m.slackSent({ where }), { id: sending, position: 'bottom-center' })
			} catch (error) {
				console.error('slack send failed:', error)
				toast.error(m.slackSendFailed(), { id: sending, description: String(error), position: 'bottom-center' })
			}
		},
		[method, preference.exportOptions, preference.textAreaDirection, preference.textFormatTranscript, summaryAs, token],
	)

	const offer = useCallback(
		(job: Job, call: Call) => {
			const where = call.channel || m.slackDirectMessage()
			const go = () => void send(job.id, call, where)
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
		for (const [jobId, call] of calls.current) {
			const job = queue.jobs.find((candidate) => candidate.id === jobId)
			if (!job || job.status === 'error' || job.status === 'cancelled') {
				calls.current.delete(jobId)
				continue
			}
			if (job.status !== 'done' || job.segments.length === 0) continue
			// ponytail: waits on the auto summary's result; one that neither lands nor fails holds the offer.
			if (autoSummary && !job.summary && !summaries.errors[jobId]) continue
			calls.current.delete(jobId)
			offer(job, call)
		}
	}, [autoSummary, offer, queue.jobs, summaries.errors])

	return { track }
}
