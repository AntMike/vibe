import { Square } from 'lucide-react'
import { useEffect, useState } from 'react'
import { m } from '~/paraglide/messages.js'
import { Spinner } from '~/components/ui/spinner'
import { Tooltip, TooltipContent, TooltipTrigger } from '~/components/ui/tooltip'
import { useSession } from '../session'
import { formatElapsed } from './idle-hero'

const pill =
	'flex h-8 cursor-pointer items-center gap-2 rounded-full border border-border bg-muted/60 px-3 text-[13px] text-foreground transition-colors duration-150 hover:bg-muted'

/** Titlebar pills for work that is running but not on screen; each one opens its view. */
export default function ActivityStatus() {
	const { mode, panel, queue, recording, recordElapsed, recorderInFront, showRecorder, showTranscription } = useSession()
	const [stopping, setStopping] = useState(false)
	useEffect(() => setStopping(false), [recording.isRecording])

	const recorderVisible = panel === 'record' && (mode === 'idle' || recorderInFront)
	const showRecordPill = recording.isRecording && !recorderVisible
	const showTranscribePill = queue.running && (recorderInFront || queue.selectedId !== queue.activeId)
	const active = queue.jobs.find((job) => job.id === queue.activeId)

	return (
		<>
			{showTranscribePill && (
				<Tooltip>
					<TooltipTrigger asChild>
						<button type="button" className={pill} onClick={showTranscription}>
							<Spinner className="h-3.5 w-3.5" />
							<span className="tabular-nums">{m.transcribingPercent({ percent: Math.round(active?.progress ?? 0) })}</span>
							{queue.batch && (
								<span className="text-muted-foreground tabular-nums">
									{queue.batch.done + 1}/{queue.batch.total}
								</span>
							)}
						</button>
					</TooltipTrigger>
					<TooltipContent side="bottom">{m.showTranscription()}</TooltipContent>
				</Tooltip>
			)}
			{showRecordPill && (
				<div className="flex h-8 items-center rounded-full border border-border bg-muted/60 text-[13px] text-foreground">
					<Tooltip>
						<TooltipTrigger asChild>
							<button type="button" className={`${pill} border-0 bg-transparent pe-2`} onClick={showRecorder}>
								<span className="relative flex h-2 w-2">
									<span className="absolute inset-0 animate-ping rounded-full bg-red-500/60" />
									<span className="relative h-2 w-2 rounded-full bg-red-500" />
								</span>
								<span className="font-mono tabular-nums">{formatElapsed(recordElapsed)}</span>
							</button>
						</TooltipTrigger>
						<TooltipContent side="bottom">{m.showRecording()}</TooltipContent>
					</Tooltip>
					<Tooltip>
						<TooltipTrigger asChild>
							<button
								type="button"
								aria-label={m.stopRecording()}
								disabled={stopping}
								className="me-1 flex h-6 w-6 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors duration-150 hover:bg-background hover:text-foreground disabled:opacity-50"
								onClick={() => {
									setStopping(true)
									void recording.stopRecord()
								}}>
								{stopping ? <Spinner className="h-3 w-3" /> : <Square className="h-3 w-3 fill-current" />}
							</button>
						</TooltipTrigger>
						<TooltipContent side="bottom">{m.stopRecording()}</TooltipContent>
					</Tooltip>
				</div>
			)}
		</>
	)
}
