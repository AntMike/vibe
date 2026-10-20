import { useEffect, useState } from 'react'
import { m } from '~/paraglide/messages.js'
import { getLocale } from '~/paraglide/runtime.js'
import { Button } from '~/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '~/components/ui/dialog'
import { cn } from '~/lib/style'
import { useSession } from '../session'

/**
 * Asked mid-transcription when the detector heard someone in Ukrainian and Russian both: pick what
 * each of them speaks. The picks are remembered by name, so the same people are not asked again.
 * Closing it goes on with the recording's language for everyone, remembering nothing.
 */
export function SpeakerLanguageDialog() {
	const { queue } = useSession()
	const question = queue.languageQuestion
	const [picks, setPicks] = useState<Record<string, string>>({})

	useEffect(() => {
		if (question) setPicks(Object.fromEntries(question.doubts.map((doubt) => [doubt.name, question.language])))
	}, [question])

	if (!question) return null
	const names = new Intl.DisplayNames([getLocale()], { type: 'language' })
	const share = (seconds: Record<string, number>) => {
		const total = Object.values(seconds).reduce((sum, value) => sum + value, 0)
		return Object.entries(seconds)
			.sort((a, b) => b[1] - a[1])
			.map(([code, value]) => `${names.of(code) ?? code} ${Math.round((value / total) * 100)}%`)
			.join(' · ')
	}

	return (
		<Dialog open onOpenChange={(open) => !open && queue.answerLanguageQuestion(null)}>
			<DialogContent className="max-w-lg rounded-2xl border-border/60 bg-card/95 p-6 shadow-xl">
				<DialogHeader>
					<DialogTitle className="text-lg font-semibold">{m.speakerLanguagesTitle()}</DialogTitle>
					<DialogDescription>{m.speakerLanguagesInfo({ name: question.jobName })}</DialogDescription>
				</DialogHeader>
				<div className="space-y-4 pt-2">
					{question.doubts.map((doubt) => (
						<div key={doubt.name} className="space-y-1.5">
							<div className="flex items-baseline justify-between gap-3">
								<span className="text-sm font-medium">{doubt.name}</span>
								<span className="text-[11px] text-muted-foreground">{m.speakerLanguagesHeard({ shares: share(doubt.seconds) })}</span>
							</div>
							<div className="flex flex-wrap gap-1.5">
								{question.languages.map((code) => (
									<button
										key={code}
										type="button"
										aria-pressed={picks[doubt.name] === code}
										onClick={() => setPicks((current) => ({ ...current, [doubt.name]: code }))}
										className={cn(
											'cursor-pointer rounded-full border px-2.5 py-1 text-[12px] font-medium transition-colors duration-150',
											picks[doubt.name] === code
												? 'border-foreground/20 bg-foreground text-background'
												: 'border-border bg-muted/50 text-muted-foreground hover:text-foreground',
										)}>
										{names.of(code) ?? code}
									</button>
								))}
							</div>
						</div>
					))}
					<Button onClick={() => queue.answerLanguageQuestion(picks)} className="w-full">
						{m.speakerLanguagesContinue()}
					</Button>
				</div>
			</DialogContent>
		</Dialog>
	)
}
