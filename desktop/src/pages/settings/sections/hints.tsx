import { useState } from 'react'
import { toast } from 'sonner'
import { m } from '~/paraglide/messages.js'
import { tuneVocabulary, type HintsRefiner, type HintsSettings } from '~/lib/hints'
import { getFriendlyModelName } from '~/lib/model'
import { Switch } from '~/components/ui/switch'
import { Input } from '~/components/ui/input'
import { Button } from '~/components/ui/button'
import { openPath } from '~/lib/app'
import { loadVocabulary, saveVocabulary, vocabularyFile, vocabularyPath } from '~/lib/vocabulary'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '~/components/ui/select'
import { SettingsGroup, SettingsNote, SettingsRow, rowControlClass, type SettingsViewModel } from './shared'

/** Select has no empty value; this stands for "no draft pass". */
const NO_DRAFT = 'none'
const DRAFT_RUNS = [1, 2, 3, 4, 5]
/** Recordings between AI tidy-ups of the vocabulary; 0 is off. */
const TUNE_EVERY = [0, 10, 25, 50]

/** Recognition hints: a glossary for Whisper from call names and an optional draft pass. */
export function HintsGroup({ vm }: { vm: SettingsViewModel }) {
	const { hints, setHints } = vm.preference
	const set = (patch: Partial<HintsSettings>) => setHints({ ...hints, ...patch })
	const connection = vm.preference.ai.connection
	const customCli = connection.platform === 'cli' && connection.cli === 'custom'
	const [tuning, setTuning] = useState(false)

	// ponytail: a transcription learning at the same moment can write over the tidy-up; the next one redoes it.
	async function tuneNow() {
		setTuning(true)
		try {
			const path = await vocabularyPath(vm.preference.projectsPath)
			const vocabulary = await loadVocabulary(path)
			const tuned = await tuneVocabulary(vocabulary, hints, connection)
			await saveVocabulary(path, tuned)
			toast.success(m.hintsTuneDone({ before: Object.keys(vocabulary.terms).length, after: Object.keys(tuned.terms).length }))
		} catch (error) {
			toast.error(m.hintsTuneFailed(), { description: String(error instanceof Error ? error.message : error) })
		} finally {
			setTuning(false)
		}
	}

	return (
		<SettingsGroup title={m.hints()}>
			<SettingsRow label={m.hintsEnable()} description={m.hintsEnableInfo()}>
				<Switch checked={hints.enabled} onCheckedChange={(enabled) => set({ enabled })} />
			</SettingsRow>
			{hints.enabled && vm.preference.modelMetadata?.capabilities.text_prompts === false && <SettingsNote>{m.hintsNeedsWhisper()}</SettingsNote>}
			{hints.enabled && (
				<>
					<SettingsRow label={m.hintsCallNames()} description={m.hintsCallNamesInfo()}>
						<Switch checked={hints.callNames} onCheckedChange={(callNames) => set({ callNames })} />
					</SettingsRow>
					<SettingsRow label={m.hintsDraftModel()} description={m.hintsDraftModelInfo()}>
						<Select
							value={hints.draftModelPath ?? NO_DRAFT}
							onValueChange={(value) => set({ draftModelPath: value === NO_DRAFT ? null : value })}
							onOpenChange={(open) => {
								if (open) vm.loadModels()
							}}>
							<SelectTrigger className={`w-56 ${rowControlClass}`}>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value={NO_DRAFT}>{m.hintsNoDraft()}</SelectItem>
								{vm.models.map((model) => (
									<SelectItem key={model.path} value={model.path}>
										{vm.preference.modelDisplayNames[model.path] ?? getFriendlyModelName(model.name)}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</SettingsRow>
					{hints.draftModelPath && (
						<SettingsRow label={m.hintsDraftRuns()} description={m.hintsDraftRunsInfo()}>
							<Select value={String(hints.draftRuns)} onValueChange={(value) => set({ draftRuns: Number(value) })}>
								<SelectTrigger className={`w-56 ${rowControlClass}`}>
									<SelectValue />
								</SelectTrigger>
								<SelectContent>
									{DRAFT_RUNS.map((runs) => (
										<SelectItem key={runs} value={String(runs)}>
											{runs}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</SettingsRow>
					)}
					{hints.draftModelPath && (
						<SettingsRow label={m.hintsRefiner()} description={m.hintsRefinerInfo()}>
							<Select value={hints.refiner === 'words' ? 'words' : 'ai'} onValueChange={(refiner: HintsRefiner) => set({ refiner })}>
								<SelectTrigger className={`w-56 ${rowControlClass}`}>
									<SelectValue />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value="words">{m.hintsRefinerWords()}</SelectItem>
									<SelectItem value="ai">{m.hintsRefinerAi()}</SelectItem>
								</SelectContent>
							</Select>
						</SettingsRow>
					)}
					{hints.draftModelPath && hints.refiner !== 'words' && (
						<>
							<SettingsRow
								label={customCli ? m.hintsAiCommand() : m.hintsAiModel()}
								description={customCli ? m.hintsAiCommandInfo() : m.hintsAiModelInfo()}>
								<Input
									value={hints.aiModel ?? ''}
									onChange={(e) => set({ aiModel: e.target.value })}
									placeholder={customCli ? connection.cliCommand : connection.model}
									className={`${customCli ? 'w-64' : 'w-56'} ${rowControlClass}`}
								/>
							</SettingsRow>
							<SettingsNote>{m.hintsRefinerAiNote()}</SettingsNote>
						</>
					)}
					<SettingsRow label={m.hintsLearn()} description={m.hintsLearnInfo()}>
						<div className="flex items-center gap-2">
							{hints.learn && (
								<Button
									variant="outline"
									size="sm"
									onClick={async () => openPath({ name: '', path: await vocabularyFile(vm.preference.projectsPath) })}>
									{m.hintsLearnOpen()}
								</Button>
							)}
							<Switch checked={hints.learn} onCheckedChange={(learn) => set({ learn })} />
						</div>
					</SettingsRow>
					{hints.learn && hints.draftModelPath && hints.refiner !== 'words' && (
						<SettingsRow label={m.hintsTune()} description={m.hintsTuneInfo()}>
							<div className="flex items-center gap-2">
								<Button variant="outline" size="sm" disabled={tuning} onClick={tuneNow}>
									{tuning ? m.hintsTuneRunning() : m.hintsTuneNow()}
								</Button>
								<Select value={String(hints.tuneEvery)} onValueChange={(value) => set({ tuneEvery: Number(value) })}>
									<SelectTrigger className={`w-56 ${rowControlClass}`}>
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										{TUNE_EVERY.map((count) => (
											<SelectItem key={count} value={String(count)}>
												{count === 0 ? m.hintsTuneOff() : m.hintsTuneEvery({ count })}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
							</div>
						</SettingsRow>
					)}
				</>
			)}
		</SettingsGroup>
	)
}
