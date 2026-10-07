import { m } from '~/paraglide/messages.js'
import type { HintsRefiner, HintsSettings } from '~/lib/hints'
import { getFriendlyModelName } from '~/lib/model'
import { Switch } from '~/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '~/components/ui/select'
import { SettingsGroup, SettingsNote, SettingsRow, rowControlClass, type SettingsViewModel } from './shared'

/** Select has no empty value; this stands for "no draft pass". */
const NO_DRAFT = 'none'

/** Recognition hints: a glossary for Whisper from call names and an optional draft pass. */
export function HintsGroup({ vm }: { vm: SettingsViewModel }) {
	const { hints, setHints } = vm.preference
	const set = (patch: Partial<HintsSettings>) => setHints({ ...hints, ...patch })

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
					{hints.draftModelPath && hints.refiner !== 'words' && <SettingsNote>{m.hintsRefinerAiNote()}</SettingsNote>}
				</>
			)}
		</SettingsGroup>
	)
}
