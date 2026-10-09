import { ChevronRight, SlidersHorizontal } from 'lucide-react'
import { DEFAULT_MODEL_OPTIONS } from '~/providers/preference'
import { m } from '~/paraglide/messages.js'
import { Button } from '~/components/ui/button'
import { Switch } from '~/components/ui/switch'
import { getFriendlyModelName } from '~/lib/model'
import { HintsGroup } from './hints'
import { ActionRow, SettingsGroup, SettingsNote, SettingsRow, type SettingsViewModel } from './shared'

export function TuningSection({ vm, onOpenWhisper }: { vm: SettingsViewModel; onOpenWhisper: () => void }) {
	const options = vm.preference.modelOptions
	const customized =
		Boolean(options.translate) ||
		(Object.keys(DEFAULT_MODEL_OPTIONS) as (keyof typeof DEFAULT_MODEL_OPTIONS)[])
			.filter((key) => key !== 'lang' && key !== 'verbose')
			.some((key) => (options[key] ?? DEFAULT_MODEL_OPTIONS[key]) !== DEFAULT_MODEL_OPTIONS[key])

	return (
		<div className="space-y-6">
			<SettingsGroup>
				<SettingsRow label={m.enableDiarization()} description={m.infoDiarization()}>
					<Switch checked={vm.preference.diarizeEnabled} onCheckedChange={vm.toggleDiarization} />
				</SettingsRow>
				{vm.preference.diarizeEnabled && <SettingsNote>{m.diarizeMaxSpeakersNote()}</SettingsNote>}
				<SettingsRow label={m.enableStableTimestamps()} description={m.stableTimestampsInfo()}>
					<Switch checked={vm.preference.stableTimestampsEnabled} onCheckedChange={vm.handleStableTimestampsToggle} />
				</SettingsRow>
				{vm.preference.stableTimestampsEnabled && <SettingsNote>{m.stableTimestampsSlowNote()}</SettingsNote>}
			</SettingsGroup>
			<HintsGroup vm={vm} />
			<SettingsGroup title={m.extraPasses()}>
				<SettingsNote>{m.extraPassesInfo()}</SettingsNote>
				<div className="flex flex-wrap gap-2 px-4 pb-3">
					{vm.models
						.filter((model) => model.path !== vm.preference.modelPath)
						.map((model) => {
							const chosen = vm.preference.extraPasses.includes(model.path)
							return (
								<Button
									key={model.path}
									type="button"
									size="sm"
									variant={chosen ? 'default' : 'outline'}
									onClick={() =>
										vm.preference.setExtraPasses(
											chosen
												? vm.preference.extraPasses.filter((path) => path !== model.path)
												: [...vm.preference.extraPasses, model.path],
										)
									}>
									{vm.preference.modelDisplayNames[model.path] ?? getFriendlyModelName(model.name)}
								</Button>
							)
						})}
				</div>
			</SettingsGroup>
			<SettingsGroup>
				<ActionRow
					label={
						<span className="flex items-center gap-2">
							<SlidersHorizontal className="h-4 w-4 text-muted-foreground" />
							{m.whisperOptions()}
							{customized && <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">{m.tuningCustomized()}</span>}
						</span>
					}
					description={m.whisperOptionsInfo()}
					icon={<ChevronRight className="h-4 w-4 rtl:rotate-180" />}
					activateOnClick
					onClick={onOpenWhisper}
				/>
			</SettingsGroup>
		</div>
	)
}
