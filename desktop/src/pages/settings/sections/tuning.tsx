import { ChevronRight, SlidersHorizontal, X } from 'lucide-react'
import { useState } from 'react'
import { DEFAULT_MODEL_OPTIONS } from '~/providers/preference'
import { m } from '~/paraglide/messages.js'
import { Button } from '~/components/ui/button'
import { Switch } from '~/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '~/components/ui/select'
import { LanguageCombobox, matchesQuery } from '~/components/language-combobox'
import { useLanguageOptions } from '~/components/language-input'
import type { LanguageMode } from '~/lib/languages'
import { getFriendlyModelName } from '~/lib/model'
import { HintsGroup } from './hints'
import { ActionRow, SettingsGroup, SettingsNote, SettingsRow, rowControlClass, type SettingsViewModel } from './shared'

/** The select's value for no model; a select item can't be empty. */
const NO_MODEL = 'none'

export function TuningSection({ vm, onOpenWhisper }: { vm: SettingsViewModel; onOpenWhisper: () => void }) {
	const options = vm.preference.modelOptions
	const spoken = vm.preference.spokenLanguages
	const languages = useLanguageOptions()
	const nameOf = (code: string) => languages.find((option) => option.code === code)?.label ?? code
	const [query, setQuery] = useState('')
	const addable = languages.filter((option) => !spoken.includes(option.code) && matchesQuery(option, query.trim().toLowerCase()))
	const passes = vm.preference.languagePasses
	const [passQuery, setPassQuery] = useState('')
	const addablePass = languages.filter((option) => !(option.code in passes) && matchesQuery(option, passQuery.trim().toLowerCase()))
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
				<SettingsRow label={m.spokenLanguages()} description={m.spokenLanguagesInfo()}>
					<Select value={vm.preference.languageMode} onValueChange={(value: LanguageMode) => vm.preference.setLanguageMode(value)}>
						<SelectTrigger className={`w-64 ${rowControlClass}`}>
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							<SelectItem value="list">{m.languageModeList()}</SelectItem>
							<SelectItem value="detect">{m.languageModeDetect()}</SelectItem>
							<SelectItem value="detect-list">{m.languageModeDetectList()}</SelectItem>
						</SelectContent>
					</Select>
				</SettingsRow>
				{vm.preference.languageMode !== 'detect' && (
					<div className="flex flex-wrap items-center gap-2 px-4 py-3">
						{spoken.map((code) => (
							<Button
								key={code}
								type="button"
								size="sm"
								aria-label={m.removeLanguage({ language: nameOf(code) })}
								onClick={() => vm.preference.setSpokenLanguages(spoken.filter((other) => other !== code))}>
								{nameOf(code)}
								<X />
							</Button>
						))}
						<LanguageCombobox
							value=""
							onSelect={(code) => vm.preference.setSpokenLanguages([...spoken, code])}
							groups={[{ label: null, items: addable }]}
							query={query}
							onQueryChange={setQuery}
							triggerLabel={m.addLanguage()}
							ariaLabel={m.addLanguage()}
							className="w-56"
						/>
					</div>
				)}
			</SettingsGroup>
			<HintsGroup vm={vm} />
			<SettingsGroup title={m.extraPasses()}>
				<SettingsNote>{m.extraPassesInfo()}</SettingsNote>
				<SettingsRow label={m.mainPasses()} description={m.mainPassesInfo()}>
					<Select value={String(vm.preference.mainPasses)} onValueChange={(value) => vm.preference.setMainPasses(Number(value))}>
						<SelectTrigger className={`w-24 ${rowControlClass}`}>
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							{[0, 1, 2, 3, 4].map((count) => (
								<SelectItem key={count} value={String(count)}>
									{count}
								</SelectItem>
							))}
						</SelectContent>
					</Select>
				</SettingsRow>
				<SettingsNote>{m.languagePassInfo()}</SettingsNote>
				{Object.keys(passes).map((code) => (
					<SettingsRow key={code} label={m.languagePassModel({ language: nameOf(code) })}>
						<div className="flex items-center gap-2">
							<Select
								value={passes[code] || NO_MODEL}
								onValueChange={(value) => vm.preference.setLanguagePasses({ ...passes, [code]: value === NO_MODEL ? '' : value })}>
								<SelectTrigger className={`w-56 ${rowControlClass}`}>
									<SelectValue />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value={NO_MODEL}>{m.languagePassNone()}</SelectItem>
									{vm.models.map((model) => (
										<SelectItem key={model.path} value={model.path}>
											{vm.preference.modelDisplayNames[model.path] ?? getFriendlyModelName(model.name)}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
							<Button
								type="button"
								size="icon"
								variant="ghost"
								aria-label={m.removeLanguage({ language: nameOf(code) })}
								onClick={() => {
									const { [code]: _, ...rest } = passes
									vm.preference.setLanguagePasses(rest)
								}}>
								<X />
							</Button>
						</div>
					</SettingsRow>
				))}
				<div className="px-4 py-3">
					<LanguageCombobox
						value=""
						onSelect={(code) => vm.preference.setLanguagePasses({ ...passes, [code]: '' })}
						groups={[{ label: null, items: addablePass }]}
						query={passQuery}
						onQueryChange={setPassQuery}
						triggerLabel={m.addLanguage()}
						ariaLabel={m.addLanguage()}
						className="w-56"
					/>
				</div>
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
