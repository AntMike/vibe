import { disable, enable, isEnabled } from '@tauri-apps/plugin-autostart'
import { Moon, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import { toast } from 'sonner'
import { m } from '~/paraglide/messages.js'
import * as config from '~/lib/config'
import { DisplayLanguageInput } from '~/components/display-language-input'
import { Switch } from '~/components/ui/switch'
import { SettingsGroup, SettingsRow, type SettingsViewModel } from './shared'

/**
 * The OS owns this setting, not our config: the user can remove the login item from
 * system settings and a reinstall can clear it. So we read the real state on mount and
 * re-read it after every write instead of persisting a key that would drift.
 */
function LaunchAtStartupSwitch() {
	const [enabled, setEnabled] = useState(false)
	const [busy, setBusy] = useState(false)

	useEffect(() => {
		let cancelled = false
		isEnabled()
			.then((value) => {
				if (!cancelled) {
					setEnabled(value)
				}
			})
			.catch((error) => {
				console.error('failed to read autostart state', error)
			})
		return () => {
			cancelled = true
		}
	}, [])

	async function onCheckedChange(next: boolean) {
		setBusy(true)
		try {
			if (next) {
				await enable()
			} else {
				await disable()
			}
		} catch (error) {
			const message = next ? m.couldNotEnableLaunchAtStartup : m.couldNotDisableLaunchAtStartup
			toast.error(message({ error: String(error) }))
		}
		try {
			setEnabled(await isEnabled())
		} catch (error) {
			console.error('failed to read autostart state', error)
		}
		setBusy(false)
	}

	return <Switch checked={enabled} disabled={busy} onCheckedChange={onCheckedChange} />
}

export function GeneralSection({ vm }: { vm: SettingsViewModel }) {
	const themeLabels = { light: m.light, dark: m.dark } as const
	const themeIcons = { light: Sun, dark: Moon } as const

	return (
		<div className="space-y-5">
			<SettingsGroup>
				<SettingsRow label={m.displayLanguage()}>
					<DisplayLanguageInput value={vm.preference.displayLanguage} onSelect={vm.preference.setDisplayLanguage} className="w-52" />
				</SettingsRow>
				<SettingsRow label={m.theme()}>
					<div role="radiogroup" aria-label={m.theme()} className="inline-flex rounded-lg border border-border bg-muted/40 p-0.5">
						{config.themes.map((theme) => {
							const Icon = themeIcons[theme as keyof typeof themeIcons]
							const selected = vm.preference.theme === theme
							return (
								<button
									key={theme}
									type="button"
									role="radio"
									aria-checked={selected}
									aria-label={themeLabels[theme as keyof typeof themeLabels]()}
									title={themeLabels[theme as keyof typeof themeLabels]()}
									onClick={() => vm.preference.setTheme(theme as 'light' | 'dark')}
									className={`flex h-8 cursor-pointer items-center justify-center gap-1.5 rounded-md px-3 text-xs transition-colors ${
										selected ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
									}`}>
									{Icon && <Icon className="h-3.5 w-3.5" />}
									{themeLabels[theme as keyof typeof themeLabels]()}
								</button>
							)
						})}
					</div>
				</SettingsRow>
			</SettingsGroup>

			<SettingsGroup title={m.generalBehavior()}>
				<SettingsRow label={m.closeToTray()} description={m.generalTrayInfo()}>
					<Switch checked={vm.preference.closeToTray} onCheckedChange={vm.preference.setCloseToTray} />
				</SettingsRow>
				<SettingsRow label={m.launchAtStartup()} description={m.launchAtStartupInfo()}>
					<LaunchAtStartupSwitch />
				</SettingsRow>
				<SettingsRow label={m.preventSleep()} description={m.generalAwakeInfo()}>
					<Switch checked={vm.preference.preventSleep} onCheckedChange={vm.preference.setPreventSleep} />
				</SettingsRow>
				<SettingsRow label={m.saveProjects()} description={m.saveProjectsInfo()}>
					<Switch checked={vm.preference.saveTranscripts} onCheckedChange={vm.preference.setSaveTranscripts} />
				</SettingsRow>
			</SettingsGroup>
		</div>
	)
}
