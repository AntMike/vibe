import { open, save } from '@tauri-apps/plugin-dialog'
import * as fs from '@tauri-apps/plugin-fs'
import { Download, Upload } from 'lucide-react'
import { toast } from 'sonner'
import { CONFIG_KEYS } from '~/lib/config-keys'
import { readConfig, writeConfig } from '~/lib/config-store'
import { setMeetingDetectionEnabled } from '~/lib/meeting-prompt'
import { exportSettings, importSettings } from '~/lib/settings-transfer'
import { m } from '~/paraglide/messages.js'
import { ActionRow, SettingsGroup } from './shared'

const read = (key: string) => readConfig<unknown>(key, undefined)
const filters = [{ name: 'Vibe settings', extensions: ['json'] }]

/** Save the settings to a file to keep or share, or load one; keys, models and folders stay put. */
export function SettingsFileGroup() {
	async function exportFile() {
		const path = await save({ defaultPath: 'vibe-settings.json', filters })
		if (!path) return
		try {
			await fs.writeTextFile(path, JSON.stringify(exportSettings(read), null, '\t'))
			toast.success(m.settingsExported())
		} catch (error) {
			toast.error(`${m.error()}: ${String(error)}`)
		}
	}

	async function importFile() {
		const path = await open({ multiple: false, filters })
		if (typeof path !== 'string') return
		try {
			const writes = importSettings(await fs.readTextFile(path), read)
			for (const [key, value] of Object.entries(writes)) writeConfig(key, value)
			// The detector runs in Rust and only starts or stops through its command.
			const detection = writes[CONFIG_KEYS.meetingDetectionEnabled]
			if (typeof detection === 'boolean') await setMeetingDetectionEnabled(detection)
			toast.success(m.settingsImported({ count: String(Object.keys(writes).length) }))
		} catch (error) {
			toast.error(m.settingsImportFailed({ error: error instanceof Error ? error.message : String(error) }))
		}
	}

	return (
		<SettingsGroup title={m.settingsFile()}>
			<ActionRow
				label={m.exportSettings()}
				description={m.exportSettingsInfo()}
				icon={<Upload className="h-4 w-4" />}
				onClick={() => void exportFile()}
				activateOnClick
			/>
			<ActionRow
				label={m.importSettings()}
				description={m.importSettingsInfo()}
				icon={<Download className="h-4 w-4" />}
				onClick={() => void importFile()}
				activateOnClick
			/>
		</SettingsGroup>
	)
}
