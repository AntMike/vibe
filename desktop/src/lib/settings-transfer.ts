import { CONFIG_KEYS } from './config-keys'

/**
 * Settings as a file to keep or hand to someone else. What is tied to this computer stays out:
 * API keys, paths to files on this disk, devices, and one-time state like the first-run flag.
 * Importing merges: the receiver keeps their own keys, models and folders.
 */
export const SETTINGS_FILE_VERSION = 1

export interface SettingsFile {
	vibeSettings: number
	exportedAt: string
	settings: Record<string, unknown>
}

const LOCAL_KEYS = new Set<string>([
	CONFIG_KEYS.firstRun,
	CONFIG_KEYS.skippedSetup,
	CONFIG_KEYS.analyticsEnabled,
	CONFIG_KEYS.modelPath,
	CONFIG_KEYS.modelDisplayNames,
	CONFIG_KEYS.gpuDevice,
	CONFIG_KEYS.noGpu,
	CONFIG_KEYS.gpuOutOfMemoryModels,
	CONFIG_KEYS.cpuVariant,
	CONFIG_KEYS.modelPromptDismissed,
	CONFIG_KEYS.projectsPath,
	CONFIG_KEYS.inputDeviceId,
	CONFIG_KEYS.outputDeviceId,
	CONFIG_KEYS.transcriptTab,
	CONFIG_KEYS.homeTab,
	CONFIG_KEYS.legacyLlmConfig,
	CONFIG_KEYS.legacyAutoSummarizeOnFinish,
	CONFIG_KEYS.ytDlpVersion,
	CONFIG_KEYS.ytDlpLastUpdateCheck,
	CONFIG_KEYS.ytDlpDeclinedVersion,
])

/** Fields inside a shared setting that still belong to this computer. */
const LOCAL_FIELDS: Record<string, string[][]> = {
	[CONFIG_KEYS.ai]: [
		['connection', 'claudeApiKey'],
		['connection', 'openaiApiKey'],
	],
	[CONFIG_KEYS.hints]: [['draftModelPath']],
	[CONFIG_KEYS.autoExport]: [['folder']],
}

const SHARED_KEYS = Object.values(CONFIG_KEYS).filter((key) => !LOCAL_KEYS.has(key))

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** A copy of `value` with `path` set to `field` (or removed when `field` is undefined). */
function withField(value: unknown, path: string[], field: unknown): unknown {
	if (!isObject(value)) return value
	const [head, ...rest] = path
	const copy = { ...value }
	if (rest.length > 0) copy[head] = withField(copy[head], rest, field)
	else if (field === undefined) delete copy[head]
	else copy[head] = field
	return copy
}

function fieldOf(value: unknown, path: string[]): unknown {
	return path.reduce<unknown>((node, step) => (isObject(node) ? node[step] : undefined), value)
}

/** The shareable settings, given a reader for the current value of a key. */
export function exportSettings(read: (key: string) => unknown, now = new Date()): SettingsFile {
	const settings: Record<string, unknown> = {}
	for (const key of SHARED_KEYS) {
		let value = read(key)
		if (value === undefined) continue
		for (const path of LOCAL_FIELDS[key] ?? []) value = withField(value, path, undefined)
		settings[key] = value
	}
	return { vibeSettings: SETTINGS_FILE_VERSION, exportedAt: now.toISOString(), settings }
}

/**
 * The values to write for an imported file: only known, shareable keys, with this computer's own
 * local fields kept. Throws on anything that isn't a Vibe settings file.
 */
export function importSettings(text: string, read: (key: string) => unknown): Record<string, unknown> {
	let file: unknown
	try {
		file = JSON.parse(text)
	} catch {
		throw new Error('not a JSON file')
	}
	if (!isObject(file) || typeof file.vibeSettings !== 'number' || !isObject(file.settings)) throw new Error('not a Vibe settings file')
	if (file.vibeSettings > SETTINGS_FILE_VERSION) throw new Error('made by a newer Vibe; update Vibe first')
	const incoming = file.settings
	const writes: Record<string, unknown> = {}
	for (const key of SHARED_KEYS) {
		if (!(key in incoming)) continue
		let value = incoming[key]
		for (const path of LOCAL_FIELDS[key] ?? []) value = withField(value, path, fieldOf(read(key), path))
		writes[key] = value
	}
	return writes
}
