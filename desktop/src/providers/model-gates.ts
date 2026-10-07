import { invoke } from '@tauri-apps/api/core'
import { join } from '@tauri-apps/api/path'
import { ask } from '@tauri-apps/plugin-dialog'
import { useCallback } from 'react'
import { toast } from 'sonner'
import { m } from '~/paraglide/messages.js'
import * as config from '~/lib/config'
import type { ModelIntegrity } from '~/lib/config'
import { isModelFileUsable, type ModelMetadata } from '~/lib/model'
import { usePreferenceProvider } from '~/providers/preference'
import { useToastProvider } from '~/providers/toast'

/**
 * Speaker recognition and stable timestamps each need a model on disk. Turning them on therefore
 * means: check for the file, ask before pulling it, download with a progress toast, and only then
 * flip the preference. The transcription options popover and the settings page share this so the
 * gate behaves the same wherever the switch lives.
 */
export function useModelGates() {
	const preference = usePreferenceProvider()
	const progressToast = useToastProvider()

	const ensureModel = useCallback(
		async (options: { filename: string; url: string; title: string; question: string; downloading: string; integrity?: ModelIntegrity }) => {
			const modelsFolder = await invoke<string>('get_models_folder')
			const modelPath = await join(modelsFolder, options.filename)
			// A file that fails its integrity check does not count as installed — download it again.
			if (await isModelFileUsable(modelPath)) return true

			const confirmed = await ask(options.question, { title: options.title, kind: 'info' })
			if (!confirmed) return false

			progressToast.setMessage(options.downloading)
			progressToast.setOpen(true)
			progressToast.setProgress(0)
			try {
				await invoke('download_model', { url: options.url, path: modelPath, integrity: options.integrity })
				toast.success(m.downloadComplete())
				return true
			} finally {
				progressToast.setOpen(false)
				progressToast.setProgress(null)
			}
		},
		[progressToast],
	)

	const toggleDiarization = useCallback(
		async (checked: boolean) => {
			if (!checked) {
				preference.setDiarizeEnabled(false)
				return
			}
			try {
				const ready = await ensureModel({
					filename: config.diarizeModelFilename,
					url: config.diarizeModelUrl,
					integrity: config.diarizeModelIntegrity,
					title: m.diarization(),
					question: m.downloadDiarizeModel(),
					downloading: m.downloadingDiarizeModel(),
				})
				if (ready) preference.setDiarizeEnabled(true)
			} catch (error) {
				console.error('diarization setup failed:', error)
				toast.error(String(error))
			}
		},
		[ensureModel, preference],
	)

	const toggleStableTimestamps = useCallback(
		async (checked: boolean) => {
			if (!checked) {
				preference.setStableTimestampsEnabled(false)
				return
			}
			try {
				const ready = await ensureModel({
					filename: config.vadModelFilename,
					url: config.vadModelUrl,
					title: m.stableTimestamps(),
					question: m.stableTimestampsConfirm(),
					downloading: m.downloadingVadModel(),
				})
				if (ready) preference.setStableTimestampsEnabled(true)
			} catch (error) {
				console.error('stable timestamps setup failed:', error)
				toast.error(String(error))
			}
		},
		[ensureModel, preference],
	)

	/**
	 * Make `modelPath` the transcription model: read what it can do, fetch the VAD model first if it
	 * needs one, and move the language off one the model doesn't know. Shared by the settings page
	 * and the model pickers next to the language, so switching works the same everywhere.
	 */
	const selectModel = useCallback(
		async (modelPath: string) => {
			let metadata: ModelMetadata | null = null
			try {
				metadata = await invoke<ModelMetadata>('get_model_metadata', { modelPath })
			} catch (error) {
				// Unknown GGUF formats may still be loadable by Server (for example Whisper GGUF).
				console.error('failed to read GGUF metadata:', error)
			}
			if (metadata?.capabilities.requires_vad) {
				const ready = await ensureModel({
					filename: config.vadModelFilename,
					url: config.vadModelUrl,
					title: 'Download required VAD model',
					question: 'This transcription model requires Silero VAD. Download it before selecting the model?',
					downloading: 'Downloading Silero VAD model…',
				})
				if (!ready) return
			}
			preference.setModelMetadata(metadata)
			const capabilities = metadata?.capabilities
			const lang = preference.modelOptions.lang
			if (capabilities && !(lang === 'auto' ? capabilities.language_detection : capabilities.languages.includes(lang))) {
				preference.setModelOptions({ ...preference.modelOptions, lang: capabilities.language_detection ? 'auto' : (capabilities.languages[0] ?? 'en') })
			}
			preference.setModelPath(modelPath)
		},
		[ensureModel, preference],
	)

	return { toggleDiarization, toggleStableTimestamps, selectModel }
}
