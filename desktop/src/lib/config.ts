export const repoURL = 'https://github.com/thewh1teagle/vibe'
export const modelsDocURL = 'https://thewh1teagle.github.io/vibe/docs#models'
export const storeFilename = 'app_config.json'

/** What the catalog knows about a finished download, beyond the bytes themselves. */
export interface ModelIntegrity {
	/** Exact size in bytes of the published artifact. */
	size?: number
	/** Lowercase hex SHA-256 of the published artifact. */
	sha256?: string
}

/** One downloadable model. Only catalog entries carry integrity metadata — a URL the user pastes
 * into settings or opens with `vibe://download/?url=` never does, and falls back to the size and
 * magic-byte checks the backend runs on every download. */
export interface ModelDownload extends ModelIntegrity {
	url: string
}

// TODO: fill in `size` and `sha256` for every entry below from the real published artifacts
// (`curl -sI <url>` for the length, `shasum -a 256 <file>` for the hash). They are deliberately
// left unset rather than guessed: a wrong value would reject a perfectly good download.
export const modelUrls: Record<'default' | 'hebrew', ModelDownload[]> = {
	default: [
		{ url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin' },
		{ url: 'https://huggingface.co/vibe-app/whisper-large-v3-turbo-gguf/resolve/main/ggml-large-v3-turbo.bin' }, // Hugging Face fallback
		{ url: 'https://github.com/thewh1teagle/vibe/releases/download/model-files-v1.0/ggml-large-v3-turbo.bin' }, // GitHub fallback
	],
	hebrew: [{ url: 'https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/main/ggml-model.bin' }],
}

/**
 * Diarization runs NVIDIA Nemotron 3 Diarization (Sortformer v3) on ggml: up to 8 speakers, overlap
 * included, at 10 ms. NVIDIA's own q8_0 GGUF (`nvidia/Nemotron-3-Diarization`, OpenMDW-1.1). Q8_0:
 * the k-quant tiers are unsafe for Sortformer — its speaker-cache compression makes near-tie
 * decisions that quant error can flip, permuting speaker labels mid-stream.
 *
 * On 26 recorded calls of 9-12 people it scores 23.9% DER against the call's own speaker turns,
 * where the 4-speaker Sortformer v2 it replaces scored 53.5% (scripts/diar-eval).
 */
export const diarizeModelFilename = 'Nemotron-3-Diarization.q8_0.gguf'
export const diarizeModelUrl = 'https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/Nemotron-3-Diarization.q8_0.gguf'
export const diarizeModelIntegrity: ModelIntegrity = {
	size: 107012128,
	sha256: '08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1',
}
/**
 * Diarization models earlier versions downloaded. Nothing reads them, but they sit in the models
 * folder and the GGUF one would otherwise list as a transcription model; deleting them is left to
 * the user.
 */
export const retiredDiarizeModelFilenames = ['diar_streaming_sortformer_4spk-v2.q8_0.gguf', 'wespeaker_en_voxceleb_CAM++.onnx', 'segmentation-3.0.onnx']
export const vadModelFilename = 'ggml-silero-v6.2.0.bin'
export const vadModelUrl = 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin'

export const llmApiKeyUrl = 'https://console.anthropic.com/settings/keys'
export const llmDefaultMaxTokens = 8192 // https://docs.anthropic.com/en/docs/about-claude/models
export const llmLimitsUrl = 'https://console.anthropic.com/settings/limits'
export const llmCostUrl = 'https://console.anthropic.com/settings/cost'

export const ytDlpAssetNames: Record<string, string> = {
	'windows-x86_64': 'yt-dlp.exe',
	'windows-aarch64': 'yt-dlp_arm64.exe',
	'linux-x86_64': 'yt-dlp_linux',
	'linux-aarch64': 'yt-dlp_linux_aarch64',
	'macos-x86_64': 'yt-dlp_macos',
	'macos-aarch64': 'yt-dlp_macos',
}

export function ytDlpDownloadUrl(version: string, key: string): string {
	return `https://github.com/yt-dlp/yt-dlp/releases/download/${version}/${ytDlpAssetNames[key]}`
}

export const videoExtensions = ['mp4', 'mkv', 'avi', 'mov', 'wmv', 'webm', 'mxf']
export const audioExtensions = ['mp3', 'wav', 'aac', 'flac', 'oga', 'ogg', 'opic', 'opus', 'm4a', 'm4b', 'wma']
export const themes = ['light', 'dark']
export const defaultAutoSummarizeOnFinish = false

/**
 * Default dictation shortcut: Option+Space on macOS, Ctrl+Space elsewhere. Resolved on call rather
 * than at module scope: reading the platform at import time makes this file unimportable outside a
 * webview (and untestable).
 */
export function getDefaultHotkeyShortcut() {
	const isMac = navigator.platform.toUpperCase().includes('MAC')
	return isMac ? 'Alt+Space' : 'Ctrl+Space'
}

/** Global recording uses one portable accelerator on every desktop platform. */
export function getDefaultRecordingShortcut() {
	return 'CmdOrCtrl+Shift+R'
}
