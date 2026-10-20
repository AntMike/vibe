/** The pre-3.2 shape of `summarize.llm`, kept only for migration. */
export interface LegacyLlmConfig {
	platform?: string
	enabled?: boolean
	model?: string
	prompt?: string
	contextTokens?: number
	maxTokens?: number
	claudeApiKey?: string
	ollamaBaseUrl?: string
	openaiBaseUrl?: string
	openaiApiKey?: string
}

/**
 * AI in Vibe: one connection, separate tasks.
 *
 * The connection is where the model lives. A task is one thing Vibe asks it to do, with
 * its own switch and its own prompt, so tuning the summary never changes what dictation
 * does to a sentence. Prompts belong to tasks, so changing the platform keeps them.
 */

export type AiPlatform = 'ollama' | 'claude' | 'openai' | 'cli'
/** A command-line AI, which runs on the user's own subscription instead of an API key. */
export type AiCli = 'claude' | 'codex' | 'gemini' | 'custom'

export interface AiConnection {
	platform: AiPlatform
	model: string
	contextTokens: number
	claudeApiKey: string
	ollamaBaseUrl: string
	openaiBaseUrl: string
	openaiApiKey: string
	/** For `platform: 'cli'`; absent in settings saved before it existed. */
	cli?: AiCli
	/** Reads the prompt on stdin and prints the answer, for `cli: 'custom'`. */
	cliCommand?: string
}

/** Each reads the prompt on stdin and prints only the answer; an empty model keeps the tool's own default. */
const CLI_COMMANDS: Record<Exclude<AiCli, 'custom'>, (model: string) => string> = {
	// Lean: no tools, MCP servers, settings, hooks or saved session — ~7k tokens a call instead of ~43k.
	claude: (model) => `claude -p${model ? ` --model ${model}` : ''} --tools "" --strict-mcp-config --setting-sources "" --no-session-persistence`,
	codex: (model) => `codex exec --skip-git-repo-check -s read-only --color never${model ? ` -m ${model}` : ''} -`,
	gemini: (model) => `gemini${model ? ` -m ${model}` : ''}`,
}

/** The shell command for a CLI connection. A model name that isn't a plain identifier is left out rather than reach the shell. */
export function cliCommand(connection: AiConnection) {
	const cli = connection.cli ?? 'claude'
	if (cli === 'custom') return connection.cliCommand?.trim() ?? ''
	const model = /^[\w.:/[\]-]+$/.test(connection.model.trim()) ? connection.model.trim() : ''
	return CLI_COMMANDS[cli](model)
}

export interface AiTask {
	enabled: boolean
	/** A preset id from `presets`, or `custom` once the prompt was edited. */
	preset: string
	prompt: string
}

export interface AiSettings {
	connection: AiConnection
	tasks: {
		/** `clickbaitTitle` names an auto-named project with a clickbait headline instead of a plain title. */
		summary: AiTask & { autoOnFinish: boolean; clickbaitTitle?: boolean }
		dictation: AiTask
	}
}

export const PLACEHOLDERS = {
	transcript: '{transcript}',
	text: '{text}',
	language: '{language}',
	speakers: '{speakers}',
} as const

const RULE = 'Output only the requested content. No introductions, explanations, or commentary.'
/**
 * A skeleton instead of a description: shorter, and every recording of a type comes out with the
 * same sections in the same order. Headings follow the meeting, not the app language.
 */
const FILL =
	'Fill this markdown template from the transcript. Keep its sections, order, heading levels and list styles exactly. Write everything, headings included, in the main language of the transcript. Leave out a section with nothing to put in it.'

function template(skeleton: string) {
	return `${RULE} ${FILL}\n\n${skeleton}\n\nTranscript:\n"""\n{transcript}\n"""`
}

export interface Preset {
	id: string
	task: 'summary' | 'dictation'
	prompt: string
}

/** Starting points. Picking one fills the prompt; editing the prompt makes it "custom". */
export const presets: Preset[] = [
	{
		id: 'summary',
		task: 'summary',
		prompt: template('## Overview\n<2-3 sentences>\n\n## Key points\n- <3-5 points>\n\n## Action items\n- [ ] <who>: <what>'),
	},
	{
		id: 'meeting-notes',
		task: 'summary',
		prompt: template('## <topic> (one section per topic)\n- <key point>\n- **Decision:** <decision>\n\n## Action items\n- [ ] <who>: <what>'),
	},
	{
		id: 'tldr',
		task: 'summary',
		prompt: template('**TL;DR:** <one paragraph>\n\n- **<takeaway>** (3-5, most important first)'),
	},
	{
		id: 'translate',
		task: 'summary',
		prompt: `${RULE}\n\nTranslate this transcript into {language}, keeping its structure. Markdown headings and paragraphs, no code fences.\n\n"""\n{transcript}\n"""`,
	},
	{
		id: 'article',
		task: 'summary',
		prompt: template('# <title>\n<intro paragraph>\n\n## <section>\n<paragraphs>'),
	},
	{
		id: 'punctuation',
		task: 'dictation',
		prompt: `${RULE}\n\nThis is dictated speech. Fix punctuation, capitalization and obvious speech-to-text mistakes. Keep every word and the original language; do not add, remove, translate, summarize or rephrase anything. Return the corrected text only.\n\n"""\n{text}\n"""`,
	},
	{
		id: 'formal',
		task: 'dictation',
		prompt: `${RULE}\n\nThis is dictated speech. Rewrite it as polished, formal prose in the same language, keeping the meaning and every fact. Return the text only.\n\n"""\n{text}\n"""`,
	},
	{
		id: 'casual',
		task: 'dictation',
		prompt: `${RULE}\n\nThis is dictated speech. Make it read like a relaxed chat message in the same language, keeping the meaning. Return the text only.\n\n"""\n{text}\n"""`,
	},
]

export function presetPrompt(id: string) {
	return presets.find((preset) => preset.id === id)?.prompt ?? ''
}

/** A preset's text comes from here, not the saved copy, so improving a preset reaches everyone who didn't edit theirs. */
export function withCurrentPresets(ai: AiSettings): AiSettings {
	const refresh = <T extends AiTask>(task: T): T => ({ ...task, prompt: presetPrompt(task.preset) || task.prompt })
	return { ...ai, tasks: { summary: refresh(ai.tasks.summary), dictation: refresh(ai.tasks.dictation) } }
}

export const DEFAULT_AI: AiSettings = {
	connection: {
		platform: 'claude',
		model: defaultModel('claude'),
		contextTokens: 65_536,
		claudeApiKey: '',
		ollamaBaseUrl: 'http://localhost:11434',
		openaiBaseUrl: 'https://api.openai.com/v1',
		openaiApiKey: '',
		cli: 'claude',
		cliCommand: '',
	},
	tasks: {
		summary: { enabled: false, autoOnFinish: false, preset: 'summary', prompt: presetPrompt('summary') },
		dictation: { enabled: false, preset: 'punctuation', prompt: presetPrompt('punctuation') },
	},
}

/** The default model for a platform, so switching platforms never leaves a Claude model on Ollama. */
export function defaultModel(platform: AiPlatform) {
	// A CLI starts on the tool's own default model until one is named.
	if (platform === 'cli') return ''
	return platform === 'ollama' ? 'gemma4:e2b' : platform === 'openai' ? 'gpt-5.6-luna' : 'claude-sonnet-5'
}

/** Old `%s` prompts become `{transcript}` or `{text}`; anything else is left alone. */
export function modernizePrompt(prompt: string, task: 'summary' | 'dictation') {
	return prompt.replace(/%s/g, task === 'summary' ? PLACEHOLDERS.transcript : PLACEHOLDERS.text)
}

/**
 * The single `summarize.llm` object that served both summaries and dictation becomes a
 * connection plus two tasks. Its prompt was written for summaries, so it goes there;
 * dictation gets the punctuation preset, on only if the old switch was on, since that
 * switch used to run the summary prompt over dictated text.
 */
export function migrateLegacy(legacy: LegacyLlmConfig | null | undefined, autoSummarizeOnFinish = false): AiSettings {
	if (!legacy || typeof legacy !== 'object') return DEFAULT_AI
	const platform: AiPlatform = legacy.platform === 'ollama' || legacy.platform === 'openai' ? legacy.platform : 'claude'
	const prompt = legacy.prompt?.trim() ? modernizePrompt(legacy.prompt, 'summary') : presetPrompt('summary')
	const isDefault = presets.some((preset) => preset.task === 'summary' && preset.prompt === prompt)
	return {
		connection: {
			platform,
			model: legacy.model || defaultModel(platform),
			contextTokens: legacy.contextTokens ?? DEFAULT_AI.connection.contextTokens,
			claudeApiKey: legacy.claudeApiKey ?? '',
			ollamaBaseUrl: legacy.ollamaBaseUrl || DEFAULT_AI.connection.ollamaBaseUrl,
			openaiBaseUrl: legacy.openaiBaseUrl || DEFAULT_AI.connection.openaiBaseUrl,
			openaiApiKey: legacy.openaiApiKey ?? '',
		},
		tasks: {
			summary: {
				enabled: Boolean(legacy.enabled),
				autoOnFinish: Boolean(legacy.enabled && autoSummarizeOnFinish),
				preset: isDefault ? 'summary' : 'custom',
				prompt,
			},
			dictation: { ...DEFAULT_AI.tasks.dictation, enabled: Boolean(legacy.enabled) },
		},
	}
}
