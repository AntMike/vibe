import { invoke } from '@tauri-apps/api/core'
import { useState } from 'react'
import { toast } from 'sonner'
import { m } from '~/paraglide/messages.js'
import { CLI_COMMANDS, cliCommand, type HintsCli, type HintsRefiner, type HintsSettings } from '~/lib/hints'
import { getFriendlyModelName } from '~/lib/model'
import { Button } from '~/components/ui/button'
import { Input } from '~/components/ui/input'
import { Switch } from '~/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '~/components/ui/select'
import { SettingsGroup, SettingsNote, SettingsRow, rowControlClass, type SettingsViewModel } from './shared'

/** Select has no empty value; this stands for "no draft pass". */
const NO_DRAFT = 'none'

const cliLabels: Record<HintsCli, string> = { claude: 'Claude Code', codex: 'Codex', gemini: 'Gemini CLI', custom: '' }

/** Recognition hints: a glossary for Whisper from call names and an optional draft pass. */
export function HintsGroup({ vm }: { vm: SettingsViewModel }) {
	const { hints, setHints } = vm.preference
	const set = (patch: Partial<HintsSettings>) => setHints({ ...hints, ...patch })
	const [testing, setTesting] = useState(false)

	async function testCli() {
		setTesting(true)
		try {
			const reply = await invoke<string>('ask_cli', { command: cliCommand(hints), prompt: 'Reply with the single word OK.' })
			toast.success(m.hintsCliWorks({ reply: reply.slice(0, 40) }))
		} catch (error) {
			toast.error(String((error as { message?: string })?.message ?? error))
		} finally {
			setTesting(false)
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
						<SettingsRow label={m.hintsRefiner()} description={m.hintsRefinerInfo()}>
							<Select value={hints.refiner} onValueChange={(refiner: HintsRefiner) => set({ refiner })}>
								<SelectTrigger className={`w-56 ${rowControlClass}`}>
									<SelectValue />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value="words">{m.hintsRefinerWords()}</SelectItem>
									<SelectItem value="cli">{m.hintsRefinerCli()}</SelectItem>
									<SelectItem value="ai">{m.hintsRefinerAi()}</SelectItem>
								</SelectContent>
							</Select>
						</SettingsRow>
					)}
					{hints.draftModelPath && hints.refiner === 'ai' && <SettingsNote>{m.hintsRefinerAiNote()}</SettingsNote>}
					{hints.draftModelPath && hints.refiner === 'cli' && (
						<>
							<SettingsRow label={m.hintsCli()} description={hints.cli === 'custom' ? m.hintsCustomCommandInfo() : CLI_COMMANDS[hints.cli]}>
								<Select value={hints.cli} onValueChange={(cli: HintsCli) => set({ cli })}>
									<SelectTrigger className={`w-44 ${rowControlClass}`}>
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										{(Object.keys(cliLabels) as HintsCli[]).map((cli) => (
											<SelectItem key={cli} value={cli}>
												{cli === 'custom' ? m.hintsCustomCommand() : cliLabels[cli]}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
								<Button
									variant="outline"
									size="sm"
									className="rounded-lg"
									disabled={testing || !cliCommand(hints)}
									onClick={() => void testCli()}>
									{m.hintsTest()}
								</Button>
							</SettingsRow>
							{hints.cli === 'custom' && (
								<SettingsRow label={m.hintsCustomCommand()}>
									<Input
										value={hints.customCommand}
										onChange={(e) => set({ customCommand: e.target.value })}
										placeholder="my-ai --stdin"
										className={`w-64 font-mono ${rowControlClass}`}
									/>
								</SettingsRow>
							)}
						</>
					)}
				</>
			)}
		</SettingsGroup>
	)
}
