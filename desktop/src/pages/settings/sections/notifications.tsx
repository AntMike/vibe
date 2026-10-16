import { invoke } from '@tauri-apps/api/core'
import { platform } from '@tauri-apps/plugin-os'
import { toast } from 'sonner'
import { Button } from '~/components/ui/button'
import { Input } from '~/components/ui/input'
import { Switch } from '~/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '~/components/ui/select'
import { CONFIG_KEYS } from '~/lib/config-keys'
import type { SlackMethod, SummaryAs } from '~/lib/slack'
import type { TranscriptExportContent } from '~/lib/transcript-export'
import { usePersisted } from '~/lib/config-store'
import { m } from '~/paraglide/messages.js'
import { usePreferenceProvider } from '~/providers/preference'
import { SettingsGroup, SettingsNote, SettingsRow, rowControlClass } from './shared'

/** 0 keeps the alert up until it is dismissed or the call ends. */
const STAY_SECONDS = [0, 7, 15, 25, 60]

export function NotificationsSection() {
	const preference = usePreferenceProvider()
	const isWindows = platform() === 'windows'
	const [style, setStyle] = usePersisted<'popup' | 'system'>(CONFIG_KEYS.notificationsMeetingStyle, 'popup')
	const [seconds, setSeconds] = usePersisted(CONFIG_KEYS.notificationsMeetingSeconds, 0)
	const [sound, setSound] = usePersisted(CONFIG_KEYS.notificationsMeetingSound, true)
	const [autoRecord, setAutoRecord] = usePersisted(CONFIG_KEYS.notificationsAutoRecord, true)
	const [dictation, setDictation] = usePersisted(CONFIG_KEYS.notificationsDictation, true)
	const [slackMethod, setSlackMethod] = usePersisted<SlackMethod>(CONFIG_KEYS.slackMethod, 'off')
	const [slackToken, setSlackToken] = usePersisted(CONFIG_KEYS.slackToken, '')
	const [slackSummaryAs, setSlackSummaryAs] = usePersisted<SummaryAs>(CONFIG_KEYS.slackSummaryAs, 'message')
	const [slackContent, setSlackContent] = usePersisted<TranscriptExportContent>(CONFIG_KEYS.slackContent, 'both')
	const [slackWithoutAsking, setSlackWithoutAsking] = usePersisted(CONFIG_KEYS.slackSendWithoutAsking, false)
	const system = isWindows && style === 'system'

	return (
		<div className="space-y-6">
			<SettingsGroup title={m.notifyMeetingGroup()}>
				{!preference.meetingDetectionEnabled && <SettingsNote>{m.notifyMeetingDetectionOff()}</SettingsNote>}
				{isWindows && (
					<SettingsRow label={m.notifyStyle()} description={system ? m.notifyStyleSystemInfo() : m.notifyStylePopupInfo()}>
						<Select value={style} onValueChange={(value: 'popup' | 'system') => setStyle(value)}>
							<SelectTrigger className={`w-52 ${rowControlClass}`}>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="popup">{m.notifyStylePopup()}</SelectItem>
								<SelectItem value="system">{m.notifyStyleSystem()}</SelectItem>
							</SelectContent>
						</Select>
					</SettingsRow>
				)}
				<SettingsRow label={m.notifyStay()} description={system ? m.notifyStaySystemInfo() : undefined}>
					<Select value={String(seconds)} onValueChange={(value) => setSeconds(Number(value))}>
						<SelectTrigger className={`w-52 ${rowControlClass}`}>
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							{STAY_SECONDS.map((value) => (
								<SelectItem key={value} value={String(value)}>
									{value === 0 ? m.notifyStayUntilDismissed() : m.notifyStaySeconds({ seconds: String(value) })}
								</SelectItem>
							))}
						</SelectContent>
					</Select>
				</SettingsRow>
				{system && (
					<SettingsRow label={m.notifySound()}>
						<Switch checked={sound} onCheckedChange={setSound} aria-label={m.notifySound()} />
					</SettingsRow>
				)}
				<SettingsRow label={m.notifyTest()} description={m.notifyTestInfo()}>
					<Button
						variant="outline"
						size="sm"
						className="rounded-lg"
						onClick={() => void invoke('test_meeting_notification').catch((error) => toast.error(String(error)))}>
						{m.notifyTestButton()}
					</Button>
				</SettingsRow>
			</SettingsGroup>

			<SettingsGroup title={m.notifyOtherGroup()}>
				{isWindows && (
					<SettingsRow label={m.notifyAutoRecord()} description={m.notifyAutoRecordInfo()}>
						<Switch checked={autoRecord} onCheckedChange={setAutoRecord} aria-label={m.notifyAutoRecord()} />
					</SettingsRow>
				)}
				<SettingsRow label={m.notifyDictation()} description={m.notifyDictationInfo()}>
					<Switch checked={dictation} onCheckedChange={setDictation} aria-label={m.notifyDictation()} />
				</SettingsRow>
				<SettingsRow label={m.playSoundOnFinish()}>
					<Switch checked={preference.soundOnFinish} onCheckedChange={preference.setSoundOnFinish} aria-label={m.playSoundOnFinish()} />
				</SettingsRow>
				<SettingsRow label={m.focusWindowOnFinish()}>
					<Switch checked={preference.focusOnFinish} onCheckedChange={preference.setFocusOnFinish} aria-label={m.focusWindowOnFinish()} />
				</SettingsRow>
			</SettingsGroup>

			{isWindows && (
				<SettingsGroup title={m.slackGroup()}>
					<SettingsRow label={m.slackMethod()} description={slackMethod === 'connector' ? m.slackConnectorInfo() : m.slackMethodInfo()}>
						<Select value={slackMethod} onValueChange={(value: SlackMethod) => setSlackMethod(value)}>
							<SelectTrigger className={`w-64 ${rowControlClass}`}>
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="off">{m.slackMethodOff()}</SelectItem>
								<SelectItem value="token">{m.slackMethodToken()}</SelectItem>
								<SelectItem value="connector">{m.slackMethodConnector()}</SelectItem>
							</SelectContent>
						</Select>
					</SettingsRow>
					{slackMethod === 'token' && (
						<SettingsRow label={m.slackToken()} description={m.slackTokenInfo()}>
							<Input
								value={slackToken}
								onChange={(event) => setSlackToken(event.target.value)}
								placeholder={m.slackTokenPlaceholder()}
								type="password"
								className={`w-64 ${rowControlClass}`}
							/>
						</SettingsRow>
					)}
					{slackMethod !== 'off' && (
						<>
							<SettingsRow label={m.slackContent()} description={m.slackContentInfo()}>
								<Select value={slackContent} onValueChange={(value: TranscriptExportContent) => setSlackContent(value)}>
									<SelectTrigger className={`w-64 ${rowControlClass}`}>
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="both">{m.slackSendBoth()}</SelectItem>
										<SelectItem value="summary">{m.slackSendSummary()}</SelectItem>
										<SelectItem value="transcript">{m.slackSendTranscript()}</SelectItem>
									</SelectContent>
								</Select>
							</SettingsRow>
							<SettingsRow label={m.slackSummaryAs()}>
								<Select value={slackSummaryAs} onValueChange={(value: SummaryAs) => setSlackSummaryAs(value)}>
									<SelectTrigger className={`w-64 ${rowControlClass}`}>
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="message">{m.slackSummaryAsMessage()}</SelectItem>
										<SelectItem value="file">{m.slackSummaryAsFile()}</SelectItem>
									</SelectContent>
								</Select>
							</SettingsRow>
							<SettingsRow label={m.slackSendWithoutAsking()} description={m.slackSendWithoutAskingInfo()}>
								<Switch checked={slackWithoutAsking} onCheckedChange={setSlackWithoutAsking} aria-label={m.slackSendWithoutAsking()} />
							</SettingsRow>
						</>
					)}
				</SettingsGroup>
			)}
		</div>
	)
}
