import { invoke } from '@tauri-apps/api/core'
import { useEffect } from 'react'
import { m } from '~/paraglide/messages.js'

/**
 * Hands Rust the notification text in the app's language, since it has no translations of its own:
 * the Windows meeting notification and the notices about calls recorded without asking. Runs again
 * on a language change, like the tray labels.
 */
export function useNotificationLabels(locale: string) {
	useEffect(() => {
		void invoke('set_notification_labels', {
			labels: {
				meetingTitle: m.meetingPromptTitle({ source: '{source}' }),
				meetingBody: m.meetingPromptDescription(),
				record: m.meetingPromptStart(),
				dismiss: m.meetingPromptDismiss(),
				recordingStarted: m.notifyRecordingStarted({ source: '{source}' }),
				recordingStopped: m.notifyRecordingStopped(),
			},
		}).catch((error) => console.error('failed to hand over notification labels:', error))
	}, [locale])
}
