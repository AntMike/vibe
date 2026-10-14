import { fetch as tauriFetch } from '@tauri-apps/plugin-http'

/**
 * Posting a finished call to its Slack huddle thread, with a user token (xoxp-) the user pastes in
 * Settings. A huddle leaves a `huddle_thread` message in its channel or DM when it starts; its
 * replies are the thread, so finding that message is finding the thread.
 *
 * User token scopes: chat:write, files:write, channels:history, groups:history, im:history,
 * mpim:history, channels:read, groups:read, im:read, mpim:read.
 */

export interface HuddleThread {
	channel: string
	ts: string
	/** `#general` for a channel; empty for a DM or a group DM. */
	label: string
}

interface HuddleRoom {
	date_start?: number
	date_end?: number
	participants?: string[]
	participant_history?: string[]
}

export interface SlackMessage {
	ts: string
	subtype?: string
	room?: HuddleRoom
}

interface Conversation {
	id: string
	name?: string
	is_channel?: boolean
	is_group?: boolean
	is_im?: boolean
	is_mpim?: boolean
}

/** Seconds a recording may start before Slack says the huddle did, or end after. */
const SLACK_CLOCK = 120

async function api<T>(token: string, method: string, params: Record<string, unknown>, json = false): Promise<T> {
	for (;;) {
		const response = await tauriFetch(`https://slack.com/api/${method}`, {
			method: 'POST',
			headers: { Authorization: `Bearer ${token}`, 'Content-Type': json ? 'application/json; charset=utf-8' : 'application/x-www-form-urlencoded' },
			body: json ? JSON.stringify(params) : new URLSearchParams(Object.entries(params).map(([key, value]) => [key, String(value)])).toString(),
		})
		if (response.status === 429) {
			const seconds = Number(response.headers.get('Retry-After') ?? 5)
			await new Promise((resolve) => setTimeout(resolve, seconds * 1000))
			continue
		}
		const body = await response.json()
		if (!body.ok) throw new Error(`Slack ${method}: ${body.error ?? response.status}`)
		return body as T
	}
}

/** The huddle message for a recording that started at `startedAt` (unix seconds), if `me` was in it. */
export function pickHuddle(messages: SlackMessage[], me: string, startedAt: number): SlackMessage | undefined {
	return messages.find(({ subtype, room }) => {
		if (subtype !== 'huddle_thread' || !room?.date_start) return false
		const people = [...(room.participant_history ?? []), ...(room.participants ?? [])]
		if (people.length > 0 && !people.includes(me)) return false
		const ended = room.date_end && room.date_end > 0 ? room.date_end : Infinity
		return room.date_start <= startedAt + SLACK_CLOCK && ended >= startedAt - SLACK_CLOCK
	})
}

/**
 * Look through every conversation the user is in for the huddle they were recording.
 */
// ponytail: one history call per conversation, four at a time; with hundreds of channels Slack's rate
// limit (~50/min) makes this take minutes. Read the huddle's channel from Slack's UI if that bites.
export async function findHuddleThread(token: string, startedAt: number): Promise<HuddleThread | null> {
	const { user_id: me } = await api<{ user_id: string }>(token, 'auth.test', {})
	const conversations: Conversation[] = []
	let cursor = ''
	do {
		const page = await api<{ channels: Conversation[]; response_metadata?: { next_cursor?: string } }>(token, 'users.conversations', {
			types: 'public_channel,private_channel,mpim,im',
			exclude_archived: true,
			limit: 200,
			...(cursor ? { cursor } : {}),
		})
		conversations.push(...page.channels)
		cursor = page.response_metadata?.next_cursor ?? ''
	} while (cursor)

	let found: HuddleThread | null = null
	const queue = [...conversations]
	const worker = async () => {
		for (let conversation = queue.shift(); conversation && !found; conversation = queue.shift()) {
			const { messages } = await api<{ messages: SlackMessage[] }>(token, 'conversations.history', {
				channel: conversation.id,
				// A huddle leaves its message when it starts: before the recording, within a day.
				oldest: startedAt - 24 * 3600,
				latest: startedAt + SLACK_CLOCK,
				limit: 200,
			}).catch(() => ({ messages: [] }))
			const huddle = pickHuddle(messages, me, startedAt)
			if (huddle && !found) {
				const named = (conversation.is_channel || conversation.is_group) && !conversation.is_mpim && conversation.name
				found = { channel: conversation.id, ts: huddle.ts, label: named ? `#${conversation.name}` : '' }
			}
		}
	}
	await Promise.all([worker(), worker(), worker(), worker()])
	return found
}

/** Reply in the thread with the summary, then attach the transcript file. */
export async function postToThread(token: string, thread: HuddleThread, summary: string | undefined, file: { name: string; bytes: Uint8Array }) {
	if (summary?.trim()) {
		await api(token, 'chat.postMessage', { channel: thread.channel, thread_ts: thread.ts, markdown_text: summary }, true)
	}
	const upload = await api<{ upload_url: string; file_id: string }>(token, 'files.getUploadURLExternal', {
		filename: file.name,
		length: file.bytes.byteLength,
	})
	const sent = await tauriFetch(upload.upload_url, { method: 'POST', body: new Uint8Array(file.bytes) })
	if (!sent.ok) throw new Error(`Slack upload: ${sent.status}`)
	await api(
		token,
		'files.completeUploadExternal',
		{ files: [{ id: upload.file_id, title: file.name }], channel_id: thread.channel, thread_ts: thread.ts },
		true,
	)
}
