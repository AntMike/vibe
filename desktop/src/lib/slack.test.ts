import { describe, expect, it, vi } from 'vitest'
import { pickHuddle, type SlackMessage } from './slack'

vi.mock('@tauri-apps/plugin-http', () => ({ fetch: vi.fn() }))

const huddle = (ts: string, start: number, end: number, people: string[]): SlackMessage => ({
	ts,
	subtype: 'huddle_thread',
	room: { date_start: start, date_end: end, participant_history: people },
})

describe('pickHuddle', () => {
	const startedAt = 10_000

	it('finds the huddle I was in that was running when the recording started', () => {
		const messages: SlackMessage[] = [
			{ ts: '1' },
			huddle('2', 5_000, 6_000, ['me']), // ended long before
			huddle('3', 9_000, 0, ['someone']), // ongoing, but not mine
			huddle('4', 9_950, 0, ['someone', 'me']), // ongoing and mine
		]
		expect(pickHuddle(messages, 'me', startedAt)?.ts).toBe('4')
	})

	it('allows for Slack and the recording disagreeing by a minute or two', () => {
		expect(pickHuddle([huddle('1', startedAt + 60, startedAt + 600, ['me'])], 'me', startedAt)?.ts).toBe('1')
		expect(pickHuddle([huddle('1', startedAt + 600, startedAt + 900, ['me'])], 'me', startedAt)).toBeUndefined()
	})

	it('trusts the time alone when Slack lists no participants', () => {
		expect(pickHuddle([huddle('1', 9_000, 11_000, [])], 'me', startedAt)?.ts).toBe('1')
	})
})
