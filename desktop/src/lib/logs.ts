import { app } from '@tauri-apps/api'

export async function getPrettyVersion() {
	const appVersion = await app.getVersion()
	const appName = await app.getName()
	return `${appName} ${appVersion}`
}
