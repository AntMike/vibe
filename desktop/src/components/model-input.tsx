import { useCallback, useEffect, useState } from 'react'
import { ModelGlyph } from '~/components/brand-glyph'
import { Label } from '~/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '~/components/ui/select'
import { getFriendlyModelName, listInstalledModels, type InstalledModel } from '~/lib/model'
import { m } from '~/paraglide/messages.js'
import { useModelGates } from '~/providers/model-gates'
import { usePreferenceProvider } from '~/providers/preference'

/** The installed models as a picker, laid out like the language one, so a run can switch models in place. */
export default function ModelInput() {
	const preference = usePreferenceProvider()
	const { selectModel } = useModelGates()
	const [models, setModels] = useState<InstalledModel[]>([])

	const load = useCallback(() => {
		void listInstalledModels()
			.then((installed) => setModels(installed.filter((model) => model.valid)))
			.catch((error) => console.error('failed to list models:', error))
	}, [])
	useEffect(load, [load])

	const nameOf = (model: InstalledModel) => preference.modelDisplayNames[model.path] ?? getFriendlyModelName(model.name)

	return (
		<div className="w-full space-y-2">
			<Label>{m.model()}</Label>
			<Select value={preference.modelPath ?? undefined} onValueChange={(path) => void selectModel(path)} onOpenChange={(open) => open && load()}>
				<SelectTrigger className="h-9 w-full rounded-lg" aria-label={m.model()}>
					<SelectValue placeholder={m.selectModel()} />
				</SelectTrigger>
				<SelectContent>
					{models.map((model) => (
						<SelectItem key={model.path} value={model.path}>
							<span className="flex items-center gap-2">
								<ModelGlyph name={model.name} className="text-muted-foreground" />
								{nameOf(model)}
							</span>
						</SelectItem>
					))}
				</SelectContent>
			</Select>
		</div>
	)
}
