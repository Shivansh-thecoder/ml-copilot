# lib/

Framework-agnostic utilities and shared types that don't belong to a
specific component or service.

- `types.ts` — every shared TypeScript type: `Project`, `Dataset`,
  `CopilotRun`/`CopilotResult`, plus the small UI-only types
  (`ProjectTabId`, `PageId`, `BadgeTone`). Types that mirror a backend
  response use this app's camelCase convention; `services/*.ts` map the
  backend's snake_case JSON into them.
- `statusMeta.ts` — maps `ProjectStatus` values to a `{ label, tone }`
  pair for `Badge`, so the generic badge component never has to know
  what "archived" means.
