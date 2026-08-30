# services/

Client-side API integration layer — functions that call the FastAPI backend.
Keeping API calls here (instead of inline in components) makes it easy to
swap the backend URL/transport later, and means every component gets typed
responses instead of raw `fetch()` JSON.

- `api.ts` — shared `apiRequest()` helper (base URL from
  `NEXT_PUBLIC_API_URL`, JSON parsing, `ApiError`, multipart/form-data
  support). Every other file here builds on this instead of calling
  `fetch()` directly.
- `projectService.ts` — the project endpoints (`GET/POST/PUT/DELETE /projects`).
- `datasetService.ts` — the dataset endpoints (`GET/POST/DELETE /datasets`),
  including real multipart file upload.
- `copilotService.ts` — the Copilot endpoints (`startCopilotRun`,
  `continueCopilotRun`, `getCopilotRun`), used by
  `components/project/CopilotTab.tsx`.
