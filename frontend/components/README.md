# components/

Reusable, presentational React components, split by concern:

- `layout/` — the app shell (Navbar, Sidebar, Workspace, ActivityPanel,
  AppLayout). AppLayout owns cross-page shared state: `projects` (fetched
  from the real backend), `selectedProjectId`, `logs`, `activePage`, and
  `searchQuery` — the nearest common parent of Sidebar/Navbar and
  Workspace/ActivityPanel. `Workspace` is a thin, stateless page router
  (`switch (activePage)`).
- `project/` — the Projects page and everything nested under it:
  - `ProjectsPage` — page-level composition (list/detail view switch,
    create-modal state, search filtering).
  - `ProjectCard` / `ProjectList` / `ProjectModal` — the dashboard (list
    + create flow), backed by `services/projectService.ts`.
  - `ProjectWorkspace` / `ProjectTabs` / `OverviewTab` / `DatasetTab` /
    `CopilotTab` / `ModelsTab` / `ExperimentsTab` / `SettingsTab` — the
    tabbed single-project view. `CopilotTab` is where a real Copilot run
    happens (`services/copilotService.ts`); `ModelsTab`/`ExperimentsTab`
    are honest empty states pointing there, since a project isn't linked
    to a specific dataset id on the backend.
- `dataset/` — `DatasetsPage` (list + search + delete, real data) and
  `UploadDatasetModal` (real multipart upload to `POST /datasets`).
- `model/` — `ModelsPage`: an honest empty state. There's no backend
  endpoint that lists models across all datasets; real results live in
  each project's Copilot tab.
- `experiment/` — `ExperimentsPage`: same reasoning as `model/`.
- `settings/` — `SettingsPage`: real connection info only (the backend
  API URL this frontend is configured to use). No preference toggles
  that don't actually do anything.
- `ui/` — generic primitives with no domain knowledge, reused across
  every page above: `Card`, `Badge`, `Field`, `Button`, `Input`,
  `Textarea`, `Select`, `Modal`, `SearchBar`, `EmptyState`,
  `SectionHeader`, `Table`. `Badge` takes a label/tone pair — the
  status → label/tone mapping lives in `lib/statusMeta.ts`.

Keep components framework-agnostic of data-fetching where possible --
that belongs in `services/`.
