// Shared low-level HTTP client every services/*.ts file builds on — the
// one place that knows the backend's base URL, JSON conventions, and
// error shape, so individual service functions (copilotService,
// datasetService) stay focused on their own endpoints' request/response
// mapping instead of each re-implementing fetch/error handling.

// NEXT_PUBLIC_ prefix is required for Next.js to expose an env var to
// client-side code (this file runs in the browser, not just on the
// server) — see frontend/.env.local.example. Falls back to the local
// FastAPI dev server's default port so `npm run dev` works out of the
// box without requiring the env var for local development.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Thrown for every failure mode `apiRequest` can detect — network
// failure, non-2xx response, or a response body that isn't valid JSON —
// so every caller can catch ONE error type rather than three. `status`
// is 0 specifically for "never reached the server" (as opposed to a
// real HTTP status), letting callers distinguish "backend unavailable"
// from "backend returned an error" if they want to.
export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  // FormData bodies (dataset upload) must NOT get a manual Content-Type:
  // the browser sets one itself, including the multipart boundary value,
  // which we have no way to compute here. JSON bodies get the header
  // explicitly since fetch() doesn't infer it from a plain string.
  const isFormData = init?.body instanceof FormData;

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: isFormData ? init?.headers : { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    // fetch() itself throwing means the request never reached a server
    // at all (backend not running, wrong URL, network down, CORS
    // rejection at the browser level) — the one case with no HTTP
    // response to inspect.
    throw new ApiError(
      `Could not reach the backend at ${API_BASE_URL}. Is it running?`,
      0
    );
  }

  let body: unknown = null;
  const rawText = await response.text();
  if (rawText) {
    try {
      body = JSON.parse(rawText);
    } catch {
      throw new ApiError(
        `Backend returned a response that isn't valid JSON (status ${response.status}).`,
        response.status
      );
    }
  }

  if (!response.ok) {
    // FastAPI's HTTPException responses are {"detail": "..."} — surface
    // that message when present, otherwise fall back to the HTTP status
    // text rather than showing the user a raw stack trace or nothing.
    const detail = isRecord(body) && typeof body.detail === "string" ? body.detail : response.statusText;
    throw new ApiError(detail || `Request failed with status ${response.status}.`, response.status);
  }

  return body as T;
}
