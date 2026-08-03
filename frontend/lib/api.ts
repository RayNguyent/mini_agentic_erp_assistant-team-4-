export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type RiskSeverity = "low" | "medium" | "high";

/** Pre-fill for the create_risk approval form — whatever the backend could
 *  extract from the message, blank for whatever it could not. */
export type RiskDraft = {
  project_code: string;
  title: string;
  severity: RiskSeverity;
  description: string;
};

export type Citation = {
  doc_id: string;
  chunk_id: string;
  title: string;
  heading: string;
  snippet: string;
  score: number;
  classification: string;
};

export type ChatResponse = {
  answer: string;
  tool_used: string | null;
  approval_required: boolean;
  approval_id: string | null;
  error_code: string | null;
  risk_draft: RiskDraft | null;
  route: string | null;
  agent_path: string[];
  citations: Citation[];
  request_id: string | null;
  provider: string | null;
  model: string | null;
};

export type ChatTurn = { role: "user" | "assistant"; content: string };

export type Role = "developer" | "project_manager" | "auditor";

/** Demo token held in memory only for the session — never persisted to
 *  localStorage/sessionStorage/cookies. See app/security/auth.py; a page
 *  reload always returns to the "not signed in" state by design. */
let authToken: string | null = null;

export function setAuthToken(token: string | null) {
  authToken = token;
  console.log("Auth token set to:", token);
}

export function getAuthToken(): string | null {
  return authToken;
}

function authHeaders(): Record<string, string> {
  const headers = authToken ? { Authorization: `Bearer ${authToken}` } : {};
  console.log("Auth headers:", headers, "Token:", authToken);
  return headers;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parseErrorBody(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return typeof body?.detail === "string" ? body.detail : res.statusText;
  } catch {
    return res.statusText;
  }
}

type StreamHandlers = {
  onStatus?: (status: string) => void;
  onRoute?: (route: string, agentPath: string[]) => void;
  onCitation?: (citation: Citation) => void;
  onToken: (text: string) => void;
  onDone: (payload: ChatResponse) => void;
};

async function streamSSE(
  path: string,
  body: unknown,
  { onStatus, onRoute, onCitation, onToken, onDone }: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new ApiError(res.status, await parseErrorBody(res));
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      let event = "message";
      let data = "";
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data = line.slice(6);
      }
      if (!data) continue;

      const parsed = JSON.parse(data);
      if (event === "status" && onStatus) onStatus(parsed.message as string);
      else if (event === "route" && onRoute) onRoute(parsed.route, parsed.agent_path ?? []);
      else if (event === "citation" && onCitation) onCitation(parsed as Citation);
      else if (event === "token") onToken(parsed.text as string);
      else if (event === "done") onDone(parsed as ChatResponse);
    }
  }
}

export function streamChat(
  message: string,
  history: ChatTurn[] | undefined,
  handlers: StreamHandlers,
  signal?: AbortSignal
) {
  return streamSSE("/chat/stream", { message, history: history ?? null }, handlers, signal);
}

export function streamApprove(
  approvalId: string,
  handlers: StreamHandlers,
  toolInput?: Record<string, unknown>,
  signal?: AbortSignal
) {
  return streamSSE(
    "/approve/stream",
    { approval_id: approvalId, tool_input: toolInput ?? null },
    handlers,
    signal
  );
}

export function streamReject(approvalId: string, handlers: StreamHandlers, signal?: AbortSignal) {
  return streamSSE("/reject/stream", { approval_id: approvalId }, handlers, signal);
}

export type ReadinessResponse = {
  status: "ok" | "degraded";
  llm_provider: string;
  model: string | null;
  base_url: string | null;
  credential_configured: boolean;
  tokenizer: string;
  rag_index_chunks: number;
  rag_vector_index: string;
  rag_vector_count: number;
  embeddings_enabled: boolean;
  erp_provider: string;
};

export async function fetchReadiness(): Promise<ReadinessResponse> {
  const res = await fetch(`${API_BASE_URL}/readiness`);
  if (!res.ok) throw new ApiError(res.status, await parseErrorBody(res));
  return res.json();
}
