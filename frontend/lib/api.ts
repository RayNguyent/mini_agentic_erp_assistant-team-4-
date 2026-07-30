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

export type ChatResponse = {
  answer: string;
  tool_used: string | null;
  approval_required: boolean;
  approval_id: string | null;
  error_code: string | null;
  risk_draft: RiskDraft | null;
};

export type ChatTurn = { role: "user" | "assistant"; content: string };

type StreamHandlers = {
  onStatus?: (status: string) => void;
  onToken: (text: string) => void;
  onDone: (payload: ChatResponse) => void;
};

async function streamSSE(
  path: string,
  body: unknown,
  { onStatus, onToken, onDone }: StreamHandlers
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) {
    throw new Error(`Request to ${path} failed with status ${res.status}`);
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

      try {
        const parsed = JSON.parse(data);
        if (event === "status" && onStatus) onStatus(parsed.message as string);
        else if (event === "token") onToken(parsed.text as string);
        else if (event === "done") onDone(parsed as ChatResponse);
      } catch (e) {
        console.error(`Failed to parse ${event} event:`, data, e);
        throw e;
      }
    }
  }
}

export function streamChat(
  message: string,
  history: ChatTurn[] | undefined,
  handlers: StreamHandlers
) {
  return streamSSE("/chat/stream", { message, history: history ?? null }, handlers);
}

export function streamApprove(
  approvalId: string,
  handlers: StreamHandlers,
  toolInput?: Record<string, unknown>
) {
  return streamSSE(
    "/approve/stream",
    { approval_id: approvalId, tool_input: toolInput ?? null },
    handlers
  );
}

export function streamReject(approvalId: string, handlers: StreamHandlers) {
  return streamSSE("/reject/stream", { approval_id: approvalId }, handlers);
}
