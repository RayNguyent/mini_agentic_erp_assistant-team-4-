"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { CitationList } from "@/components/CitationCard";
import { MetadataBar } from "@/components/MetadataBar";
import { RoleSelector } from "@/components/RoleSelector";
import {
  ApiError,
  ChatResponse,
  ChatTurn,
  Citation,
  Role,
  RiskDraft,
  streamApprove,
  streamChat,
  streamReject,
} from "@/lib/api";

const HISTORY_TURNS = 8;

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
  isError?: boolean;
  statuses?: string[];
  route?: string | null;
  agentPath?: string[];
  citations?: Citation[];
  provider?: string | null;
  model?: string | null;
  requestId?: string | null;
  approval?: {
    approvalId: string;
    toolUsed: string | null;
    resolution: "pending" | "approved" | "rejected";
    riskDraft?: RiskDraft;
  };
};

const EMPTY_RISK_DRAFT: RiskDraft = {
  project_code: "",
  title: "",
  severity: "medium",
  description: "",
};

/** The form can only be submitted once the fields create_risk actually
 *  requires are filled; everything else is optional. */
function isRiskDraftComplete(draft: RiskDraft) {
  return Boolean(draft.project_code.trim() && draft.title.trim());
}

function newId() {
  return Math.random().toString(36).slice(2);
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [role, setRole] = useState<Role | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Cancel any in-flight request on unmount so a stream doesn't keep writing
  // to state after the page has gone away.
  useEffect(() => () => abortRef.current?.abort(), []);

  function appendToken(id: string, text: string) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, text: m.text + text } : m))
    );
  }

  function addStatus(id: string, status: string) {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, statuses: [...(m.statuses || []), status] } : m
      )
    );
  }

  function addCitation(id: string, citation: Citation) {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id ? { ...m, citations: [...(m.citations || []), citation] } : m
      )
    );
  }

  function setRoute(id: string, route: string, agentPath: string[]) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, route, agentPath } : m))
    );
  }

  function finishMessage(id: string, payload: ChatResponse) {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id
          ? {
              ...m,
              text: payload.answer,
              streaming: false,
              isError: Boolean(payload.error_code),
              route: payload.route,
              agentPath: payload.agent_path,
              citations: payload.citations,
              provider: payload.provider,
              model: payload.model,
              requestId: payload.request_id,
              approval: payload.approval_required
                ? {
                    approvalId: payload.approval_id!,
                    toolUsed: payload.tool_used,
                    resolution: "pending",
                    riskDraft:
                      payload.tool_used === "create_risk"
                        ? { ...EMPTY_RISK_DRAFT, ...(payload.risk_draft ?? {}) }
                        : undefined,
                  }
                : undefined,
            }
          : m
      )
    );
  }

  function reportStreamError(id: string, error: unknown) {
    const text =
      error instanceof ApiError && error.status === 401
        ? "You need to sign in (top right) before the assistant can respond."
        : error instanceof ApiError && error.status === 403
          ? "Your role doesn't have permission for that action."
          : "Something went wrong talking to the assistant.";
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, streaming: false, isError: true, text } : m))
    );
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const message = input.trim();
    if (!message || isBusy) return;

    const userMsg: ChatMessage = { id: newId(), role: "user", text: message };
    const assistantId = newId();
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      text: "",
      streaming: true,
    };

    const history: ChatTurn[] = messages
      .filter((m) => !m.streaming)
      .slice(-HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.text }));

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        message,
        history,
        {
          onStatus: (status) => addStatus(assistantId, status),
          onRoute: (route, agentPath) => setRoute(assistantId, route, agentPath),
          onCitation: (citation) => addCitation(assistantId, citation),
          onToken: (text) => appendToken(assistantId, text),
          onDone: (payload) => finishMessage(assistantId, payload),
        },
        controller.signal
      );
    } catch (error) {
      reportStreamError(assistantId, error);
    } finally {
      setIsBusy(false);
    }
  }

  function updateRiskDraft(sourceId: string, patch: Partial<RiskDraft>) {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === sourceId && m.approval?.riskDraft
          ? { ...m, approval: { ...m.approval, riskDraft: { ...m.approval.riskDraft, ...patch } } }
          : m
      )
    );
  }

  async function handleApproval(sourceId: string, decision: "approved" | "rejected") {
    const source = messages.find((m) => m.id === sourceId);
    if (!source?.approval || isBusy) return;
    if (
      decision === "approved" &&
      source.approval.riskDraft &&
      !isRiskDraftComplete(source.approval.riskDraft)
    ) {
      return;
    }

    const approvalId = source.approval.approvalId;
    const riskDraft = source.approval.riskDraft;
    setMessages((prev) =>
      prev.map((m) =>
        m.id === sourceId && m.approval
          ? { ...m, approval: { ...m.approval, resolution: decision } }
          : m
      )
    );

    const resultId = newId();
    setMessages((prev) => [
      ...prev,
      { id: resultId, role: "assistant", text: "", streaming: true },
    ]);
    setIsBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const handlers = {
      onStatus: (status: string) => addStatus(resultId, status),
      onRoute: (route: string, agentPath: string[]) => setRoute(resultId, route, agentPath),
      onCitation: (citation: Citation) => addCitation(resultId, citation),
      onToken: (text: string) => appendToken(resultId, text),
      onDone: (payload: ChatResponse) => finishMessage(resultId, payload),
    };

    try {
      if (decision === "approved") {
        await streamApprove(
          approvalId,
          handlers,
          riskDraft
            ? {
                project_code: riskDraft.project_code.trim(),
                risk_payload: {
                  title: riskDraft.title.trim(),
                  severity: riskDraft.severity,
                  description: riskDraft.description.trim(),
                },
              }
            : undefined,
          controller.signal
        );
      } else {
        await streamReject(approvalId, handlers, controller.signal);
      }
    } catch (error) {
      reportStreamError(resultId, error);
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <div className="flex flex-1 flex-col bg-zinc-50 dark:bg-black">
      <header className="flex items-center justify-between border-b border-black/[.08] dark:border-white/[.145] py-4 px-4 sm:px-6">
        <h1 className="text-base sm:text-lg font-semibold text-zinc-900 dark:text-zinc-50">
          Mini Agentic ERP Assistant
        </h1>
        <RoleSelector role={role} onChange={setRole} />
      </header>

      <main className="flex flex-1 flex-col overflow-hidden">
        <div
          className="flex-1 overflow-y-auto px-4 py-6 sm:px-6"
          role="log"
          aria-live="polite"
          aria-relevant="additions"
          aria-label="Conversation"
        >
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
            {messages.length === 0 && (
              <p className="text-center text-sm text-zinc-500 dark:text-zinc-400 mt-16">
                {role
                  ? "Ask about project status, sprint progress, budget, risks, or project documents."
                  : "Sign in (top right) to try the assistant, or ask as an unauthenticated guest — most actions will require a role."}
              </p>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[90%] sm:max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-zinc-900 text-zinc-50 dark:bg-zinc-100 dark:text-zinc-900"
                      : m.isError
                        ? "bg-red-50 text-red-700 border border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-900"
                        : "bg-white text-zinc-900 border border-black/[.08] dark:bg-zinc-900 dark:text-zinc-50 dark:border-white/[.145]"
                  }`}
                >
                  {m.statuses && m.statuses.length > 0 && (
                    <div className="mb-3 flex flex-col gap-1 pb-3 border-b border-current/[.2]">
                      {m.statuses.map((status, idx) => (
                        <div key={idx} className="text-xs opacity-75">
                          {status}
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="whitespace-pre-wrap">
                    {m.text}
                    {m.streaming && (
                      <span
                        aria-hidden="true"
                        className="ml-0.5 inline-block h-4 w-1.5 align-middle bg-current motion-safe:animate-pulse"
                      />
                    )}
                    {m.streaming && <span className="sr-only">Assistant is responding…</span>}
                  </div>

                  {m.role === "assistant" && !m.streaming && (
                    <CitationList citations={m.citations ?? []} />
                  )}

                  {m.role === "assistant" && !m.streaming && (
                    <MetadataBar
                      route={m.route ?? null}
                      agentPath={m.agentPath ?? []}
                      provider={m.provider ?? null}
                      model={m.model ?? null}
                      requestId={m.requestId ?? null}
                    />
                  )}

                  {m.approval && (
                    <div className="mt-3 flex flex-col gap-2">
                      {m.approval.resolution === "pending" && m.approval.riskDraft && (
                        <div className="flex flex-col gap-2 rounded-xl bg-black/[.03] p-3 dark:bg-white/[.06]">
                          <label className="flex flex-col gap-1 text-xs text-zinc-500 dark:text-zinc-400">
                            Project code
                            <input
                              value={m.approval.riskDraft.project_code}
                              onChange={(e) => updateRiskDraft(m.id, { project_code: e.target.value })}
                              placeholder="e.g. PRJ-001"
                              className="rounded-lg border border-black/[.1] bg-white px-3 py-1.5 text-xs text-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-zinc-500 dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-zinc-500 dark:text-zinc-400">
                            Risk title
                            <input
                              value={m.approval.riskDraft.title}
                              onChange={(e) => updateRiskDraft(m.id, { title: e.target.value })}
                              placeholder="Risk title"
                              className="rounded-lg border border-black/[.1] bg-white px-3 py-1.5 text-xs text-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-zinc-500 dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
                            />
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-zinc-500 dark:text-zinc-400">
                            Severity
                            <select
                              value={m.approval.riskDraft.severity}
                              onChange={(e) =>
                                updateRiskDraft(m.id, { severity: e.target.value as "low" | "medium" | "high" })
                              }
                              className="rounded-lg border border-black/[.1] bg-white px-3 py-1.5 text-xs text-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-zinc-500 dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
                            >
                              <option value="low">Low severity</option>
                              <option value="medium">Medium severity</option>
                              <option value="high">High severity</option>
                            </select>
                          </label>
                          <label className="flex flex-col gap-1 text-xs text-zinc-500 dark:text-zinc-400">
                            Description (optional)
                            <textarea
                              value={m.approval.riskDraft.description}
                              onChange={(e) => updateRiskDraft(m.id, { description: e.target.value })}
                              placeholder="Description (optional)"
                              rows={2}
                              className="resize-none rounded-lg border border-black/[.1] bg-white px-3 py-1.5 text-xs text-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-zinc-500 dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
                            />
                          </label>
                        </div>
                      )}

                      {m.approval.resolution === "pending" ? (
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleApproval(m.id, "approved")}
                            disabled={
                              isBusy ||
                              Boolean(m.approval.riskDraft && !isRiskDraftComplete(m.approval.riskDraft))
                            }
                            className="rounded-full bg-emerald-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-emerald-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700 disabled:opacity-50"
                          >
                            Approve
                          </button>
                          <button
                            onClick={() => handleApproval(m.id, "rejected")}
                            disabled={isBusy}
                            className="rounded-full border border-black/[.15] px-4 py-1.5 text-xs font-medium hover:bg-black/[.04] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:opacity-50 dark:border-white/[.2] dark:hover:bg-white/[.08]"
                          >
                            Reject
                          </button>
                        </div>
                      ) : (
                        <span className="text-xs text-zinc-500 dark:text-zinc-400">
                          {m.approval.resolution === "approved" ? "Approved" : "Rejected"}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={scrollRef} />
          </div>
        </div>

        <form
          onSubmit={handleSubmit}
          className="border-t border-black/[.08] dark:border-white/[.145] p-3 sm:p-4"
        >
          <div className="mx-auto flex w-full max-w-3xl items-end gap-2">
            <label htmlFor="chat-input" className="sr-only">
              Message the assistant
            </label>
            <textarea
              id="chat-input"
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Message the assistant..."
              rows={1}
              className="flex-1 resize-none rounded-2xl border border-black/[.1] bg-white px-4 py-3 text-sm text-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-zinc-500 dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
            />
            <button
              type="submit"
              disabled={isBusy || !input.trim()}
              className="rounded-full bg-zinc-900 px-4 sm:px-5 py-3 text-sm font-medium text-white hover:bg-zinc-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
            >
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
