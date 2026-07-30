"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  ChatResponse,
  ChatTurn,
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
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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

  function finishMessage(id: string, payload: ChatResponse) {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id
          ? {
              ...m,
              text: payload.answer,
              streaming: false,
              isError: Boolean(payload.error_code),
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

    try {
      await streamChat(message, history, {
        onStatus: (status) => addStatus(assistantId, status),
        onToken: (text) => appendToken(assistantId, text),
        onDone: (payload) => {
          console.log("Done received:", payload);
          finishMessage(assistantId, payload);
        },
      });
    } catch (error) {
      console.error("Stream error:", error);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, streaming: false, isError: true, text: "Something went wrong talking to the assistant." }
            : m
        )
      );
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

    try {
      if (decision === "approved") {
        await streamApprove(
          approvalId,
          {
            onStatus: (status) => addStatus(resultId, status),
            onToken: (text) => appendToken(resultId, text),
            onDone: (payload) => finishMessage(resultId, payload),
          },
          riskDraft
            ? {
                project_code: riskDraft.project_code.trim(),
                risk_payload: {
                  title: riskDraft.title.trim(),
                  severity: riskDraft.severity,
                  description: riskDraft.description.trim(),
                },
              }
            : undefined
        );
      } else {
        await streamReject(approvalId, {
          onStatus: (status) => addStatus(resultId, status),
          onToken: (text) => appendToken(resultId, text),
          onDone: (payload) => finishMessage(resultId, payload),
        });
      }
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === resultId
            ? { ...m, streaming: false, isError: true, text: "Something went wrong resolving the approval." }
            : m
        )
      );
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <div className="flex flex-1 flex-col bg-zinc-50 dark:bg-black">
      <header className="border-b border-black/[.08] dark:border-white/[.145] py-4 px-6">
        <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
          Mini Agentic ERP Assistant
        </h1>
      </header>

      <main className="flex flex-1 flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
            {messages.length === 0 && (
              <p className="text-center text-sm text-zinc-500 dark:text-zinc-400 mt-16">
                Ask about project status, list risks, or create a new risk.
              </p>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
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
                      <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-current align-middle" />
                    )}
                  </div>

                  {m.approval && (
                    <div className="mt-3 flex flex-col gap-2">
                      {m.approval.resolution === "pending" && m.approval.riskDraft && (
                        <div className="flex flex-col gap-2 rounded-xl bg-black/[.03] p-3 dark:bg-white/[.06]">
                          <input
                            value={m.approval.riskDraft.project_code}
                            onChange={(e) => updateRiskDraft(m.id, { project_code: e.target.value })}
                            placeholder="Project code (e.g. PRJ-001)"
                            className="rounded-lg border border-black/[.1] bg-white px-3 py-1.5 text-xs text-zinc-900 outline-none dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
                          />
                          <input
                            value={m.approval.riskDraft.title}
                            onChange={(e) => updateRiskDraft(m.id, { title: e.target.value })}
                            placeholder="Risk title"
                            className="rounded-lg border border-black/[.1] bg-white px-3 py-1.5 text-xs text-zinc-900 outline-none dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
                          />
                          <select
                            value={m.approval.riskDraft.severity}
                            onChange={(e) =>
                              updateRiskDraft(m.id, { severity: e.target.value as "low" | "medium" | "high" })
                            }
                            className="rounded-lg border border-black/[.1] bg-white px-3 py-1.5 text-xs text-zinc-900 outline-none dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
                          >
                            <option value="low">Low severity</option>
                            <option value="medium">Medium severity</option>
                            <option value="high">High severity</option>
                          </select>
                          <textarea
                            value={m.approval.riskDraft.description}
                            onChange={(e) => updateRiskDraft(m.id, { description: e.target.value })}
                            placeholder="Description (optional)"
                            rows={2}
                            className="resize-none rounded-lg border border-black/[.1] bg-white px-3 py-1.5 text-xs text-zinc-900 outline-none dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
                          />
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
                            className="rounded-full bg-emerald-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                          >
                            Approve
                          </button>
                          <button
                            onClick={() => handleApproval(m.id, "rejected")}
                            disabled={isBusy}
                            className="rounded-full border border-black/[.15] px-4 py-1.5 text-xs font-medium hover:bg-black/[.04] disabled:opacity-50 dark:border-white/[.2] dark:hover:bg-white/[.08]"
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
          className="border-t border-black/[.08] dark:border-white/[.145] p-4"
        >
          <div className="mx-auto flex w-full max-w-3xl items-end gap-2">
            <textarea
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
              className="flex-1 resize-none rounded-2xl border border-black/[.1] bg-white px-4 py-3 text-sm text-zinc-900 outline-none focus:border-zinc-400 dark:border-white/[.15] dark:bg-zinc-900 dark:text-zinc-50"
            />
            <button
              type="submit"
              disabled={isBusy || !input.trim()}
              className="rounded-full bg-zinc-900 px-5 py-3 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
            >
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
