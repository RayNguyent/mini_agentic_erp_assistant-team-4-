"use client";

/** Route/model/trace metadata for one assistant turn — required so a reviewer
 *  (or a curious user) can see *how* an answer was produced without opening
 *  devtools: which route the supervisor picked, which specialists ran, which
 *  provider/model answered, and the trace id to look up in /traces/{id}. */
export function MetadataBar({
  route,
  agentPath,
  provider,
  model,
  requestId,
}: {
  route: string | null;
  agentPath: string[];
  provider: string | null;
  model: string | null;
  requestId: string | null;
}) {
  if (!route && agentPath.length === 0 && !provider) return null;

  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-zinc-400 dark:text-zinc-500">
      {route && <span title="Route">route: {route}</span>}
      {agentPath.length > 0 && <span title="Specialist agents that ran">agents: {agentPath.join(" → ")}</span>}
      {provider && <span title="LLM provider">provider: {provider}</span>}
      {model && <span title="Model">model: {model}</span>}
      {requestId && (
        <span title="Trace id (GET /traces/{id})" className="font-mono">
          trace: {requestId.slice(0, 8)}
        </span>
      )}
    </div>
  );
}
