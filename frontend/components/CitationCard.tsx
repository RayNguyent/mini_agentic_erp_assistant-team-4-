"use client";

import { useState } from "react";
import type { Citation } from "@/lib/api";

/** A grounded answer's source card: doc title, section, and an expandable
 *  snippet with its relevance score. Renders nothing when there are no
 *  citations — a refusal or a tool-backed answer legitimately has none. */
export function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-3 flex flex-col gap-1.5" aria-label="Sources">
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
        Sources ({citations.length})
      </span>
      <ul className="flex flex-col gap-1.5">
        {citations.map((citation) => (
          <li key={citation.chunk_id}>
            <CitationCard citation={citation} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function CitationCard({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false);
  const detailId = `citation-detail-${citation.chunk_id}`;

  return (
    <div className="rounded-lg border border-black/[.08] bg-black/[.02] px-3 py-2 text-xs dark:border-white/[.1] dark:bg-white/[.03]">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        aria-controls={detailId}
        className="flex w-full items-center justify-between gap-2 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 rounded"
      >
        <span className="min-w-0 flex-1">
          <span className="font-medium text-zinc-700 dark:text-zinc-300">{citation.title}</span>
          {citation.heading && (
            <span className="text-zinc-500 dark:text-zinc-400"> — {citation.heading}</span>
          )}
        </span>
        <span className="shrink-0 rounded-full bg-black/[.06] px-2 py-0.5 text-[10px] tabular-nums text-zinc-500 dark:bg-white/[.08] dark:text-zinc-400">
          {(citation.score * 100).toFixed(0)}%
        </span>
      </button>
      {expanded && (
        <p id={detailId} className="mt-2 text-zinc-600 dark:text-zinc-400">
          {citation.snippet}
          <span className="ml-1 text-zinc-400 dark:text-zinc-500">[{citation.chunk_id}]</span>
        </p>
      )}
    </div>
  );
}
