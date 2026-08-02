"use client";

import { useState } from "react";
import { setAuthToken, type Role } from "@/lib/api";

/** Demo sign-in: picks one of the three baseline roles and holds its bearer
 *  token in memory only (see lib/api.ts setAuthToken) — never written to
 *  localStorage/sessionStorage/cookies/URL, per final-project.pdf §7
 *  "Browser credential handling." A page reload always starts signed out. */
const DEMO_TOKENS: Record<Role, string> = {
  developer: "dev-token",
  project_manager: "pm-token",
  auditor: "audit-token",
};

const ROLE_LABELS: Record<Role, string> = {
  developer: "Developer",
  project_manager: "Project Manager",
  auditor: "Auditor",
};

export function RoleSelector({
  role,
  onChange,
}: {
  role: Role | null;
  onChange: (role: Role | null) => void;
}) {
  const [open, setOpen] = useState(false);

  function select(next: Role) {
    setAuthToken(DEMO_TOKENS[next]);
    onChange(next);
    setOpen(false);
  }

  function signOut() {
    setAuthToken(null);
    onChange(null);
    setOpen(false);
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="rounded-full border border-black/[.1] px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-black/[.04] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:border-white/[.15] dark:text-zinc-300 dark:hover:bg-white/[.08]"
      >
        {role ? ROLE_LABELS[role] : "Sign in (demo)"}
      </button>
      {open && (
        <ul
          role="listbox"
          aria-label="Select a demo role"
          className="absolute right-0 z-10 mt-1 w-44 rounded-lg border border-black/[.08] bg-white py-1 text-xs shadow-lg dark:border-white/[.145] dark:bg-zinc-900"
        >
          {(Object.keys(ROLE_LABELS) as Role[]).map((r) => (
            <li key={r}>
              <button
                type="button"
                role="option"
                aria-selected={role === r}
                onClick={() => select(r)}
                className="block w-full px-3 py-1.5 text-left text-zinc-700 hover:bg-black/[.04] focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-zinc-500 dark:text-zinc-300 dark:hover:bg-white/[.08]"
              >
                {ROLE_LABELS[r]}
              </button>
            </li>
          ))}
          {role && (
            <li className="mt-1 border-t border-black/[.06] pt-1 dark:border-white/[.1]">
              <button
                type="button"
                onClick={signOut}
                className="block w-full px-3 py-1.5 text-left text-zinc-500 hover:bg-black/[.04] dark:text-zinc-400 dark:hover:bg-white/[.08]"
              >
                Sign out
              </button>
            </li>
          )}
        </ul>
      )}
    </div>
  );
}
