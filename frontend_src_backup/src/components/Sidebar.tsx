// frontend/src/components/Sidebar.tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", hint: "Live view" },
  { href: "/setup", label: "Setup", hint: "Teach your home" },
  { href: "/preferences", label: "Preferences", hint: "Voice & alerts" },
  { href: "/evaluation", label: "Evaluation", hint: "Accuracy & logs" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-64 flex-shrink-0 flex-col bg-[#0F2E22] px-5 py-6 text-white">
      <div className="mb-10 flex items-center gap-2 px-2">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#1FBF82] font-display text-sm font-bold text-[#0F2E22]">
          M
        </span>
        <span className="font-display text-lg font-semibold tracking-tight">
          MemoryNav
        </span>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`group rounded-xl px-3 py-2.5 transition-colors ${
                active
                  ? "bg-white/10 text-white"
                  : "text-white/60 hover:bg-white/5 hover:text-white/90"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{item.label}</span>
                {active && (
                  <span className="h-1.5 w-1.5 rounded-full bg-[#1FBF82]" />
                )}
              </div>
              <span className="font-mono text-[11px] text-white/40">
                {item.hint}
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="rounded-xl bg-white/5 px-3 py-3">
        <p className="font-mono text-[11px] uppercase tracking-wide text-white/40">
          Status
        </p>
        <p className="mt-1 flex items-center gap-2 text-sm text-white/90">
          <span className="h-2 w-2 rounded-full bg-[#1FBF82]" />
          Running locally
        </p>
      </div>
    </aside>
  );
}
