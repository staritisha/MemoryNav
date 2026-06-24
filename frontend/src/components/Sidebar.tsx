// frontend/src/components/Sidebar.tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import PipelineStatus from "./PipelineStatus";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", hint: "Live view" },
  { href: "/setup", label: "Setup", hint: "Teach your home" },
  { href: "/preferences", label: "Preferences", hint: "Voice & alerts" },
  { href: "/evaluation", label: "Evaluation", hint: "Ablation & metrics" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-screen w-64 flex-shrink-0 flex-col border-r border-[#1A1E21] bg-[#0E1113] px-5 py-6">
      <div className="mb-8 flex items-center gap-2.5 px-1">
        <span className="flex h-7 w-7 items-center justify-center rounded border border-[#5EEAD4]/30 bg-[#5EEAD4]/10 font-mono text-xs font-bold text-[#5EEAD4]">
          M
        </span>
        <div>
          <p className="font-display text-[15px] font-medium leading-none text-[#E7ECEE]">
            MemoryNav
          </p>
          <p className="font-mono text-[10px] leading-none text-[#565E66]">
            v1.0 · edge AI
          </p>
        </div>
      </div>

      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`group rounded-md px-3 py-2.5 transition-colors ${
                active
                  ? "border border-[#262B2F] bg-[#181C1F] text-[#E7ECEE]"
                  : "border border-transparent text-[#6B7378] hover:bg-[#14181B] hover:text-[#C5CBCF]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{item.label}</span>
                {active && (
                  <span className="h-1.5 w-1.5 rounded-full bg-[#5EEAD4]" />
                )}
              </div>
              <span className="font-mono text-[10px] text-[#565E66]">
                {item.hint}
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto rounded-md border border-[#1A1E21] bg-[#0A0C0E] px-3.5 py-4">
        <p className="mb-3 font-mono text-[10px] uppercase tracking-wider text-[#565E66]">
          Pipeline
        </p>
        <PipelineStatus orientation="vertical" size="sm" />
      </div>
    </aside>
  );
}
