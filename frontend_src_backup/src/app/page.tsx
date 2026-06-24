// frontend/src/app/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useDetection } from "@/lib/ws";
import { getMemory, getPrefs } from "@/lib/api";
import LiveFeed from "@/components/LiveFeed";
import RiskBadge from "@/components/RiskBadge";
import MemoryCard from "@/components/MemoryCard";
import AudioToggle from "@/components/AudioToggle";
import type { HomeContext } from "@/lib/types";

export default function DashboardPage() {
  const { frame, status, sendFrame } = useDetection();
  const [memory, setMemory] = useState<HomeContext[]>([]);
  const [audioEnabled, setAudioEnabled] = useState(true);

  useEffect(() => {
    getMemory().then(setMemory).catch(() => setMemory([]));
    getPrefs()
      .then((p) => setAudioEnabled(p.audioEnabled))
      .catch(() => {});
  }, []);

  const riskLevel = frame?.riskLevel ?? "LOW";
  const detections = frame?.detections ?? [];
  const topMemory = memory[0] ?? null;

  return (
    <div className="mx-auto max-w-5xl">
      <header className="mb-6 flex items-start justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-wide text-[#6B7280]">
            Dashboard
          </p>
          <h1 className="mt-1 font-display text-2xl font-semibold text-[#0F2E22]">
            What MemoryNav sees right now
          </h1>
        </div>
        <AudioToggle enabled={audioEnabled} onChange={setAudioEnabled} />
      </header>

      <div className="mb-4 flex items-center justify-between rounded-xl border border-[#E5E7EB] bg-white px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-[#6B7280]">
            Connection
          </span>
          <span
            className={`font-mono text-xs font-medium ${
              status === "open" ? "text-[#0F6B3F]" : "text-[#92590A]"
            }`}
          >
            {status === "open" ? "connected" : status}
          </span>
        </div>
        <RiskBadge level={riskLevel} />
      </div>

      <LiveFeed
        detections={detections}
        riskLevel={riskLevel}
        onFrame={sendFrame}
      />

      {frame?.riskReason && (
        <p className="mt-3 rounded-xl bg-[#FDF1DE] px-4 py-3 text-sm text-[#92590A]">
          {frame.riskReason}
        </p>
      )}

      <div className="mt-6">
        <MemoryCard context={topMemory} detections={detections} />
      </div>
    </div>
  );
}
