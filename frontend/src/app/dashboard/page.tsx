// frontend/src/app/dashboard/page.tsx
"use client";

import { useEffect, useState } from "react";
import { useDetection } from "@/lib/ws";
import { getMemory, getPrefs } from "@/lib/api";
import LiveFeed from "@/components/LiveFeed";
import RiskBadge from "@/components/RiskBadge";
import MemoryCard from "@/components/MemoryCard";
import AudioToggle from "@/components/AudioToggle";
import PipelineStatus from "@/components/PipelineStatus";
import type {
  HomeContext,
  PipelineStageId,
  PipelineStageStatus,
  UserPreferences,
} from "@/lib/types";

interface LogEntry {
  id: string;
  time: string;
  message: string;
  tone: "info" | "warn" | "suppressed" | "ghost";
}

export default function DashboardPage() {
  const { frame, status, sendFrame } = useDetection();
  const [memory, setMemory]         = useState<HomeContext[]>([]);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [log, setLog]               = useState<LogEntry[]>([]);

  useEffect(() => {
    getMemory().then(setMemory).catch(() => setMemory([]));
    getPrefs()
      .then((p: UserPreferences) => setAudioEnabled(p.audioEnabled ?? true))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!frame) return;

    let message: string;
    let tone: LogEntry["tone"];

    if (frame.spokeGhost && frame.spoke) {
      message = frame.spoke;
      tone = "ghost";
    } else if (frame.suppressed) {
      message = frame.suppressionReason ?? "Alert suppressed, within suppression window";
      tone = "suppressed";
    } else if (frame.riskLevel === "HIGH" && frame.riskReason) {
      message = frame.riskReason;
      tone = "warn";
    } else {
      message =
        frame.riskReason ??
        `${frame.riskLevel} risk, ${frame.detections.length} object(s) in view`;
      tone = "info";
    }

    const time = new Date(
      typeof frame.timestamp === "number"
        ? frame.timestamp * 1000
        : frame.timestamp
    ).toLocaleTimeString();

    setLog((prev) => {
      const entry: LogEntry = { id: `${frame.timestamp}-${prev.length}`, time, message, tone };
      return [entry, ...prev].slice(0, 10);
    });
  }, [frame]);

  const riskLevel  = frame?.riskLevel  ?? "LOW";
  const detections = frame?.detections ?? [];
  const topMemory  = memory[0] ?? null;

  // Derive pipeline stage statuses from the latest frame
  const pipelineStatuses: Partial<Record<PipelineStageId, PipelineStageStatus>> = frame
    ? {
        quality: "active",
        detect:  detections.length > 0 ? "active" : "idle",
        depth:   detections.some((d) => d.distanceMeters !== undefined) ? "active" : "idle",
        risk:    riskLevel === "HIGH" ? "warn" : "active",
        memory:  (frame.memoryContext || frame.spokeGhost) ? "active" : "idle",
        alert:   frame.suppressed ? "warn" : "active",
        voice:   audioEnabled && frame.spoke ? "active" : "idle",
      }
    : {};

  return (
    <div className="mx-auto max-w-6xl">
      <header className="mb-6 flex items-start justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-wider text-[#565E66]">
            Live dashboard
          </p>
          <h1 className="mt-1 font-display text-2xl font-semibold text-[#E7ECEE]">
            What MemoryNav sees right now
          </h1>
        </div>
        <AudioToggle enabled={audioEnabled} onChange={setAudioEnabled} />
      </header>

      {/* Status bar */}
      <div className="mb-5 flex items-center justify-between rounded-lg border border-[#262B2F] bg-[#0E1113] px-5 py-3">
        <div className="flex items-center gap-3">
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: status === "open" ? "#4ADE80" : "#FFB84D" }}
          />
          <span className="font-mono text-xs text-[#8B95A1]">
            {status === "open" ? "stream connected" : status}
          </span>
          {frame?.latencyMs !== undefined && (
            <span className="font-mono text-xs text-[#565E66]">
              {frame.latencyMs.toFixed(0)}ms
            </span>
          )}
        </div>
        <PipelineStatus statuses={pipelineStatuses} size="sm" />
        <RiskBadge level={riskLevel} score={frame?.riskScore} />
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.6fr_1fr]">
        <div>
          <LiveFeed
            detections={detections}
            riskLevel={riskLevel}
            onFrame={sendFrame}
          />

          {frame?.riskReason && (
            <div
              className="mt-3 rounded-lg border px-4 py-3 font-mono text-sm"
              style={{
                borderColor: riskLevel === "HIGH" ? "#FF5C5C" : "#262B2F",
                backgroundColor:
                  riskLevel === "HIGH" ? "rgba(255,92,92,0.08)" : "#0E1113",
                color: riskLevel === "HIGH" ? "#FF5C5C" : "#8B95A1",
              }}
            >
              {frame.riskReason}
            </div>
          )}

          {/* Memory context banner - shows when ChromaDB boost fired */}
          {frame?.memoryContext && (
            <div className="mt-3 rounded-lg border border-[#818CF8]/30 bg-[#818CF8]/08 px-4 py-3">
              <p className="font-mono text-xs text-[#818CF8]">◈ memory context</p>
              <p className="mt-1 text-sm text-[#C5CBCF]">{frame.memoryContext}</p>
            </div>
          )}

          <div className="mt-5">
            <MemoryCard context={topMemory} detections={detections} />
          </div>
        </div>

        {/* Alert log */}
        <div className="rounded-lg border border-[#262B2F] bg-[#0E1113]">
          <div className="flex items-center justify-between border-b border-[#262B2F] px-4 py-3">
            <h3 className="font-display text-sm font-medium text-[#E7ECEE]">
              Alert manager
            </h3>
            <span className="font-mono text-[10px] uppercase tracking-wider text-[#565E66]">
              WalkVLM suppression · 4s
            </span>
          </div>

          {/* Suppression stats */}
          {frame?.suppressionStats && (
            <div className="flex gap-4 border-b border-[#1A1E21] px-4 py-2">
              {Object.entries(frame.suppressionStats).map(([k, v]) => (
                <div key={k} className="text-center">
                  <p className="font-display text-lg font-semibold text-[#5EEAD4]">
                    {String(v)}
                  </p>
                  <p className="font-mono text-[9px] uppercase text-[#565E66]">{k}</p>
                </div>
              ))}
            </div>
          )}

          <div className="max-h-[380px] overflow-y-auto px-4 py-3">
            {log.length === 0 ? (
              <p className="font-mono text-xs text-[#565E66]">
                Waiting for the first processed frame…
              </p>
            ) : (
              <ul className="space-y-2.5">
                {log.map((entry) => (
                  <li key={entry.id} className="flex gap-2.5 text-xs">
                    <span className="flex-shrink-0 font-mono text-[#565E66]">
                      {entry.time}
                    </span>
                    <span
                      className="font-mono"
                      style={{
                        color:
                          entry.tone === "warn"
                            ? "#FF5C5C"
                            : entry.tone === "suppressed"
                            ? "#FFB84D"
                            : entry.tone === "ghost"
                            ? "#818CF8"
                            : "#8B95A1",
                      }}
                    >
                      {entry.tone === "suppressed"
                        ? "⦸ "
                        : entry.tone === "warn"
                        ? "▲ "
                        : entry.tone === "ghost"
                        ? "◈ "
                        : "· "}
                      {entry.message}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
