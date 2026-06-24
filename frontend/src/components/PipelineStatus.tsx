// frontend/src/components/PipelineStatus.tsx
// The architecture, alive. Renders the 7 processing stages a frame actually
// passes through (Quality -> Detect -> Depth -> Risk -> Memory -> Alert ->
// Voice) as connected nodes. This is not a decorative diagram - feed it
// real per-frame status and it becomes the system's own telemetry.

import type { PipelineStage, PipelineStageId, PipelineStageStatus } from "@/lib/types";

const DEFAULT_LABELS: Record<PipelineStageId, string> = {
  quality: "Quality",
  detect: "Detect",
  depth: "Depth",
  risk: "Risk",
  memory: "Memory",
  alert: "Alert",
  voice: "Voice",
};

const STAGE_ORDER: PipelineStageId[] = [
  "quality",
  "detect",
  "depth",
  "risk",
  "memory",
  "alert",
  "voice",
];

const STATUS_COLOR: Record<PipelineStageStatus, string> = {
  idle: "#3A4146",
  active: "#5EEAD4",
  warn: "#FFB84D",
  off: "#262B2F",
};

interface PipelineStatusProps {
  /** Partial status overrides per stage; unlisted stages default to "idle". */
  statuses?: Partial<Record<PipelineStageId, PipelineStageStatus>>;
  orientation?: "horizontal" | "vertical";
  size?: "sm" | "md";
}

export default function PipelineStatus({
  statuses = {},
  orientation = "horizontal",
  size = "md",
}: PipelineStatusProps) {
  const stages: PipelineStage[] = STAGE_ORDER.map((id) => ({
    id,
    label: DEFAULT_LABELS[id],
    status: statuses[id] ?? "idle",
  }));

  const nodeSize = size === "sm" ? 7 : 9;
  const gap = size === "sm" ? "gap-1" : "gap-2";

  if (orientation === "vertical") {
    return (
      <div className="flex flex-col gap-3" aria-label="Pipeline status">
        {stages.map((stage, i) => (
          <div key={stage.id} className="flex items-center gap-2.5">
            <div className="relative flex flex-col items-center">
              <span
                className="relative flex items-center justify-center rounded-full"
                style={{ width: nodeSize, height: nodeSize }}
              >
                {stage.status === "active" && (
                  <span
                    className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
                    style={{ backgroundColor: STATUS_COLOR.active }}
                  />
                )}
                <span
                  className="relative inline-flex rounded-full"
                  style={{
                    width: nodeSize,
                    height: nodeSize,
                    backgroundColor: STATUS_COLOR[stage.status],
                  }}
                />
              </span>
              {i < stages.length - 1 && (
                <span
                  className="mt-1 h-3 w-px"
                  style={{ backgroundColor: "#262B2F" }}
                />
              )}
            </div>
            <span className="font-mono text-[11px] uppercase tracking-wide text-[#8B95A1]">
              {stage.label}
            </span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div
      className={`flex items-center ${gap}`}
      role="img"
      aria-label={`Pipeline status: ${stages
        .map((s) => `${s.label} ${s.status}`)
        .join(", ")}`}
    >
      {stages.map((stage, i) => (
        <div key={stage.id} className="flex items-center">
          <div className="flex flex-col items-center gap-1.5">
            <span
              className="relative flex items-center justify-center"
              style={{ width: nodeSize, height: nodeSize }}
            >
              {stage.status === "active" && (
                <span
                  className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
                  style={{ backgroundColor: STATUS_COLOR.active }}
                />
              )}
              <span
                className="relative inline-flex rounded-full"
                style={{
                  width: nodeSize,
                  height: nodeSize,
                  backgroundColor: STATUS_COLOR[stage.status],
                }}
              />
            </span>
            <span className="font-mono text-[10px] uppercase tracking-wide text-[#8B95A1]">
              {stage.label}
            </span>
          </div>
          {i < stages.length - 1 && (
            <span
              className="mx-1.5 mb-4 h-px w-5"
              style={{ backgroundColor: "#262B2F" }}
            />
          )}
        </div>
      ))}
    </div>
  );
}
