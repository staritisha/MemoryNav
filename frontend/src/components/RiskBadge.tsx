// frontend/src/components/RiskBadge.tsx
import type { RiskLevel } from "@/lib/types";

const RISK_STYLES: Record<
  RiskLevel,
  { color: string; bg: string; label: string }
> = {
  LOW: { color: "#4ADE80", bg: "rgba(74,222,128,0.12)", label: "Low risk" },
  MEDIUM: { color: "#FFB84D", bg: "rgba(255,184,77,0.12)", label: "Medium risk" },
  HIGH: { color: "#FF5C5C", bg: "rgba(255,92,92,0.14)", label: "High risk" },
};

interface RiskBadgeProps {
  level: RiskLevel;
  /** Raw 0..1 score from the risk formula, shown as a mono readout if given. */
  score?: number;
  animate?: boolean;
  className?: string;
}

export default function RiskBadge({
  level,
  score,
  animate = true,
  className = "",
}: RiskBadgeProps) {
  const style = RISK_STYLES[level];

  return (
    <span
      role="status"
      aria-label={style.label}
      className={`inline-flex items-center gap-2.5 rounded-md border px-3 py-1.5 font-mono text-xs font-medium uppercase tracking-wider ${className}`}
      style={{ backgroundColor: style.bg, borderColor: style.color, color: style.color }}
    >
      <span className="relative flex h-2 w-2">
        {animate && level === "HIGH" && (
          <span
            className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
            style={{ backgroundColor: style.color }}
          />
        )}
        <span
          className="relative inline-flex h-2 w-2 rounded-full"
          style={{ backgroundColor: style.color }}
        />
      </span>
      {level}
      {score !== undefined && (
        <span className="text-[#8B95A1]">· {score.toFixed(2)}</span>
      )}
    </span>
  );
}
