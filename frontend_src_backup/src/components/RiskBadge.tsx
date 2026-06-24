// frontend/src/components/RiskBadge.tsx
import type { RiskLevel } from "@/lib/types";

const RISK_STYLES: Record<
  RiskLevel,
  { bg: string; text: string; dot: string; label: string }
> = {
  LOW: {
    bg: "bg-[#E8F8EF]",
    text: "text-[#0F6B3F]",
    dot: "bg-[#1FBF82]",
    label: "Low risk",
  },
  MEDIUM: {
    bg: "bg-[#FDF1DE]",
    text: "text-[#92590A]",
    dot: "bg-[#D97706]",
    label: "Medium risk",
  },
  HIGH: {
    bg: "bg-[#FBE6E6]",
    text: "text-[#B91C1C]",
    dot: "bg-[#DC2626]",
    label: "High risk",
  },
};

interface RiskBadgeProps {
  level: RiskLevel;
  /** Show the pulsing dot for HIGH risk to draw the eye. */
  animate?: boolean;
  className?: string;
}

export default function RiskBadge({
  level,
  animate = true,
  className = "",
}: RiskBadgeProps) {
  const style = RISK_STYLES[level];

  return (
    <span
      role="status"
      aria-label={style.label}
      className={`inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 font-mono text-xs font-semibold uppercase tracking-wide ${style.bg} ${style.text} ${className}`}
    >
      <span className="relative flex h-2 w-2">
        {animate && level === "HIGH" && (
          <span
            className={`absolute inline-flex h-full w-full animate-ping rounded-full ${style.dot} opacity-60`}
          />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${style.dot}`} />
      </span>
      {level}
    </span>
  );
}
