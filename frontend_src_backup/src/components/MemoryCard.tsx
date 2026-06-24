// frontend/src/components/MemoryCard.tsx
import type { Detection, HomeContext } from "@/lib/types";

interface MemoryCardProps {
  context: HomeContext | null;
  detections: Detection[];
}

export default function MemoryCard({ context, detections }: MemoryCardProps) {
  return (
    <div className="rounded-2xl border border-[#E5E7EB] bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="font-display text-sm font-semibold text-[#0F2E22]">
          Home memory
        </h3>
        {context?.score !== undefined && (
          <span className="font-mono text-[11px] text-[#6B7280]">
            match {Math.round(context.score * 100)}%
          </span>
        )}
      </div>

      {context ? (
        <p className="mt-3 text-sm leading-relaxed text-[#374151]">
          {context.text}
        </p>
      ) : (
        <p className="mt-3 text-sm text-[#9CA3AF]">
          No relevant memory found for what&apos;s in view right now.
        </p>
      )}

      {detections.length > 0 && (
        <div className="mt-4 border-t border-[#F1F2F0] pt-3">
          <p className="mb-2 font-mono text-[11px] uppercase tracking-wide text-[#9CA3AF]">
            In view
          </p>
          <div className="flex flex-wrap gap-1.5">
            {detections.map((d) => (
              <span
                key={d.id}
                className="rounded-full bg-[#F3F4F1] px-2.5 py-1 font-mono text-[11px] text-[#374151]"
              >
                {d.label}
                {d.distanceMeters !== undefined && (
                  <span className="text-[#9CA3AF]"> · {d.distanceMeters.toFixed(1)}m</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
