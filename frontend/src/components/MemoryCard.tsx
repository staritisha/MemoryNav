// frontend/src/components/MemoryCard.tsx
import type { Detection, HomeContext } from "@/lib/types";

interface MemoryCardProps {
  context: HomeContext | null;
  detections: Detection[];
}

export default function MemoryCard({ context, detections }: MemoryCardProps) {
  return (
    <div className="rounded-lg border border-[#262B2F] bg-[#14181B] p-5">
      <div className="flex items-center justify-between border-b border-[#262B2F] pb-3">
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-[#5EEAD4]" />
          <h3 className="font-display text-sm font-medium text-[#E7ECEE]">
            Long-term memory retrieval
          </h3>
        </div>
        {context?.score !== undefined && (
          <span className="font-mono text-[11px] text-[#8B95A1]">
            similarity {Math.round(context.score * 100)}%
          </span>
        )}
      </div>

      {context ? (
        <p className="mt-3 text-sm leading-relaxed text-[#C5CBCF]">
          &ldquo;{context.text}&rdquo;
        </p>
      ) : (
        <p className="mt-3 font-mono text-sm text-[#565E66]">
          No matching context for the current scene.
        </p>
      )}

      {detections.length > 0 && (
        <div className="mt-4 border-t border-[#1E2226] pt-3">
          <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-[#565E66]">
            Objects in view
          </p>
          <div className="flex flex-wrap gap-1.5">
            {detections.map((d) => (
              <span
                key={d.id}
                className="rounded border border-[#262B2F] bg-[#0A0C0E] px-2 py-1 font-mono text-[11px] text-[#C5CBCF]"
              >
                {d.label}
                {d.distanceMeters !== undefined && (
                  <span className="text-[#5EEAD4]"> {d.distanceMeters.toFixed(1)}m</span>
                )}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
