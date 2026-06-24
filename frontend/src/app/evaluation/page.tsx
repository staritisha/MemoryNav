// frontend/src/app/evaluation/page.tsx
//
// Ablation study + per-component metrics.
// All values are real measured numbers from evaluation/results.json
// (generated 2026-06-24, 4 indoor videos, every 5th frame, all components REAL).

import type { AblationRow, ComponentMetric } from "@/lib/types";

// ── Real measured values from evaluation/results.json ───────────────────────

const ABLATION: AblationRow[] = [
  {
    configuration: "A - YOLO only",
    description: "Detection only - no depth, no memory, no alert suppression",
    successRate: 0.167,   // 1/6 events warned
  },
  {
    configuration: "B - Detection + Depth + Risk",
    description: "Distance-aware risk scoring, no memory or suppression",
    successRate: 0.667,   // 4/6 events warned
  },
  {
    configuration: "Full system",
    description: "Detection + depth + risk + ChromaDB memory + alert suppression",
    successRate: 0.667,   // 4/6 events warned - suppression reduces false alerts, not misses
  },
];

// False alerts per config (from aggregate.false_alerts_total):
// A: 46  B: 314  Full: 16
// Suppression reduced false alerts by 94.9% (314 → 16)

const COMPONENT_METRICS: ComponentMetric[] = [
  {
    component: "Perception",
    metric: "Detection latency",
    description: "Per-frame YOLO inference time - CPU, no GPU acceleration",
    value: 62,
    unit: "ms",
  },
  {
    component: "Depth",
    metric: "Pipeline latency",
    description: "Per-frame YOLO + Depth-Anything (CPU) - MPS/GPU needed for real-time",
    value: 2127,
    unit: "ms",
  },
  {
    component: "Alert system",
    metric: "False alerts (full system)",
    description: "Total false alerts across 4 test videos with suppression active",
    value: 16,
    unit: "",
  },
  {
    component: "Alert system",
    metric: "False alerts (baseline B)",
    description: "False alerts without suppression - shows 94.9% reduction",
    value: 314,
    unit: "",
  },
  {
    component: "Alert suppression",
    metric: "Redundancy reduction",
    description: "False alerts eliminated by suppression: Baseline B → Full System",
    value: 94.9,
    unit: "%",
  },
  {
    component: "Navigation",
    metric: "Success rate (full system)",
    description: "Events where user was warned before reaching the obstacle",
    value: 66.7,
    unit: "%",
  },
  {
    component: "Navigation",
    metric: "Miss rate",
    description: "Obstacles not warned - 2/6 events (kitchen video, label mismatch)",
    value: 33.3,
    unit: "%",
  },
  {
    component: "Memory",
    metric: "Module status",
    description: "ChromaDB + sentence-transformers loaded from real persistent store",
    value: null,
    unit: "REAL ✓",
  },
];

const LIMITATIONS = [
  "Depth model (Depth-Anything) runs at ~2,127ms/frame on CPU - real-time use requires MPS or CUDA. YOLO alone runs at 62ms/frame on the same hardware.",
  "Kitchen video scored 0% success - YOLO did not detect 'refrigerator' or 'dining table' at the annotated timestamps. Annotation labels must match COCO class names that actually appear.",
  "Alert suppression maintains recall (66.7%) but does not improve it - memory context currently has no measurable impact on which obstacles get detected.",
  "Depth estimates are relative, not metric-calibrated. The relative→metric scale conversion is a heuristic (see run_ablation.py depth wrapper).",
  "Detection accuracy degrades below FRAME_MIN_BLUR_VARIANCE = 3.0 or FRAME_MIN_BRIGHTNESS = 20.0 - tuned for stock footage, may need adjustment for dim rooms.",
  "Not a medical device. MemoryNav augments user judgment - it does not replace mobility aids, white canes, or professional guidance.",
];

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatValue(value: number | null, unit: string): string {
  if (value === null) return unit; // used for "REAL ✓" above
  return unit ? `${value}${unit}` : `${value}`;
}

function SuccessBar({ rate }: { rate: number }) {
  const pct = Math.round(rate * 100);
  const color = pct >= 60 ? "#5EEAD4" : pct >= 30 ? "#FFB84D" : "#FF5C5C";
  return (
    <div className="flex items-center gap-3">
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-[#1A1E21]">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="font-mono text-sm font-medium" style={{ color }}>
        {pct}%
      </span>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function EvaluationPage() {
  return (
    <div className="mx-auto max-w-5xl">
      <header className="mb-8">
        <p className="font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Evaluation
        </p>
        <h1 className="mt-1 font-display text-2xl font-semibold text-[#E7ECEE]">
          Ablation study &amp; component metrics
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-[#8B95A1]">
          All values measured on 4 indoor walking clips (bedroom, kitchen, hallway,
          living room), every 5th frame. All four pipeline components confirmed{" "}
          <span className="font-mono text-[#5EEAD4]">REAL</span> - no stubs.
          Generated 2026-06-24.
        </p>
      </header>

      {/* ── Summary stat row ─────────────────────────────────────────────── */}
      <div className="mb-10 grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[
          { v: "66.7%", l: "Full system success rate",  s: "4/6 events warned" },
          { v: "94.9%", l: "False-alert reduction",      s: "314 → 16 (B → Full)" },
          { v: "62ms",  l: "YOLO latency",               s: "per frame · CPU" },
          { v: "414",   l: "Frames processed",           s: "all components REAL" },
        ].map((card) => (
          <div key={card.l} className="rounded-xl border border-[#262B2F] bg-[#0E1113] px-5 py-4">
            <p className="font-display text-2xl font-semibold text-[#5EEAD4]">{card.v}</p>
            <p className="mt-1 text-sm font-medium text-[#E7ECEE]">{card.l}</p>
            <p className="font-mono text-[11px] text-[#565E66]">{card.s}</p>
          </div>
        ))}
      </div>

      {/* ── Ablation study ───────────────────────────────────────────────── */}
      <section className="mb-10">
        <h2 className="mb-3 font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Ablation study - navigation success rate
        </h2>
        <div className="overflow-hidden rounded-xl border border-[#262B2F]">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[#262B2F] bg-[#0E1113]">
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Configuration</th>
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Description</th>
                <th className="px-4 py-3 text-right font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Success rate</th>
              </tr>
            </thead>
            <tbody>
              {ABLATION.map((row, i) => (
                <tr key={row.configuration} className={i < ABLATION.length - 1 ? "border-b border-[#1A1E21]" : ""}>
                  <td className="px-4 py-3.5 font-medium text-[#E7ECEE]">{row.configuration}</td>
                  <td className="px-4 py-3.5 text-[#8B95A1]">{row.description}</td>
                  <td className="px-4 py-3.5">
                    <div className="flex justify-end">
                      <SuccessBar rate={row.successRate ?? 0} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 font-mono text-[11px] text-[#565E66]">
          Key finding: depth+risk scoring lifts success rate 16.7% → 66.7% (+50pp).
          Memory+suppression maintains that recall while cutting false alerts by 94.9% (314 → 16).
        </p>
      </section>

      {/* ── Per-component metrics ─────────────────────────────────────────── */}
      <section className="mb-10">
        <h2 className="mb-3 font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Per-component metrics
        </h2>
        <div className="overflow-hidden rounded-xl border border-[#262B2F]">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[#262B2F] bg-[#0E1113]">
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Component</th>
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Metric</th>
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Description</th>
                <th className="px-4 py-3 text-right font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Value</th>
              </tr>
            </thead>
            <tbody>
              {COMPONENT_METRICS.map((row, i) => (
                <tr
                  key={`${row.component}-${row.metric}`}
                  className={i < COMPONENT_METRICS.length - 1 ? "border-b border-[#1A1E21]" : ""}
                >
                  <td className="px-4 py-3.5 text-[#C5CBCF]">{row.component}</td>
                  <td className="px-4 py-3.5 font-mono text-[12px] text-[#E7ECEE]">{row.metric}</td>
                  <td className="px-4 py-3.5 text-[#8B95A1]">{row.description}</td>
                  <td className="px-4 py-3.5 text-right font-mono">
                    <span className={row.value === null ? "text-[#5EEAD4]" : "text-[#5EEAD4]"}>
                      {formatValue(row.value, row.unit)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── Methodology note ─────────────────────────────────────────────── */}
      <section className="mb-10 rounded-xl border border-[#262B2F] bg-[#0E1113] p-6">
        <h2 className="mb-3 font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Methodology
        </h2>
        <ul className="space-y-2 text-sm text-[#8B95A1]">
          {[
            "4 stock-footage indoor walking clips (Pexels/Pixabay CC0): bedroom, kitchen, hallway, living room.",
            "Ground-truth annotation: per-video .json sidecars with obstacle label and critical_time_s (last moment a warning would be useful).",
            "A configuration 'succeeds' on an event if it raises a HIGH-risk alert at or before critical_time_s.",
            "Hallway video has empty events [] - contributes only to false-alert measurement.",
            "Every 5th frame sampled (--sample-every 5) to keep Depth-Anything tractable on CPU.",
            "All 4 pipeline components confirmed REAL via component banner - no stub fallbacks used.",
          ].map((point) => (
            <li key={point} className="flex gap-3">
              <span className="mt-1.5 h-1 w-1 flex-shrink-0 rounded-full bg-[#565E66]" />
              {point}
            </li>
          ))}
        </ul>
      </section>

      {/* ── Limitations ──────────────────────────────────────────────────── */}
      <section>
        <h2 className="mb-3 font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Known limitations
        </h2>
        <ul className="space-y-2.5 rounded-xl border border-[#262B2F] bg-[#14181B] p-5">
          {LIMITATIONS.map((point) => (
            <li key={point} className="flex gap-3 text-sm text-[#C5CBCF]">
              <span className="mt-1.5 h-1 w-1 flex-shrink-0 rounded-full bg-[#565E66]" />
              {point}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
