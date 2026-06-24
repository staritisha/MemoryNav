// frontend/src/app/evaluation/page.tsx
//
// Ablation study + per-component metrics, matching the evaluation
// methodology in the design doc (Section 6-7). Values default to null and
// render as "pending measurement" rather than fabricated numbers — drop in
// real results from your test runs as you collect them. Never hardcode a
// plausible-looking number here; an interviewer asking "how did you measure
// this" is the whole point of this page.

import type { AblationRow, ComponentMetric } from "@/lib/types";

// --- Replace these with real measured values as you collect them. ---

const ABLATION: AblationRow[] = [
  {
    configuration: "A — Detection only",
    description: "YOLOv8 + risk engine, no memory retrieval, no suppression",
    successRate: null,
  },
  {
    configuration: "B — Detection + memory",
    description: "Adds ChromaDB home-context retrieval, no alert suppression",
    successRate: null,
  },
  {
    configuration: "Full system",
    description: "Detection + depth + memory + alert manager + voice",
    successRate: null,
  },
];

const COMPONENT_METRICS: ComponentMetric[] = [
  {
    component: "Perception",
    metric: "mAP@0.5",
    description: "YOLOv8-nano detection accuracy on held-out test frames",
    value: null,
    unit: "",
  },
  {
    component: "Depth",
    metric: "MAE",
    description: "Mean absolute error vs. ground-truth distance",
    value: null,
    unit: "m",
  },
  {
    component: "Risk engine",
    metric: "Precision",
    description: "Correct HIGH-risk calls / total HIGH-risk calls",
    value: null,
    unit: "",
  },
  {
    component: "Risk engine",
    metric: "Recall",
    description: "Correct HIGH-risk calls / actual hazards present",
    value: null,
    unit: "",
  },
  {
    component: "Memory",
    metric: "Retrieval relevance",
    description: "Human-rated relevance of top-1 retrieved context",
    value: null,
    unit: "",
  },
  {
    component: "Alert manager",
    metric: "Suppression rate",
    description: "Fraction of repeat alerts correctly suppressed",
    value: null,
    unit: "",
  },
  {
    component: "End-to-end",
    metric: "Latency",
    description: "Frame capture to spoken alert, on target hardware",
    value: null,
    unit: "ms",
  },
  {
    component: "End-to-end",
    metric: "Throughput",
    description: "Sustained frames processed per second",
    value: null,
    unit: "fps",
  },
];

const LIMITATIONS = [
  "Detection accuracy degrades in low-light conditions below the frame-quality gate threshold.",
  "Depth estimates are relative, not metric-calibrated, without a known reference object in frame.",
  "Memory retrieval quality depends entirely on how thoroughly the home was described during setup.",
  "Voice latency on CPU-only hardware may exceed comfortable real-time use; MPS/GPU recommended.",
];

function formatValue(value: number | null, unit: string): string {
  if (value === null) return "pending measurement";
  return unit ? `${value}${unit}` : `${value}`;
}

export default function EvaluationPage() {
  return (
    <div className="mx-auto max-w-5xl">
      <header className="mb-8">
        <p className="font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Evaluation
        </p>
        <h1 className="mt-1 font-display text-2xl font-semibold text-[#E7ECEE]">
          Ablation study & component metrics
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-[#8B95A1]">
          Measured results go here as they&apos;re collected. Fields show{" "}
          <span className="font-mono text-[#565E66]">pending measurement</span>{" "}
          rather than placeholder numbers — this page is built to be filled
          in, not faked.
        </p>
      </header>

      {/* Ablation study */}
      <section className="mb-10">
        <h2 className="mb-3 font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Ablation study — navigation success rate
        </h2>
        <div className="overflow-hidden rounded-lg border border-[#262B2F]">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[#262B2F] bg-[#0E1113]">
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">
                  Configuration
                </th>
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">
                  Description
                </th>
                <th className="px-4 py-3 text-right font-mono text-[11px] uppercase tracking-wider text-[#565E66]">
                  Success rate
                </th>
              </tr>
            </thead>
            <tbody>
              {ABLATION.map((row, i) => (
                <tr
                  key={row.configuration}
                  className={i < ABLATION.length - 1 ? "border-b border-[#1A1E21]" : ""}
                >
                  <td className="px-4 py-3.5 font-medium text-[#E7ECEE]">
                    {row.configuration}
                  </td>
                  <td className="px-4 py-3.5 text-[#8B95A1]">{row.description}</td>
                  <td className="px-4 py-3.5 text-right font-mono">
                    {row.successRate === null ? (
                      <span className="text-[#565E66]">pending measurement</span>
                    ) : (
                      <span className="text-[#5EEAD4]">
                        {Math.round(row.successRate * 100)}%
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Per-component metrics */}
      <section className="mb-10">
        <h2 className="mb-3 font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Per-component metrics
        </h2>
        <div className="overflow-hidden rounded-lg border border-[#262B2F]">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-[#262B2F] bg-[#0E1113]">
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">
                  Component
                </th>
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">
                  Metric
                </th>
                <th className="px-4 py-3 font-mono text-[11px] uppercase tracking-wider text-[#565E66]">
                  Description
                </th>
                <th className="px-4 py-3 text-right font-mono text-[11px] uppercase tracking-wider text-[#565E66]">
                  Value
                </th>
              </tr>
            </thead>
            <tbody>
              {COMPONENT_METRICS.map((row, i) => (
                <tr
                  key={`${row.component}-${row.metric}`}
                  className={i < COMPONENT_METRICS.length - 1 ? "border-b border-[#1A1E21]" : ""}
                >
                  <td className="px-4 py-3.5 text-[#C5CBCF]">{row.component}</td>
                  <td className="px-4 py-3.5 font-mono text-[12px] text-[#E7ECEE]">
                    {row.metric}
                  </td>
                  <td className="px-4 py-3.5 text-[#8B95A1]">{row.description}</td>
                  <td className="px-4 py-3.5 text-right font-mono">
                    {row.value === null ? (
                      <span className="text-[#565E66]">pending measurement</span>
                    ) : (
                      <span className="text-[#5EEAD4]">
                        {formatValue(row.value, row.unit)}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Limitations — owning constraints is part of the credibility signal */}
      <section>
        <h2 className="mb-3 font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Known limitations
        </h2>
        <ul className="space-y-2.5 rounded-lg border border-[#262B2F] bg-[#14181B] p-5">
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
