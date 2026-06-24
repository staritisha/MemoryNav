// frontend/src/app/page.tsx
// Public landing page — no backend connection required.
// Shows the project pitch, architecture, real ablation numbers, and CTAs.

import Link from "next/link";

// ── Real measured numbers from evaluation/results.json ──────────────────────
const METRICS = {
  baselineA:   { rate: 16.7, label: "YOLO only" },
  baselineB:   { rate: 66.7, label: "+ Depth + Risk" },
  fullSystem:  { rate: 66.7, label: "Full system" },
  falseAlertReduction: 94.9,   // % reduction B → Full
  yoloLatencyMs: 62,           // ms/frame, CPU
  depthLatencyMs: 2127,        // ms/frame, CPU (Depth-Anything bottleneck)
  framesProcessed: 414,
};

const PIPELINE_STEPS = [
  { id: "01", label: "Frame Quality Gate",   sub: "blur · brightness · motion",   color: "#5EEAD4" },
  { id: "02", label: "YOLOv8-nano",          sub: "obstacle detection · 62 ms",    color: "#5EEAD4" },
  { id: "03", label: "Depth-Anything",       sub: "monocular distance · no LiDAR", color: "#5EEAD4" },
  { id: "04", label: "Risk Engine",          sub: "distance × motion × context",   color: "#FFB84D" },
  { id: "05", label: "Long-Term Memory",     sub: "ChromaDB · sentence-transformers", color: "#5EEAD4" },
  { id: "06", label: "Alert Manager",        sub: "4s suppression window (WalkVLM)", color: "#5EEAD4" },
  { id: "07", label: "Voice Output",         sub: "pyttsx3 · fully offline",       color: "#4ADE80" },
];

const RESEARCH_REFS = [
  { short: "WalkVLM", year: "2024", note: "Temporal alert suppression — MemoryNav's Alert Manager implements this directly.", href: "https://arxiv.org/abs/2412.20903" },
  { short: "VISA",    year: "2025", note: "Holistic multi-layer indoor assistance — the structural blueprint for MemoryNav's pipeline.", href: "https://www.mdpi.com/2313-433X/11/1/9" },
  { short: "NavSpace",year: "2026", note: "Personalized spatial memory as an open frontier — frames MemoryNav's future direction.", href: "https://arxiv.org/abs/2510.08173" },
];

const TECH_BADGES = [
  "YOLOv8-nano", "Depth-Anything", "ChromaDB", "sentence-transformers",
  "Whisper", "FastAPI", "Next.js", "pyttsx3", "OpenCV", "PyTorch MPS",
];

// ── Sub-components ────────────────────────────────────────────────────────────

function StatCard({ value, unit, label, sub }: { value: string; unit?: string; label: string; sub: string }) {
  return (
    <div className="rounded-xl border border-[#262B2F] bg-[#0E1113] px-6 py-5">
      <div className="flex items-baseline gap-1">
        <span className="font-display text-3xl font-semibold text-[#5EEAD4]">{value}</span>
        {unit && <span className="font-mono text-sm text-[#5EEAD4]">{unit}</span>}
      </div>
      <p className="mt-1 text-sm font-medium text-[#E7ECEE]">{label}</p>
      <p className="mt-0.5 font-mono text-[11px] text-[#565E66]">{sub}</p>
    </div>
  );
}

function AblationBar({ rate, label, isHighlighted }: { rate: number; label: string; isHighlighted?: boolean }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-[#8B95A1]">{label}</span>
        <span className={`font-mono text-sm font-medium ${isHighlighted ? "text-[#5EEAD4]" : "text-[#8B95A1]"}`}>
          {rate}%
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-[#1A1E21]">
        <div
          className="h-full rounded-full transition-all"
          style={{
            width: `${rate}%`,
            backgroundColor: isHighlighted ? "#5EEAD4" : "#3A4147",
          }}
        />
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function LandingPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-20 py-4">

      {/* ── HERO ─────────────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden rounded-2xl border border-[#262B2F] bg-[#0E1113] px-10 py-14">
        {/* Decorative glow */}
        <div
          aria-hidden
          className="pointer-events-none absolute -right-20 -top-20 h-72 w-72 rounded-full opacity-10"
          style={{ background: "radial-gradient(circle, #5EEAD4, transparent 70%)" }}
        />
        <div className="relative">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded border border-[#5EEAD4]/30 bg-[#5EEAD4]/10 font-mono text-[11px] font-bold text-[#5EEAD4]">
              M
            </span>
            <span className="font-mono text-xs uppercase tracking-widest text-[#565E66]">
              MemoryNav · v1.0 · edge AI
            </span>
          </div>

          <h1 className="font-display text-4xl font-semibold leading-tight text-[#E7ECEE] sm:text-5xl">
            Indoor navigation assistance<br />
            <span className="text-[#5EEAD4]">that remembers your home.</span>
          </h1>

          <p className="mt-5 max-w-2xl text-[#8B95A1]">
            Real-time obstacle detection, monocular depth estimation, and personalized
            spatial memory — running fully offline on consumer hardware.
            Built for elderly and visually impaired users.
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            {TECH_BADGES.map((b) => (
              <span
                key={b}
                className="rounded-md border border-[#262B2F] bg-[#14181B] px-2.5 py-1 font-mono text-[11px] text-[#8B95A1]"
              >
                {b}
              </span>
            ))}
          </div>

          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              href="/dashboard"
              className="rounded-lg border border-[#5EEAD4]/50 bg-[#5EEAD4]/10 px-6 py-2.5 text-sm font-medium text-[#5EEAD4] transition hover:bg-[#5EEAD4]/20"
            >
              Open Dashboard →
            </Link>
            <Link
              href="/evaluation"
              className="rounded-lg border border-[#262B2F] bg-[#14181B] px-6 py-2.5 text-sm font-medium text-[#8B95A1] transition hover:text-[#E7ECEE]"
            >
              View ablation results
            </Link>
            <a
              href="https://github.com/staritisha/MemoryNav"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-lg border border-[#262B2F] bg-[#14181B] px-6 py-2.5 text-sm font-medium text-[#8B95A1] transition hover:text-[#E7ECEE]"
            >
              GitHub ↗
            </a>
          </div>
        </div>
      </section>

      {/* ── THE PROBLEM ──────────────────────────────────────────────────── */}
      <section>
        <p className="mb-2 font-mono text-xs uppercase tracking-widest text-[#565E66]">The problem</p>
        <h2 className="mb-6 font-display text-2xl font-semibold text-[#E7ECEE]">
          Every existing solution has the same gap
        </h2>
        <div className="overflow-hidden rounded-xl border border-[#262B2F]">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#262B2F] bg-[#0E1113]">
                <th className="px-5 py-3 text-left font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Solution</th>
                <th className="px-5 py-3 text-left font-mono text-[11px] uppercase tracking-wider text-[#565E66]">Critical gap</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1A1E21]">
              {[
                ["Be My Eyes / Seeing AI", "Requires internet and a human operator or cloud inference"],
                ["Generic VLM assistants", "Verbose, causes alert fatigue — repeats the same warning every second"],
                ["Custom hardware devices", "Expensive, inaccessible, requires technical setup"],
                ["All of the above", "None retain persistent memory of the user's specific home environment"],
              ].map(([sol, gap], i) => (
                <tr key={i} className={i === 3 ? "bg-[#0E1113]" : ""}>
                  <td className={`px-5 py-3.5 font-medium ${i === 3 ? "text-[#5EEAD4]" : "text-[#C5CBCF]"}`}>{sol}</td>
                  <td className={`px-5 py-3.5 ${i === 3 ? "font-medium text-[#5EEAD4]" : "text-[#8B95A1]"}`}>{gap}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── REAL NUMBERS ─────────────────────────────────────────────────── */}
      <section>
        <p className="mb-2 font-mono text-xs uppercase tracking-widest text-[#565E66]">Measured results</p>
        <h2 className="mb-6 font-display text-2xl font-semibold text-[#E7ECEE]">
          Real numbers. No estimates.
        </h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard value="66.7" unit="%" label="Navigation success rate" sub="Full system · 4 test videos" />
          <StatCard value="94.9" unit="%" label="False-alert reduction" sub="Baseline B → Full system" />
          <StatCard value="62" unit="ms" label="YOLO inference" sub="Per frame · CPU · no GPU" />
          <StatCard value="414" label="Frames processed" sub="All components confirmed REAL" />
        </div>

        {/* Ablation bars */}
        <div className="mt-6 rounded-xl border border-[#262B2F] bg-[#0E1113] p-6">
          <p className="mb-5 font-mono text-xs uppercase tracking-widest text-[#565E66]">
            Ablation study — navigation success rate
          </p>
          <div className="space-y-4">
            <AblationBar rate={METRICS.baselineA.rate}  label="A — YOLO only (no depth, no memory)" />
            <AblationBar rate={METRICS.baselineB.rate}  label="B — + Depth + Risk scoring" />
            <AblationBar rate={METRICS.fullSystem.rate} label="Full system — + Memory + Suppression" isHighlighted />
          </div>
          <p className="mt-5 font-mono text-[11px] text-[#565E66]">
            Adding depth+risk: 16.7% → 66.7% (+50pp). Memory+suppression: maintains recall, cuts false alerts by 94.9%.
            Measured on 4 indoor walking clips · every 5th frame · all pipeline components REAL (no stubs).
          </p>
        </div>
      </section>

      {/* ── PIPELINE ─────────────────────────────────────────────────────── */}
      <section>
        <p className="mb-2 font-mono text-xs uppercase tracking-widest text-[#565E66]">Architecture</p>
        <h2 className="mb-6 font-display text-2xl font-semibold text-[#E7ECEE]">
          Seven-layer perception pipeline
        </h2>
        <div className="relative">
          {/* Connecting line */}
          <div
            aria-hidden
            className="absolute left-[27px] top-6 bottom-6 w-px"
            style={{ background: "linear-gradient(to bottom, #5EEAD4, #1A1E21)" }}
          />
          <ol className="space-y-3 pl-14">
            {PIPELINE_STEPS.map((step) => (
              <li key={step.id} className="relative">
                {/* Step dot */}
                <span
                  className="absolute -left-14 flex h-8 w-8 items-center justify-center rounded-full border font-mono text-[11px] font-medium"
                  style={{
                    borderColor: `${step.color}40`,
                    backgroundColor: `${step.color}10`,
                    color: step.color,
                  }}
                >
                  {step.id}
                </span>
                <div className="rounded-lg border border-[#262B2F] bg-[#0E1113] px-5 py-3.5">
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-[#E7ECEE]">{step.label}</span>
                    <span className="font-mono text-[11px] text-[#565E66]">{step.sub}</span>
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ── NOVEL CONTRIBUTION ───────────────────────────────────────────── */}
      <section>
        <p className="mb-2 font-mono text-xs uppercase tracking-widest text-[#565E66]">What's different</p>
        <h2 className="mb-6 font-display text-2xl font-semibold text-[#E7ECEE]">
          Novel contributions
        </h2>
        <div className="grid gap-4 sm:grid-cols-3">
          {[
            {
              icon: "◈",
              title: "Persistent home memory",
              body: "ChromaDB + sentence-transformers stores room layout across sessions. Retrievs the right context at inference time — not generic descriptions.",
            },
            {
              icon: "◎",
              title: "Temporal alert suppression",
              body: "WalkVLM-inspired 4-second suppression window. Same chair — silence. Reduces alert fatigue, the #1 usability failure in prior systems.",
            },
            {
              icon: "◉",
              title: "Fully offline, fully private",
              body: "YOLO, Depth-Anything, Whisper, ChromaDB — all on-device. Camera frames never leave the hardware. Privacy-by-Design, GDPR Art. 25.",
            },
          ].map((card) => (
            <div key={card.title} className="rounded-xl border border-[#262B2F] bg-[#0E1113] p-6">
              <span className="mb-3 block font-mono text-2xl text-[#5EEAD4]">{card.icon}</span>
              <h3 className="mb-2 font-display text-base font-semibold text-[#E7ECEE]">{card.title}</h3>
              <p className="text-sm text-[#8B95A1]">{card.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── RESEARCH BASIS ───────────────────────────────────────────────── */}
      <section>
        <p className="mb-2 font-mono text-xs uppercase tracking-widest text-[#565E66]">Research basis</p>
        <h2 className="mb-6 font-display text-2xl font-semibold text-[#E7ECEE]">
          Built on peer-reviewed work
        </h2>
        <div className="space-y-3">
          {RESEARCH_REFS.map((ref) => (
            <a
              key={ref.short}
              href={ref.href}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex gap-4 rounded-xl border border-[#262B2F] bg-[#0E1113] px-5 py-4 transition hover:border-[#5EEAD4]/30"
            >
              <div className="flex-shrink-0">
                <span className="font-display text-sm font-semibold text-[#5EEAD4]">{ref.short}</span>
                <span className="ml-2 font-mono text-[11px] text-[#565E66]">{ref.year}</span>
              </div>
              <p className="text-sm text-[#8B95A1] group-hover:text-[#C5CBCF]">{ref.note}</p>
              <span className="ml-auto flex-shrink-0 font-mono text-xs text-[#565E66] group-hover:text-[#5EEAD4]">↗</span>
            </a>
          ))}
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────────────── */}
      <section className="rounded-2xl border border-[#5EEAD4]/20 bg-[#0E1113] px-10 py-12 text-center">
        <h2 className="font-display text-2xl font-semibold text-[#E7ECEE]">
          Try it or explore the results
        </h2>
        <p className="mx-auto mt-3 max-w-lg text-sm text-[#8B95A1]">
          Connect a webcam, teach MemoryNav your home in the Setup page, and watch
          the live dashboard. Or jump straight to the ablation study for the measured numbers.
        </p>
        <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/dashboard"
            className="rounded-lg border border-[#5EEAD4]/50 bg-[#5EEAD4]/10 px-7 py-3 text-sm font-medium text-[#5EEAD4] transition hover:bg-[#5EEAD4]/20"
          >
            Open Dashboard →
          </Link>
          <Link
            href="/setup"
            className="rounded-lg border border-[#262B2F] bg-[#14181B] px-7 py-3 text-sm font-medium text-[#8B95A1] transition hover:text-[#E7ECEE]"
          >
            Teach my home
          </Link>
          <Link
            href="/evaluation"
            className="rounded-lg border border-[#262B2F] bg-[#14181B] px-7 py-3 text-sm font-medium text-[#8B95A1] transition hover:text-[#E7ECEE]"
          >
            Ablation results
          </Link>
        </div>
      </section>

      {/* ── FOOTER ───────────────────────────────────────────────────────── */}
      <footer className="border-t border-[#1A1E21] pt-8">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded border border-[#5EEAD4]/30 bg-[#5EEAD4]/10 font-mono text-[9px] font-bold text-[#5EEAD4]">M</span>
            <span className="font-mono text-xs text-[#565E66]">MemoryNav v1.0 · MIT License</span>
          </div>
          <p className="font-mono text-[11px] text-[#565E66]">
            Inspired by WalkVLM · VISA · VIALM Survey · NavSpace (ICRA 2026)
          </p>
        </div>
      </footer>

    </div>
  );
}
