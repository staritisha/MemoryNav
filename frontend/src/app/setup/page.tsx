// frontend/src/app/setup/page.tsx
"use client";

import { useEffect, useState } from "react";
import { addMemory, deleteMemory, getMemory } from "@/lib/api";
import type { HomeContext } from "@/lib/types";

export default function SetupPage() {
  const [text, setText] = useState("");
  const [entries, setEntries] = useState<HomeContext[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    refresh();
  }, []);

  function refresh() {
    getMemory().then(setEntries).catch(() => setEntries([]));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;

    setSaving(true);
    setError(null);
    try {
      await addMemory(text.trim());
      setText("");
      refresh();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not save that memory."
      );
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteMemory(id);
      setEntries((prev) => prev.filter((e) => e.id !== id));
    } catch (err) {
      console.error("Failed to delete memory", err);
    }
  }

  return (
    <div className="mx-auto max-w-3xl">
      <header className="mb-6">
        <p className="font-mono text-xs uppercase tracking-wider text-[#565E66]">
          Long-term spatial memory · ChromaDB
        </p>
        <h1 className="mt-1 font-display text-2xl font-semibold text-[#E7ECEE]">
          Teach MemoryNav your home
        </h1>
        <p className="mt-2 text-sm text-[#8B95A1]">
          Each entry below is embedded and stored as a memory node. At
          inference time, the risk engine retrieves the most relevant node
          for what&apos;s in frame — &ldquo;chair ahead, near where you said
          the rug is.&rdquo;
        </p>
      </header>

      <form onSubmit={handleSubmit} className="mb-8">
        <div className="rounded-lg border border-[#262B2F] bg-[#0E1113] p-1">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. The kitchen has a low step down from the hallway, on the left as you enter."
            rows={4}
            className="w-full resize-none rounded-md bg-transparent p-4 text-sm text-[#E7ECEE] outline-none placeholder:text-[#565E66]"
          />
        </div>
        {error && <p className="mt-2 font-mono text-xs text-[#FF5C5C]">{error}</p>}
        <div className="mt-3 flex items-center justify-between">
          <span className="font-mono text-[11px] text-[#565E66]">
            sentence-transformers · local embedding
          </span>
          <button
            type="submit"
            disabled={saving || !text.trim()}
            className="rounded-md border border-[#5EEAD4]/40 bg-[#5EEAD4]/10 px-5 py-2 text-sm font-medium text-[#5EEAD4] transition hover:bg-[#5EEAD4]/20 disabled:opacity-40"
          >
            {saving ? "Embedding…" : "Save to memory"}
          </button>
        </div>
      </form>

      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-mono text-xs uppercase tracking-wider text-[#565E66]">
            Memory nodes ({entries.length})
          </h2>
        </div>

        {entries.length === 0 ? (
          <p className="rounded-lg border border-dashed border-[#262B2F] px-4 py-10 text-center font-mono text-sm text-[#565E66]">
            No memory nodes yet. Describe a room, hazard, or routine above to
            begin.
          </p>
        ) : (
          <ul className="space-y-2">
            {entries.map((entry) => (
              <li
                key={entry.id}
                className="group flex items-start justify-between gap-4 rounded-lg border border-[#262B2F] bg-[#14181B] px-4 py-3"
              >
                <div className="flex items-start gap-3">
                  <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-[#5EEAD4]" />
                  <p className="text-sm text-[#C5CBCF]">{entry.text}</p>
                </div>
                <button
                  onClick={() => handleDelete(entry.id)}
                  aria-label="Delete this memory"
                  className="flex-shrink-0 font-mono text-[11px] text-[#565E66] opacity-0 transition group-hover:opacity-100 hover:text-[#FF5C5C]"
                >
                  remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
