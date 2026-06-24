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
        <p className="font-mono text-xs uppercase tracking-wide text-[#6B7280]">
          Setup
        </p>
        <h1 className="mt-1 font-display text-2xl font-semibold text-[#0F2E22]">
          Teach MemoryNav your home
        </h1>
        <p className="mt-2 text-sm text-[#6B7280]">
          Describe rooms, furniture, hazards, and routines in plain language.
          Each entry is stored locally and used to give context-aware alerts.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="mb-8">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. The kitchen has a low step down from the hallway, on the left as you enter."
          rows={4}
          className="w-full rounded-xl border border-[#E5E7EB] bg-white p-4 text-sm text-[#0F2E22] outline-none placeholder:text-[#9CA3AF] focus:border-[#1FBF82]"
        />
        {error && (
          <p className="mt-2 text-sm text-[#B91C1C]">{error}</p>
        )}
        <div className="mt-3 flex justify-end">
          <button
            type="submit"
            disabled={saving || !text.trim()}
            className="rounded-full bg-[#0F2E22] px-5 py-2.5 text-sm font-medium text-white transition hover:bg-[#14532D] disabled:opacity-50"
          >
            {saving ? "Saving…" : "Save to memory"}
          </button>
        </div>
      </form>

      <section>
        <h2 className="mb-3 font-mono text-xs uppercase tracking-wide text-[#6B7280]">
          Saved context ({entries.length})
        </h2>

        {entries.length === 0 ? (
          <p className="rounded-xl border border-dashed border-[#E5E7EB] px-4 py-8 text-center text-sm text-[#9CA3AF]">
            Nothing saved yet. Add a few sentences above to get started.
          </p>
        ) : (
          <ul className="space-y-2">
            {entries.map((entry) => (
              <li
                key={entry.id}
                className="flex items-start justify-between gap-4 rounded-xl border border-[#E5E7EB] bg-white px-4 py-3"
              >
                <p className="text-sm text-[#374151]">{entry.text}</p>
                <button
                  onClick={() => handleDelete(entry.id)}
                  aria-label="Delete this memory"
                  className="font-mono text-xs text-[#9CA3AF] hover:text-[#B91C1C]"
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
