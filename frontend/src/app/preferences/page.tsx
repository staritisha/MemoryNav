// frontend/src/app/preferences/page.tsx
"use client";

import { useEffect, useState } from "react";
import { getPrefs, updatePrefs, addMobilityFlag, removeMobilityFlag } from "@/lib/api";
import type { UserPreferences } from "@/lib/types";

const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "hi", label: "Hindi" },
  { value: "es", label: "Spanish" },
  { value: "ta", label: "Tamil" },
  { value: "mr", label: "Marathi" },
];

const MOBILITY_FLAGS = [
  { flag: "bad_knee",       label: "Bad knee",       hint: "+45% risk weight for floor obstacles" },
  { flag: "uses_walker",    label: "Uses walker",     hint: "+55% - wider clearance needed" },
  { flag: "low_vision",     label: "Low vision",      hint: "+35% - earlier warnings" },
  { flag: "wheelchair",     label: "Wheelchair",      hint: "+60% - most sensitive" },
  { flag: "balance_issues", label: "Balance issues",  hint: "+50% - step / edge hazards" },
];

const DEFAULTS: UserPreferences = {
  user_id: "default_user",
  speech_rate_wpm: 175,
  language: "en",
  mobility_flags: [],
  alert_suppression_seconds: 4.0,
  audioEnabled: true,
};

export default function PreferencesPage() {
  const [prefs, setPrefs]     = useState<UserPreferences>(DEFAULTS);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving]   = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    getPrefs()
      .then(setPrefs)
      .catch(() => setPrefs(DEFAULTS))
      .finally(() => setLoading(false));
  }, []);

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      const updated = await updatePrefs(prefs);
      setPrefs(updated);
      setSavedAt(new Date().toLocaleTimeString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save.");
    } finally {
      setSaving(false);
    }
  }

  async function toggleFlag(flag: string) {
    const has = prefs.mobility_flags.includes(flag);
    try {
      const updated = has
        ? await removeMobilityFlag(flag)
        : await addMobilityFlag(flag);
      setPrefs(updated);
    } catch (err) {
      console.error("Failed to toggle flag", err);
    }
  }

  if (loading) {
    return (
      <div className="mx-auto max-w-2xl">
        <p className="font-mono text-sm text-[#565E66]">Loading preferences…</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl">
      <header className="mb-6">
        <p className="font-mono text-xs uppercase tracking-wider text-[#565E66]">
          User preference memory · SQLite
        </p>
        <h1 className="mt-1 font-display text-2xl font-semibold text-[#E7ECEE]">
          Voice &amp; alerts
        </h1>
      </header>

      <div className="space-y-7 rounded-lg border border-[#262B2F] bg-[#14181B] p-6">

        {/* Speech rate */}
        <div>
          <label className="mb-2 flex items-center justify-between text-sm font-medium text-[#E7ECEE]">
            Speech rate
            <span className="font-mono text-xs text-[#5EEAD4]">
              {prefs.speech_rate_wpm} WPM
            </span>
          </label>
          <input
            type="range"
            min={80}
            max={300}
            step={5}
            value={prefs.speech_rate_wpm}
            onChange={(e) =>
              setPrefs((p) => ({ ...p, speech_rate_wpm: Number(e.target.value) }))
            }
            className="w-full"
          />
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-wider text-[#565E66]">
            <span>80 wpm</span>
            <span>175 wpm</span>
            <span>300 wpm</span>
          </div>
        </div>

        {/* Alert suppression window */}
        <div>
          <label className="mb-2 flex items-center justify-between text-sm font-medium text-[#E7ECEE]">
            Alert suppression window
            <span className="font-mono text-xs text-[#5EEAD4]">
              {prefs.alert_suppression_seconds.toFixed(1)}s
            </span>
          </label>
          <input
            type="range"
            min={1}
            max={15}
            step={0.5}
            value={prefs.alert_suppression_seconds}
            onChange={(e) =>
              setPrefs((p) => ({ ...p, alert_suppression_seconds: Number(e.target.value) }))
            }
            className="w-full"
          />
          <p className="mt-1 font-mono text-[10px] text-[#565E66]">
            Same obstacle won&apos;t repeat a warning for this many seconds (WalkVLM suppression)
          </p>
        </div>

        {/* Language */}
        <div>
          <label className="mb-2 block text-sm font-medium text-[#E7ECEE]">
            Language
          </label>
          <select
            value={prefs.language}
            onChange={(e) => setPrefs((p) => ({ ...p, language: e.target.value }))}
            className="w-full rounded-md border border-[#262B2F] bg-[#0E1113] px-3 py-2.5 text-sm text-[#E7ECEE] outline-none focus:border-[#5EEAD4]"
          >
            {LANGUAGES.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </div>

        {/* Mobility flags - affect Risk Engine context_weight */}
        <div>
          <label className="mb-3 block text-sm font-medium text-[#E7ECEE]">
            Mobility context
            <span className="ml-2 font-mono text-[10px] text-[#565E66]">
              affects risk scoring weights
            </span>
          </label>
          <div className="space-y-2">
            {MOBILITY_FLAGS.map((item) => {
              const active = prefs.mobility_flags.includes(item.flag);
              return (
                <button
                  key={item.flag}
                  type="button"
                  onClick={() => toggleFlag(item.flag)}
                  className="flex w-full items-center justify-between rounded-md border px-4 py-3 text-left transition"
                  style={{
                    borderColor: active ? "#5EEAD4" : "#262B2F",
                    backgroundColor: active ? "rgba(94,234,212,0.08)" : "#0E1113",
                  }}
                >
                  <div>
                    <p className="text-sm font-medium text-[#E7ECEE]">{item.label}</p>
                    <p className="font-mono text-[11px] text-[#565E66]">{item.hint}</p>
                  </div>
                  <span
                    className="h-4 w-4 rounded-full border-2 transition"
                    style={{
                      borderColor: active ? "#5EEAD4" : "#565E66",
                      backgroundColor: active ? "#5EEAD4" : "transparent",
                    }}
                  />
                </button>
              );
            })}
          </div>
        </div>

      </div>

      {error && (
        <p className="mt-3 font-mono text-xs text-[#FF5C5C]">{error}</p>
      )}

      <div className="mt-5 flex items-center justify-end gap-3">
        {savedAt && (
          <span className="font-mono text-xs text-[#565E66]">Saved at {savedAt}</span>
        )}
        <button
          onClick={handleSave}
          disabled={saving}
          className="rounded-md border border-[#5EEAD4]/40 bg-[#5EEAD4]/10 px-5 py-2.5 text-sm font-medium text-[#5EEAD4] transition hover:bg-[#5EEAD4]/20 disabled:opacity-40"
        >
          {saving ? "Saving…" : "Save preferences"}
        </button>
      </div>
    </div>
  );
}
