// frontend/src/app/preferences/page.tsx
"use client";

import { useEffect, useState } from "react";
import { getPrefs, updatePrefs } from "@/lib/api";
import type { UserPreferences } from "@/lib/types";

const LANGUAGES = [
  { value: "en-US", label: "English (US)" },
  { value: "en-GB", label: "English (UK)" },
  { value: "hi-IN", label: "Hindi" },
  { value: "es-ES", label: "Spanish" },
];

const ALERT_OPTIONS: { value: UserPreferences["alertFrequency"]; label: string; hint: string }[] = [
  { value: "ALL", label: "All detections", hint: "Speak every change in view" },
  { value: "MEDIUM_AND_UP", label: "Medium and up", hint: "Skip low-risk chatter" },
  { value: "HIGH_ONLY", label: "High risk only", hint: "Only urgent alerts" },
];

const DEFAULTS: UserPreferences = {
  speechSpeed: 1,
  language: "en-US",
  alertFrequency: "MEDIUM_AND_UP",
  audioEnabled: true,
};

export default function PreferencesPage() {
  const [prefs, setPrefs] = useState<UserPreferences>(DEFAULTS);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  useEffect(() => {
    getPrefs()
      .then(setPrefs)
      .catch(() => setPrefs(DEFAULTS))
      .finally(() => setLoading(false));
  }, []);

  async function handleSave() {
    setSaving(true);
    try {
      const updated = await updatePrefs(prefs);
      setPrefs(updated);
      setSavedAt(new Date().toLocaleTimeString());
    } catch (err) {
      console.error("Failed to save preferences", err);
    } finally {
      setSaving(false);
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
          Voice & alerts
        </h1>
      </header>

      <div className="space-y-7 rounded-lg border border-[#262B2F] bg-[#14181B] p-6">
        {/* Speech speed */}
        <div>
          <label className="mb-2 flex items-center justify-between text-sm font-medium text-[#E7ECEE]">
            Speech speed
            <span className="font-mono text-xs text-[#5EEAD4]">
              {prefs.speechSpeed.toFixed(2)}x
            </span>
          </label>
          <input
            type="range"
            min={0.5}
            max={2}
            step={0.05}
            value={prefs.speechSpeed}
            onChange={(e) =>
              setPrefs((p) => ({ ...p, speechSpeed: Number(e.target.value) }))
            }
            className="w-full"
          />
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-wider text-[#565E66]">
            <span>Slower</span>
            <span>Normal</span>
            <span>Faster</span>
          </div>
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

        {/* Alert frequency */}
        <div>
          <label className="mb-2 block text-sm font-medium text-[#E7ECEE]">
            Alert frequency
          </label>
          <div className="space-y-2">
            {ALERT_OPTIONS.map((opt) => (
              <label
                key={opt.value}
                className="flex cursor-pointer items-center justify-between rounded-md border px-4 py-3 transition"
                style={{
                  borderColor: prefs.alertFrequency === opt.value ? "#5EEAD4" : "#262B2F",
                  backgroundColor:
                    prefs.alertFrequency === opt.value ? "rgba(94,234,212,0.08)" : "#0E1113",
                }}
              >
                <div>
                  <p className="text-sm font-medium text-[#E7ECEE]">{opt.label}</p>
                  <p className="font-mono text-[11px] text-[#565E66]">{opt.hint}</p>
                </div>
                <input
                  type="radio"
                  name="alertFrequency"
                  value={opt.value}
                  checked={prefs.alertFrequency === opt.value}
                  onChange={() =>
                    setPrefs((p) => ({ ...p, alertFrequency: opt.value }))
                  }
                />
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-5 flex items-center justify-end gap-3">
        {savedAt && (
          <span className="font-mono text-xs text-[#565E66]">
            Saved at {savedAt}
          </span>
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
