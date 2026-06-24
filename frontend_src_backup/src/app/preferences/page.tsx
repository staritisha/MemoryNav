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
        <p className="font-mono text-sm text-[#9CA3AF]">Loading preferences…</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl">
      <header className="mb-6">
        <p className="font-mono text-xs uppercase tracking-wide text-[#6B7280]">
          Preferences
        </p>
        <h1 className="mt-1 font-display text-2xl font-semibold text-[#0F2E22]">
          Voice & alerts
        </h1>
      </header>

      <div className="space-y-6 rounded-2xl border border-[#E5E7EB] bg-white p-6">
        {/* Speech speed */}
        <div>
          <label className="mb-2 flex items-center justify-between text-sm font-medium text-[#0F2E22]">
            Speech speed
            <span className="font-mono text-xs text-[#6B7280]">
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
            className="w-full accent-[#1FBF82]"
          />
          <div className="flex justify-between font-mono text-[11px] text-[#9CA3AF]">
            <span>Slower</span>
            <span>Normal</span>
            <span>Faster</span>
          </div>
        </div>

        {/* Language */}
        <div>
          <label className="mb-2 block text-sm font-medium text-[#0F2E22]">
            Language
          </label>
          <select
            value={prefs.language}
            onChange={(e) => setPrefs((p) => ({ ...p, language: e.target.value }))}
            className="w-full rounded-xl border border-[#E5E7EB] bg-white px-3 py-2.5 text-sm text-[#0F2E22] outline-none focus:border-[#1FBF82]"
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
          <label className="mb-2 block text-sm font-medium text-[#0F2E22]">
            Alert frequency
          </label>
          <div className="space-y-2">
            {ALERT_OPTIONS.map((opt) => (
              <label
                key={opt.value}
                className={`flex cursor-pointer items-center justify-between rounded-xl border px-4 py-3 transition ${
                  prefs.alertFrequency === opt.value
                    ? "border-[#1FBF82] bg-[#E8F8EF]"
                    : "border-[#E5E7EB] bg-white"
                }`}
              >
                <div>
                  <p className="text-sm font-medium text-[#0F2E22]">{opt.label}</p>
                  <p className="font-mono text-[11px] text-[#6B7280]">{opt.hint}</p>
                </div>
                <input
                  type="radio"
                  name="alertFrequency"
                  value={opt.value}
                  checked={prefs.alertFrequency === opt.value}
                  onChange={() =>
                    setPrefs((p) => ({ ...p, alertFrequency: opt.value }))
                  }
                  className="accent-[#1FBF82]"
                />
              </label>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-5 flex items-center justify-end gap-3">
        {savedAt && (
          <span className="font-mono text-xs text-[#6B7280]">
            Saved at {savedAt}
          </span>
        )}
        <button
          onClick={handleSave}
          disabled={saving}
          className="rounded-full bg-[#0F2E22] px-5 py-2.5 text-sm font-medium text-white transition hover:bg-[#14532D] disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save preferences"}
        </button>
      </div>
    </div>
  );
}
