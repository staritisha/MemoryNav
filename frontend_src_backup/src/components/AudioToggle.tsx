// frontend/src/components/AudioToggle.tsx
"use client";

import { useState } from "react";
import { updatePrefs } from "@/lib/api";

interface AudioToggleProps {
  enabled: boolean;
  onChange?: (enabled: boolean) => void;
}

export default function AudioToggle({ enabled, onChange }: AudioToggleProps) {
  const [isEnabled, setIsEnabled] = useState(enabled);
  const [saving, setSaving] = useState(false);

  async function toggle() {
    const next = !isEnabled;
    setIsEnabled(next);
    onChange?.(next);
    setSaving(true);
    try {
      await updatePrefs({ audioEnabled: next });
    } catch (err) {
      // Revert on failure so the UI never lies about backend state.
      setIsEnabled(!next);
      onChange?.(!next);
      console.error("Failed to update audio preference", err);
    } finally {
      setSaving(false);
    }
  }

  return (
    <button
      type="button"
      role="switch"
      aria-checked={isEnabled}
      aria-label={isEnabled ? "Mute spoken alerts" : "Unmute spoken alerts"}
      onClick={toggle}
      disabled={saving}
      className="flex items-center gap-3 rounded-full border border-[#E5E7EB] bg-white px-4 py-2 text-sm font-medium text-[#0F2E22] transition disabled:opacity-60"
    >
      <span
        className={`flex h-5 w-9 items-center rounded-full px-0.5 transition-colors ${
          isEnabled ? "bg-[#1FBF82]" : "bg-[#D1D5DB]"
        }`}
      >
        <span
          className={`h-4 w-4 rounded-full bg-white shadow transition-transform ${
            isEnabled ? "translate-x-4" : "translate-x-0"
          }`}
        />
      </span>
      {isEnabled ? "Voice alerts on" : "Voice alerts off"}
    </button>
  );
}
