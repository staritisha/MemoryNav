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
      className="flex items-center gap-3 rounded-md border border-[#262B2F] bg-[#14181B] px-3.5 py-2 transition disabled:opacity-50"
    >
      <span
        className="flex h-5 w-9 items-center rounded-full px-0.5 transition-colors"
        style={{ backgroundColor: isEnabled ? "#5EEAD4" : "#3A4146" }}
      >
        <span
          className={`h-4 w-4 rounded-full bg-[#0A0C0E] shadow transition-transform ${
            isEnabled ? "translate-x-4" : "translate-x-0"
          }`}
        />
      </span>
      <span className="font-mono text-xs uppercase tracking-wide text-[#C5CBCF]">
        {isEnabled ? "Voice on" : "Voice muted"}
      </span>
    </button>
  );
}
