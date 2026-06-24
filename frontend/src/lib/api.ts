// frontend/src/lib/api.ts
// REST calls against the MemoryNav backend. All response shapes match
// the Pydantic models in backend/app/api/.

import type { HomeContext, UserPreferences } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${init?.method ?? "GET"} ${path} failed (${res.status}): ${body}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── Memory (ChromaDB via /memory) ─────────────────────────────────────────────
// Backend returns { entries: [...], count: N, total_stored: N }
// We unwrap to HomeContext[] for convenience.

interface MemoryListResponse {
  entries: Array<{
    id: string;
    text: string;
    metadata: Record<string, unknown>;
    similarity: number | null;
  }>;
  count: number;
  total_stored: number;
  query: string | null;
}

export async function getMemory(): Promise<HomeContext[]> {
  const res = await request<MemoryListResponse>("/memory");
  return res.entries.map((e) => ({
    id: e.id,
    text: e.text,
    metadata: e.metadata,
    similarity: e.similarity ?? undefined,
  }));
}

export async function searchMemory(q: string, n = 5): Promise<HomeContext[]> {
  const res = await request<MemoryListResponse>(
    `/memory?q=${encodeURIComponent(q)}&n=${n}`
  );
  return res.entries.map((e) => ({
    id: e.id,
    text: e.text,
    metadata: e.metadata,
    similarity: e.similarity ?? undefined,
  }));
}

export async function addMemory(
  text: string,
  opts?: { room?: string; type?: string }
): Promise<HomeContext> {
  const body: Record<string, string> = { text };
  if (opts?.room) body.room = opts.room;
  if (opts?.type) body.type = opts.type;

  const res = await request<{ id: string; text: string; metadata: Record<string, unknown> }>(
    "/memory",
    { method: "POST", body: JSON.stringify(body) }
  );
  return { id: res.id, text: res.text, metadata: res.metadata };
}

export async function deleteMemory(id: string): Promise<void> {
  await request<void>(`/memory/${id}`, { method: "DELETE" });
}

export async function clearAllMemory(): Promise<void> {
  await request<void>("/memory?confirm=true", { method: "DELETE" });
}

export async function getSpatialMap(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/memory/spatial-map");
}

// ── Preferences (/preferences) ────────────────────────────────────────────────

export async function getPrefs(): Promise<UserPreferences> {
  return request<UserPreferences>("/preferences");
}

export async function updatePrefs(
  prefs: Partial<UserPreferences>
): Promise<UserPreferences> {
  // Backend PUT /preferences requires all fields - fill defaults for missing ones
  const full = {
    speech_rate_wpm:           prefs.speech_rate_wpm ?? 175,
    language:                  prefs.language ?? "en",
    mobility_flags:            prefs.mobility_flags ?? [],
    alert_suppression_seconds: prefs.alert_suppression_seconds ?? 4.0,
  };
  return request<UserPreferences>("/preferences", {
    method: "PUT",
    body: JSON.stringify(full),
  });
}

export async function addMobilityFlag(flag: string): Promise<UserPreferences> {
  return request<UserPreferences>("/preferences/mobility", {
    method: "PUT",
    body: JSON.stringify({ flag, action: "add" }),
  });
}

export async function removeMobilityFlag(flag: string): Promise<UserPreferences> {
  return request<UserPreferences>("/preferences/mobility", {
    method: "PUT",
    body: JSON.stringify({ flag, action: "remove" }),
  });
}

// ── Health ─────────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<{ status: string; device: string }> {
  return request<{ status: string; device: string }>("/health");
}
