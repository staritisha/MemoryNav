// frontend/src/lib/api.ts
// REST calls against backend/app/api/memory_router.py and preferences_router.py

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

// --- Memory (long-term home context, ChromaDB) ---

export function getMemory(): Promise<HomeContext[]> {
  return request<HomeContext[]>("/memory");
}

export function addMemory(text: string, tags?: string[]): Promise<HomeContext> {
  return request<HomeContext>("/memory", {
    method: "POST",
    body: JSON.stringify({ text, tags }),
  });
}

export function deleteMemory(id: string): Promise<void> {
  return request<void>(`/memory/${id}`, { method: "DELETE" });
}

// --- Preferences ---

export function getPrefs(): Promise<UserPreferences> {
  return request<UserPreferences>("/preferences");
}

export function updatePrefs(
  prefs: Partial<UserPreferences>
): Promise<UserPreferences> {
  return request<UserPreferences>("/preferences", {
    method: "PUT",
    body: JSON.stringify(prefs),
  });
}
