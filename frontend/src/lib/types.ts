// frontend/src/lib/types.ts
// Shared types — kept in sync with the backend WS + REST contracts.

// ── Perception ───────────────────────────────────────────────────────────────

/** A single bounding box detection from the YOLOv8 backend. */
export interface Detection {
  id: string;
  label: string;
  confidence: number; // 0..1
  /** Normalized [0,1] coords relative to frame width/height. */
  box: { x: number; y: number; width: number; height: number };
  /** Metric metres from Depth-Anything V2. Undefined if depth failed. */
  distanceMeters?: number;
  // Extra telemetry from the pipeline (not required by UI components)
  riskScore?: number;
  riskLevel?: RiskLevel;
  motionTrend?: string;
  isConfident?: boolean;
  contextWeight?: number;
  spatialMemory?: string;
}

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

// ── WebSocket frame ───────────────────────────────────────────────────────────

/** Full payload pushed per processed frame from /ws. */
export interface DetectionFrame {
  timestamp: number;          // unix epoch ms (ts from backend)
  detections: Detection[];
  riskLevel: RiskLevel;
  riskScore?: number;
  riskReason?: string;
  suppressed?: boolean;
  suppressionReason?: string;
  latencyMs?: number;
  // Extension fields
  status?: string;
  spokeGhost?: boolean;
  spoke?: string | null;
  memoryContext?: string | null;
  suppressionStats?: { evaluated: number; spoken: number; suppressed: number };
  spatialMap?: SpatialMapSnapshot;
}

// ── Spatial map ───────────────────────────────────────────────────────────────

export interface SpatialObjectEntry {
  position: "left" | "center" | "right";
  distance_m: number | null;
  last_seen_s_ago: number;
  confidence: number;
  sightings: number;
}

export interface SpatialMapSnapshot {
  current_room: string;
  rooms: Record<string, Record<string, SpatialObjectEntry>>;
}

// ── Memory ────────────────────────────────────────────────────────────────────

/** A memory entry from ChromaDB, as returned by GET /memory. */
export interface HomeContext {
  id: string;
  text: string;
  metadata: Record<string, unknown>;
  similarity?: number;       // present on search results
  createdAt?: string;        // ISO 8601
  tags?: string[];
}

// ── Preferences ───────────────────────────────────────────────────────────────

/** Matches backend PreferencesOut exactly. */
export interface UserPreferences {
  user_id: string;
  speech_rate_wpm: number;   // 80–400 WPM
  language: string;          // BCP-47, e.g. "en-US"
  mobility_flags: string[];
  alert_suppression_seconds: number;
  updated_at?: string | null;
  // Frontend-only, not persisted server-side
  audioEnabled?: boolean;
}

// ── Pipeline visualization ────────────────────────────────────────────────────

export type PipelineStageId =
  | "quality" | "detect" | "depth" | "risk"
  | "memory" | "alert" | "voice";

export type PipelineStageStatus = "idle" | "active" | "warn" | "off";

export interface PipelineStage {
  id: PipelineStageId;
  label: string;
  status: PipelineStageStatus;
}

// ── Evaluation ────────────────────────────────────────────────────────────────

export interface AblationRow {
  configuration: string;
  description: string;
  successRate: number | null;
}

export interface ComponentMetric {
  component: string;
  metric: string;
  description: string;
  value: number | null;
  unit: string;
}
