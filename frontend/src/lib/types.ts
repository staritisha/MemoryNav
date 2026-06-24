// frontend/src/lib/types.ts
// Shared types for the perception pipeline, memory system, preferences,
// and evaluation/ablation reporting.

/** A single bounding box detection from the YOLOv8 backend. */
export interface Detection {
  id: string;
  label: string;
  confidence: number; // 0..1
  /** Bounding box in normalized [0,1] coords relative to frame width/height. */
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  /** Estimated distance to camera in meters, from Depth-Anything. */
  distanceMeters?: number;
}

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

/** Full payload pushed down the /ws WebSocket after each processed frame. */
export interface DetectionFrame {
  timestamp: string; // ISO 8601
  detections: Detection[];
  riskLevel: RiskLevel;
  /** Raw 0..1 score from (1/distance) x motion x context, for display. */
  riskScore?: number;
  /** Free-text reason the risk engine assigned this level, for display + TTS. */
  riskReason?: string;
  /** Whether the alert manager suppressed speech for this frame, and why. */
  suppressed?: boolean;
  suppressionReason?: string;
  /** End-to-end detection -> speech latency in ms, if measured this frame. */
  latencyMs?: number;
}

/** A piece of retrieved long-term home context from ChromaDB. */
export interface HomeContext {
  id: string;
  text: string;
  /** Cosine similarity / relevance score to the current query, if available. */
  score?: number;
  createdAt: string; // ISO 8601
  tags?: string[];
}

export interface UserPreferences {
  speechSpeed: number; // 0.5 - 2.0, 1.0 = normal
  language: string; // e.g. "en-US"
  alertFrequency: "ALL" | "MEDIUM_AND_UP" | "HIGH_ONLY";
  audioEnabled: boolean;
}

// --- Pipeline visualization (Layers 2-9 of the architecture) ---

export type PipelineStageId =
  | "quality"
  | "detect"
  | "depth"
  | "risk"
  | "memory"
  | "alert"
  | "voice";

export type PipelineStageStatus = "idle" | "active" | "warn" | "off";

export interface PipelineStage {
  id: PipelineStageId;
  label: string;
  status: PipelineStageStatus;
}

// --- Evaluation / ablation study (Section 6 of the design doc) ---

export interface AblationRow {
  configuration: string;
  description: string;
  /** Measured navigation success rate, 0..1. Null until real numbers exist. */
  successRate: number | null;
}

export interface ComponentMetric {
  component: string;
  metric: string;
  description: string;
  /** Measured value. Null renders as "pending measurement" — never fabricate. */
  value: number | null;
  unit: string;
}
