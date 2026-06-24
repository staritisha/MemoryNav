// frontend/src/lib/types.ts
// Shared types for the perception pipeline, memory system, and preferences.

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
  /** Estimated distance to camera in meters, if the depth stage has run. */
  distanceMeters?: number;
}

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

/** Full payload pushed down the /ws WebSocket after each processed frame. */
export interface DetectionFrame {
  timestamp: string; // ISO 8601
  detections: Detection[];
  riskLevel: RiskLevel;
  /** Free-text reason the risk engine assigned this level, for display + TTS. */
  riskReason?: string;
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
