// frontend/src/lib/ws.ts
// WebSocket client for the live detection stream (backend/app/api/ws_stream.py).

import { useEffect, useRef, useState, useCallback } from "react";
import type { DetectionFrame } from "./types";

const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";

export type ConnectionStatus = "connecting" | "open" | "closed" | "error";

interface UseDetectionResult {
  frame: DetectionFrame | null;
  status: ConnectionStatus;
  /** Send a raw frame (e.g. JPEG bytes from a canvas) up to the backend. */
  sendFrame: (blob: Blob) => void;
}

/**
 * Connects to the detection WebSocket, parses incoming JSON frames, and
 * exposes the latest frame + connection status. Reconnects automatically
 * with backoff if the connection drops.
 */
export function useDetection(): UseDetectionResult {
  const [frame, setFrame] = useState<DetectionFrame | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    function connect() {
      const socket = new WebSocket(WS_URL);
      socketRef.current = socket;
      setStatus("connecting");

      socket.onopen = () => {
        if (cancelled) return;
        retryRef.current = 0;
        setStatus("open");
      };

      socket.onmessage = (event) => {
        if (cancelled) return;
        try {
          const parsed = JSON.parse(event.data) as DetectionFrame;
          setFrame(parsed);
        } catch (err) {
          console.error("Failed to parse detection frame", err);
        }
      };

      socket.onerror = () => {
        if (cancelled) return;
        setStatus("error");
      };

      socket.onclose = () => {
        if (cancelled) return;
        setStatus("closed");
        const delay = Math.min(1000 * 2 ** retryRef.current, 10_000);
        retryRef.current += 1;
        setTimeout(() => {
          if (!cancelled) connect();
        }, delay);
      };
    }

    connect();

    return () => {
      cancelled = true;
      socketRef.current?.close();
    };
  }, []);

  const sendFrame = useCallback((blob: Blob) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(blob);
    }
  }, []);

  return { frame, status, sendFrame };
}
