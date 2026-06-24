// frontend/src/components/LiveFeed.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import type { Detection, RiskLevel } from "@/lib/types";

const RISK_STROKE: Record<RiskLevel, string> = {
  LOW: "#1FBF82",
  MEDIUM: "#D97706",
  HIGH: "#DC2626",
};

interface LiveFeedProps {
  detections: Detection[];
  riskLevel: RiskLevel;
  /** Called with each captured JPEG frame, for sending up the WebSocket. */
  onFrame?: (blob: Blob) => void;
  /** Capture interval in ms. Defaults to ~6 fps, enough for a risk monitor. */
  captureIntervalMs?: number;
}

export default function LiveFeed({
  detections,
  riskLevel,
  onFrame,
  captureIntervalMs = 160,
}: LiveFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Start webcam.
  useEffect(() => {
    let stream: MediaStream | null = null;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          setCameraReady(true);
        }
      } catch (err) {
        setCameraError(
          err instanceof Error ? err.message : "Could not access camera"
        );
      }
    }

    start();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Draw bounding boxes on every detections update.
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth || canvas.clientWidth;
    canvas.height = video.videoHeight || canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const stroke = RISK_STROKE[riskLevel];

    detections.forEach((d) => {
      const x = d.box.x * canvas.width;
      const y = d.box.y * canvas.height;
      const w = d.box.width * canvas.width;
      const h = d.box.height * canvas.height;

      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x, y, w, h);

      const label = `${d.label} ${Math.round(d.confidence * 100)}%${
        d.distanceMeters !== undefined ? ` · ${d.distanceMeters.toFixed(1)}m` : ""
      }`;
      ctx.font = "600 12px 'JetBrains Mono', monospace";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = stroke;
      ctx.fillRect(x, Math.max(0, y - 20), textWidth + 12, 20);
      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(label, x + 6, Math.max(14, y - 6));
    });
  }, [detections, riskLevel]);

  // Periodically capture a frame as JPEG and hand it to onFrame.
  useEffect(() => {
    if (!onFrame || !cameraReady) return;

    const interval = setInterval(() => {
      const video = videoRef.current;
      const captureCanvas = captureCanvasRef.current;
      if (!video || !captureCanvas) return;

      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      const ctx = captureCanvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      captureCanvas.toBlob(
        (blob) => {
          if (blob) onFrame(blob);
        },
        "image/jpeg",
        0.7
      );
    }, captureIntervalMs);

    return () => clearInterval(interval);
  }, [onFrame, cameraReady, captureIntervalMs]);

  const stroke = RISK_STROKE[riskLevel];

  return (
    <div className="relative overflow-hidden rounded-2xl bg-[#0F2E22] shadow-lg">
      <div className="relative aspect-video w-full">
        <video
          ref={videoRef}
          muted
          playsInline
          className="h-full w-full object-cover opacity-95"
        />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full"
        />
        <canvas ref={captureCanvasRef} className="hidden" />

        {/* Corner brackets - signature framing element, evokes a viewfinder/sensor */}
        {(["tl", "tr", "bl", "br"] as const).map((corner) => (
          <span
            key={corner}
            aria-hidden
            className={`pointer-events-none absolute h-6 w-6 border-[3px] ${
              corner === "tl" ? "left-3 top-3 border-r-0 border-b-0" : ""
            } ${corner === "tr" ? "right-3 top-3 border-l-0 border-b-0" : ""} ${
              corner === "bl" ? "left-3 bottom-3 border-r-0 border-t-0" : ""
            } ${
              corner === "br" ? "right-3 bottom-3 border-l-0 border-t-0" : ""
            }`}
            style={{ borderColor: stroke, transition: "border-color 200ms ease" }}
          />
        ))}

        {/* Live status pill */}
        <div className="absolute left-3 bottom-3 flex items-center gap-2 rounded-full bg-black/40 px-3 py-1.5 backdrop-blur-sm">
          <span className="relative flex h-2 w-2">
            <span className="absolute h-full w-full animate-ping rounded-full bg-[#1FBF82] opacity-70" />
            <span className="relative h-2 w-2 rounded-full bg-[#1FBF82]" />
          </span>
          <span className="font-mono text-[11px] uppercase tracking-wide text-white">
            {cameraReady ? "Live" : "Connecting"}
          </span>
        </div>

        {cameraError && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0F2E22] px-6 text-center">
            <p className="font-mono text-sm text-[#FBE6E6]">
              Camera unavailable - {cameraError}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
