// frontend/src/components/LiveFeed.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import type { Detection, RiskLevel } from "@/lib/types";

const RISK_STROKE: Record<RiskLevel, string> = {
  LOW: "#4ADE80",
  MEDIUM: "#FFB84D",
  HIGH: "#FF5C5C",
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
  const [fps, setFps] = useState(0);
  const frameTimesRef = useRef<number[]>([]);

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

  // Draw bounding boxes + HUD reticle on every detections update.
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth || canvas.clientWidth;
    canvas.height = video.videoHeight || canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Track rolling FPS from update cadence.
    const now = performance.now();
    frameTimesRef.current.push(now);
    frameTimesRef.current = frameTimesRef.current.filter((t) => now - t < 1000);
    setFps(frameTimesRef.current.length);

    const stroke = RISK_STROKE[riskLevel];

    detections.forEach((d) => {
      const x = d.box.x * canvas.width;
      const y = d.box.y * canvas.height;
      const w = d.box.width * canvas.width;
      const h = d.box.height * canvas.height;

      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const label = `${d.label.toUpperCase()} ${Math.round(d.confidence * 100)}%${
        d.distanceMeters !== undefined ? ` ${d.distanceMeters.toFixed(1)}M` : ""
      }`;
      ctx.font = "600 11px 'JetBrains Mono', monospace";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = stroke;
      ctx.fillRect(x, Math.max(0, y - 18), textWidth + 10, 18);
      ctx.fillStyle = "#0A0C0E";
      ctx.fillText(label, x + 5, Math.max(13, y - 5));
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
    <div className="overflow-hidden rounded-lg border border-[#262B2F] bg-[#0A0C0E]">
      <div className="relative aspect-video w-full">
        <video
          ref={videoRef}
          muted
          playsInline
          className="h-full w-full object-cover opacity-95 grayscale-[15%]"
        />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full"
        />
        <canvas ref={captureCanvasRef} className="hidden" />

        {/* Vignette for instrument feel */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            boxShadow: "inset 0 0 80px 10px rgba(0,0,0,0.55)",
          }}
        />

        {/* Corner brackets — viewfinder framing, recolors with risk level */}
        {(["tl", "tr", "bl", "br"] as const).map((corner) => (
          <span
            key={corner}
            aria-hidden
            className={`pointer-events-none absolute h-5 w-5 border-[2px] ${
              corner === "tl" ? "left-3 top-3 border-r-0 border-b-0" : ""
            } ${corner === "tr" ? "right-3 top-3 border-l-0 border-b-0" : ""} ${
              corner === "bl" ? "left-3 bottom-3 border-r-0 border-t-0" : ""
            } ${
              corner === "br" ? "right-3 bottom-3 border-l-0 border-t-0" : ""
            }`}
            style={{ borderColor: stroke, transition: "border-color 200ms ease" }}
          />
        ))}

        {/* Live status pill, top-left */}
        <div className="absolute left-3 top-3 ml-7 flex items-center gap-2 rounded bg-black/50 px-2.5 py-1 backdrop-blur-sm">
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute h-full w-full animate-ping rounded-full bg-[#5EEAD4] opacity-70" />
            <span className="relative h-1.5 w-1.5 rounded-full bg-[#5EEAD4]" />
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider text-white">
            {cameraReady ? "Rec" : "Connecting"}
          </span>
        </div>

        {/* FPS readout, top-right */}
        <div className="absolute right-3 top-3 mr-7 rounded bg-black/50 px-2.5 py-1 font-mono text-[10px] text-white backdrop-blur-sm">
          {fps} FPS
        </div>

        {cameraError && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0A0C0E] px-6 text-center">
            <p className="font-mono text-sm text-[#FF5C5C]">
              Camera unavailable — {cameraError}
            </p>
          </div>
        )}
      </div>

      {/* Telemetry strip beneath the feed */}
      <div className="flex items-center justify-between border-t border-[#262B2F] bg-[#0E1113] px-4 py-2">
        <span className="font-mono text-[11px] text-[#565E66]">
          {detections.length} object{detections.length === 1 ? "" : "s"} tracked
        </span>
        <span className="font-mono text-[11px] text-[#565E66]">
          risk = (1 / d) &times; motion &times; context
        </span>
      </div>
    </div>
  );
}
