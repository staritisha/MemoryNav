import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",   // required for the Docker multi-stage build
  // Allow images from any origin (needed if you add webcam frame previews)
  images: {
    remotePatterns: [],
  },
};

export default nextConfig;
