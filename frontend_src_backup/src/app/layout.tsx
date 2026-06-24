// frontend/src/app/layout.tsx
import type { Metadata } from "next";
import { Manrope, Inter, JetBrains_Mono } from "next/font/google";
import Sidebar from "@/components/Sidebar";
import "./globals.css";

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["500", "600", "700", "800"],
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-body",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "MemoryNav",
  description: "Local-first perception and home-context assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={`${manrope.variable} ${inter.variable} ${jetbrainsMono.variable} flex min-h-screen bg-[#F7F8F6] font-body text-[#0F2E22] antialiased`}
      >
        <Sidebar />
        <main className="flex-1 overflow-y-auto px-10 py-8">{children}</main>
      </body>
    </html>
  );
}
