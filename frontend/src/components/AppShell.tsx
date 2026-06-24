// frontend/src/components/AppShell.tsx
// Conditionally renders the sidebar. The landing page (/) gets a full-width
// centered layout; all other pages get the sidebar + content shell.
"use client";

import { usePathname } from "next/navigation";
import Sidebar from "./Sidebar";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isLanding = pathname === "/";

  if (isLanding) {
    return (
      <div className="min-h-screen overflow-y-auto px-6 py-10 sm:px-10">
        {children}
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 overflow-y-auto px-10 py-8">{children}</main>
    </div>
  );
}
