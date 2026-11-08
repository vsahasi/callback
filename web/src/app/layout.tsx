import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Callback review",
  description: "Live verification evidence for Callback calls.",
};

// Runs before paint so a saved theme does not flash.
const themeScript = `try{var t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
