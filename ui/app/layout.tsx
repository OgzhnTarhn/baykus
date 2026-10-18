import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// Fontlar repoda (Geist, SIL OFL 1.1: public/fonts/OFL.txt); CDN yok (Faz 3 tasarımı §4.4).
const geistSans = localFont({
  src: "../public/fonts/Geist-Variable.woff2",
  variable: "--font-geist-sans",
});

const geistMono = localFont({
  src: "../public/fonts/GeistMono-Variable.woff2",
  variable: "--font-geist-mono",
});

export const metadata: Metadata = {
  title: "Baykuş",
  description: "RGB + IR araç tespiti, sensör sağlığı ve takip",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="tr" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
