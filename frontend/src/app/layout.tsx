import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import Nav from "@/components/nav";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff2",
  variable: "--font-geist-sans",
  weight: "100 900",
});

const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff2",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "CineMatch",
  description:
    "Personalized movie recommendations powered by SVD collaborative filtering",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} dark antialiased`}
    >
      <body className="min-h-screen flex flex-col">
        <Nav />
        <main className="flex-1 max-w-6xl mx-auto w-full px-6 py-8">
          {children}
        </main>
        <footer className="border-t border-[var(--border)] py-6 text-center text-sm text-[var(--muted)]">
          Built with SVD collaborative filtering on the MovieLens dataset.{" "}
          <a
            href="https://github.com/CeeJay1241/Movie-Recomendation-System"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--accent-light)] hover:underline"
          >
            View source
          </a>
        </footer>
      </body>
    </html>
  );
}
