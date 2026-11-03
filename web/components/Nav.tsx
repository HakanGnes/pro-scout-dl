"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/tactics/", label: "Tactics board" },
  { href: "/rumours/", label: "Transfer rumours" },
  { href: "/market-value/", label: "Market value" },
  { href: "/method/", label: "Method" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <nav className="nav" aria-label="Site">
      <Link href="/" className="wordmark">Pro-Scout</Link>
      <ul>
        {LINKS.map((l) => (
          <li key={l.href}>
            <Link href={l.href} aria-current={path === l.href || path === l.href.slice(0, -1) ? "page" : undefined}>{l.label}</Link>
          </li>
        ))}
        <li><a href="https://github.com/HakanGnes/pro-scout-dl">Code</a></li>
      </ul>
    </nav>
  );
}
