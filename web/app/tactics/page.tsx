import type { Metadata } from "next";
import Board from "@/components/Board";

export const metadata: Metadata = { title: "Tactics board" };

export default function Page() {
  return <main><Board /></main>;
}
