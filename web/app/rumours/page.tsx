import type { Metadata } from "next";
import Rumours from "@/components/Rumours";

export const metadata: Metadata = { title: "Transfer rumours" };

export default function Page() {
  return <main><Rumours /></main>;
}
