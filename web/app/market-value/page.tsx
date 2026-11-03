import type { Metadata } from "next";
import MarketValue from "@/components/MarketValue";

export const metadata: Metadata = { title: "Market value" };

export default function Page() {
  return <main><MarketValue /></main>;
}
