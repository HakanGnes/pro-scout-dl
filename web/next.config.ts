import type { NextConfig } from "next";

// A static site: every page is exported to out/ and the models run in the visitor's browser.
const config: NextConfig = { output: "export", images: { unoptimized: true }, trailingSlash: true };
export default config;
