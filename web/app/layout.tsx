import type { Metadata, Viewport } from "next";
import "@fontsource-variable/archivo/standard.css";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: { default: "Pro-Scout", template: "%s · Pro-Scout" },
  description: "Football scouting models, each tested against a simple baseline: a tactics board, transfer rumours and market values.",
};
export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        {children}
        <footer className="footer">
          <p>
            Pro-Scout DL, a non-commercial research project. Match data: Wyscout public dataset, 2017/18
            (Pappalardo et al., <i>Scientific Data</i> 2019, CC BY 4.0). Market values and transfers: Transfermarkt via
            dcaribou/transfermarkt-datasets (CC0). Rumours: headlines archived by the Internet Archive; the headlines
            themselves are not shown. Code, pre-registrations and full results:{" "}
            <a href="https://github.com/HakanGnes/pro-scout-dl">github.com/HakanGnes/pro-scout-dl</a>.
          </p>
        </footer>
      </body>
    </html>
  );
}
