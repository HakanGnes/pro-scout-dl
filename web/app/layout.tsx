import type { Metadata, Viewport } from "next";
import "@fontsource-variable/archivo/standard.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pro-Scout · Tactics board",
  description: "Replace players in a 2017/18 squad and see how passing-network models expect the team's chances to change.",
};
export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
        <footer className="footer">
          <p>
            Pro-Scout DL, a non-commercial research project. Match data: Wyscout public dataset, 2017/18 season
            (Pappalardo et al., <i>Scientific Data</i> 2019, CC BY 4.0). Code and full results:{" "}
            <a href="https://github.com/HakanGnes/pro-scout-dl">github.com/HakanGnes/pro-scout-dl</a>.
          </p>
        </footer>
      </body>
    </html>
  );
}
