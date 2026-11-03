import Link from "next/link";

type Finding = { question: string; verdict: string; tone: "yes" | "no" | "mixed"; detail: React.ReactNode; link?: { href: string; label: string } };

const FINDINGS: Finding[] = [
  {
    question: "Does who passes to whom predict a team's chances?",
    verdict: "Not measurably",
    tone: "no",
    detail: <>A graph network over the passing links and the same network without links (DeepSets) are tied on a team&rsquo;s
      non-penalty xG in a match: −0.002 MAE, 95% interval −0.008 to +0.005. The graph network does beat 19 hand-made network statistics (−0.024 MAE).
      A pre-registered second try with richer link features did not change this.</>,
  },
  {
    question: "What if a team had played someone else?",
    verdict: "A simulator that passes its checks",
    tone: "mixed",
    detail: <>Put a player&rsquo;s season profile into another team&rsquo;s matches and ten networks predict the new xG. Checks
      fixed in advance pass: a player swapped for himself changes almost nothing; a goalkeeper up front lowers xG for every
      team. It values creating chances, not finishing them.</>,
    link: { href: "/tactics/", label: "Try it on the tactics board" },
  },
  {
    question: "Can a sequence model tell what a player will be worth a year from now?",
    verdict: "Slightly better",
    tone: "mixed",
    detail: <>An LSTM and a Transformer reading the monthly history beat gradient boosting on summaries of it, by 1.6% of
      its error (Transformer − LightGBM −0.006 MAE, interval −0.008 to −0.004). A year ahead the typical prediction is
      still about 25% off.</>,
    link: { href: "/market-value/", label: "Look up a player" },
  },
  {
    question: "Do the words of a Turkish transfer rumour tell whether it comes true?",
    verdict: "No",
    tone: "no",
    detail: <>BERTurk reading the headline adds nothing measurable to what is known without reading it: +0.012 AUC, interval
      −0.005 to +0.034. How widely and how long a pair is reported does the work (AUC 0.77). A first run said yes; a leak
      was found, the record corrected and the study run again.</>,
    link: { href: "/rumours/", label: "See the 2026 rumours" },
  },
];

export default function Home() {
  return (
    <main className="home">
      <header className="home-head">
        <h1>Pro-Scout</h1>
        <p>Deep-learning models for football scouting, each tested against a simple baseline and reported as it came out,
          including when the baseline wins. The market-value and rumour studies, and a follow-up on the passing links, were registered before they were run. <Link href="/method/">How the studies were done</Link>.</p>
      </header>
      <ol className="findings">
        {FINDINGS.map((f) => (
          <li key={f.question} className={`finding ${f.tone}`}>
            <h2>{f.question}</h2>
            <p className="verdict">{f.verdict}</p>
            <p className="detail">{f.detail}</p>
            {f.link && <p><Link href={f.link.href} className="go">{f.link.label}</Link></p>}
          </li>
        ))}
      </ol>
    </main>
  );
}
