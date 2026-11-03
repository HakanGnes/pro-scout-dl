import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "Method" };

const GH = "https://github.com/HakanGnes/pro-scout-dl/blob/main";
const Doc = ({ path, children }: { path: string; children: React.ReactNode }) => <a href={`${GH}/${path}`}>{children}</a>;

const SECTIONS = [
  { id: "rules", title: "How every study is set up" },
  { id: "xg", title: "Expected goals and passing networks" },
  { id: "swap", title: "The player-swap simulator" },
  { id: "value", title: "Market value a year ahead" },
  { id: "rumours", title: "Transfer rumours" },
  { id: "leak", title: "A leak, and how the record was corrected" },
  { id: "site", title: "What this site runs" },
];

export default function Method() {
  return (
    <main className="method">
      <header className="page-head">
        <h1>Method</h1>
        <p>How the models behind this site were built and tested, what each test found, and what went wrong along the way.
          The code, the registrations and the full reports are in the <a href="https://github.com/HakanGnes/pro-scout-dl">repository</a>.</p>
      </header>

      <div className="method-body">
        <nav className="toc" aria-label="On this page">
          <ul>{SECTIONS.map((s) => <li key={s.id}><a href={`#${s.id}`}>{s.title}</a></li>)}</ul>
        </nav>

        <article className="prose">
          <section id="rules">
            <h2>How every study is set up</h2>
            <p><strong>Against a simple baseline.</strong> Each deep model is compared with something plain: the training mean, a
              value that stays the same, a median by age and position, gradient boosting on hand-made features. A deep model is
              only worth its cost if it beats them, and the result is reported when it does not.</p>
            <p><strong>Registered before the run.</strong> For the market-value and rumour studies, and a follow-up on the passing
              links, the question, the data, the models, the test and the bar for success were written down and committed before
              any model was trained. The commit is the timestamp, and each registration was run once.</p>
            <p><strong>Only what was known on the day.</strong> An input dated after the moment of prediction would leak the answer.
              Player values, clubs and news counts are taken from records dated before the prediction, and fields that describe a
              later state (a player&rsquo;s current club, his peak value) are not used for older dates.</p>
            <p><strong>Split by time.</strong> Models are fitted on earlier data and tested on later data (later match weeks, years
              or seasons), never on a random shuffle, so the test looks like using the model on games it has not seen.</p>
            <p><strong>Intervals that respect what repeats.</strong> A player appears at many dates and a rumour pair in many
              headlines, so 95% intervals come from resampling players or player–club pairs, not single rows. A difference counts
              only if its interval stays clear of zero.</p>
          </section>

          <section id="xg">
            <h2>Expected goals and passing networks</h2>
            <p>The Wyscout public dataset has every event of the 2017/18 season in five leagues, plus the 2018 World Cup and Euro
              2016, but no expected goals. A model of its own, fitted on 45,869 shots with only what is known before each shot
              (location, body part, how the chance was made), gives each team a non-penalty xG per match.</p>
            <p>Each team in each match then becomes a graph: players are nodes with fifteen features (role, minutes, passes,
              accuracy, median position, entries into the final third and the box, progressive passes, touches), completed passes
              are the edges. Shots are left out of every feature, since the target is built from them.</p>
            <p>A graph network (GINE) was compared with the same network without edges (DeepSets) and with gradient boosting on 19
              network statistics. The graph network beats the statistics (−0.024 MAE, interval −0.037 to −0.012) but ties DeepSets
              (−0.002, interval −0.008 to +0.005). A registered second try with six edge features instead of two gave the same
              answer. For a team&rsquo;s xG in a match, who passes to whom adds nothing measurable beyond each player&rsquo;s own
              passing profile. A synthetic test confirms the pipeline does find edge signal when there is some.</p>
          </section>

          <section id="swap">
            <h2>The player-swap simulator</h2>
            <p>Because the links add nothing, the simulator swaps players, not links. A player&rsquo;s league season is pooled into
              rates per full match; in every match the replaced player played, his node gets the newcomer&rsquo;s rates scaled to the
              same minutes, and all ten saved networks (five DeepSets, five GNN) predict the team&rsquo;s xG before and after.</p>
            <p>Three checks were fixed before running it, and all pass. A player swapped for himself changes the prediction by
              0.011 npxG per match on average over 294 players (the limit was 0.05). A forward replaced by a goalkeeper lowers xG
              for all 98 teams. Ranking 160 forwards to replace Messi follows how often each gets the ball into the box
              (Spearman +0.94).</p>
            <p>The checks also show the limit: shots are not inputs, so the simulator values creating chances, not finishing them.
              Replacing Girona&rsquo;s Stuani with a goalkeeper costs only 0.005 npxG per match. A newcomer&rsquo;s profile comes from
              his own team, so a result is a simulation under the models&rsquo; assumptions, not a forecast.</p>
          </section>

          <section id="value">
            <h2>Market value a year ahead</h2>
            <p>From the Transfermarkt open data: for players in 14 European first divisions on 24 dates (1 January and 1 July,
              2013–2025), predict the value Transfermarkt gives a year later. 136,942 samples; train 2013–2020, validation
              2021–22, test 2023–25. Every input is dated before the prediction. <Doc path="docs/preregistration/2026-10-27-market-value.md">Registration</Doc>.</p>
            <p>The registered question was whether a Transformer reading the raw 36-month history beats LightGBM on hand-made
              summaries. It does, by little: −0.006 MAE on the log scale (interval −0.008 to −0.004), 1.6% of LightGBM&rsquo;s error,
              and ahead at all five test dates. An LSTM is slightly better still. Most of what can be predicted is simple: a
              median by age and position removes 0.047 of persistence&rsquo;s 0.426 error, the best model 0.062. A year ahead the
              typical prediction is still about 25% off; for players aged 21 or under every model is far worse.</p>
            <p>Players with no value a year later are dropped, so the study says nothing about players who leave the data.
              Neither model was tuned; a tuned LightGBM could narrow the gap, and testing that would need its own registration.</p>
          </section>

          <section id="rumours">
            <h2>Transfer rumours</h2>
            <p>No labelled set of Turkish transfer rumours exists, so one was built. The Internet Archive&rsquo;s index of eight
              Turkish sports sites gave 286,132 transfer articles from 2018 to 2026; their publication dates come from the archived
              pages. Each headline is linked to Transfermarkt players and clubs; a claim is a headline, a player and a club he is not
              at. Rules decide whether it is a rumour, an announcement or something else, and the label is whether Transfermarkt
              records the move within 180 days. Names are masked, so a model learns how papers write, not which names came true.</p>
            <p>Before the run, 400 headlines were labelled by hand: 163 to write the rules, then 237 fresh ones after the rules were
              committed. On those the rumour rule is right 86% of the time, the player is right 93% of the time, and about two claims
              in three are clean rumours about the right player and target club. The rest is noise that lowers every model alike.</p>
            <p>The registered question: does BERTurk reading the headline add to a model that does not read it (which paper, how
              much and how widely the pair was in the news, the player&rsquo;s value and age)? Test year 2025, 2,955 rumours.
              It does not: +0.012 AUC, interval −0.005 to +0.034. The model without the text scores 0.770; BERTurk alone 0.668,
              better than a bag of words (+0.024, interval above zero), but that matters only if the wording helps at all. As
              registered, the score in use is the model that does not read the text. <Doc path="docs/preregistration/2026-11-02-rumour-credibility-corrected.md">Registration</Doc>.</p>
          </section>

          <section id="leak">
            <h2>A leak, and how the record was corrected</h2>
            <p>The first run of the rumour study said yes: the text added +0.015 AUC with an interval above zero. Then a list of
              scored headlines showed rows that were plainly wrong, and the linking rules were taken apart. One rule linked a
              headline that named only a last name to a player who had moved to or from a named club within a year, before or
              <em> after</em> the headline. A later move could therefore decide whether a headline entered the study, and those
              claims came true by construction.</p>
            <p>Leaving those claims out of the first run&rsquo;s own test predictions removed its result (+0.010, interval −0.006 to
              +0.028). The first registration was marked as not standing, a corrected study was registered with links decided only
              by records dated before each headline, and it was run once: not supported. A new rule that would have linked a lone
              last name to the most valuable player carrying it was tested by hand on 80 headlines before the run and dropped at
              65% right, against the 85% required. Both runs stay in the <Doc path="reports/rumours/README.md">report</Doc>.</p>
            <p>The same leak had flattered an earlier check on 2026 rumours: 0.75 there came partly from moves between January
              and June adding headlines to a pair&rsquo;s news counts. On the 650 rumours of January to April 2026 whose 180 days
              have passed, the corrected model scores 0.675 (interval 0.577 to 0.771), and Transfermarkt&rsquo;s data ends in June,
              after which only moves the papers report done are seen.</p>
          </section>

          <section id="site">
            <h2>What this site runs</h2>
            <p><Link href="/tactics/">The tactics board</Link> runs the ten passing-network models in your browser. They are small
              enough to write out in TypeScript, and a test checks each network&rsquo;s prediction and every swap result against
              PyTorch to five decimals. Best replacements are ranked in advance in Python, because a ranking runs every candidate
              through every network in every match.</p>
            <p><Link href="/rumours/">The rumours page</Link> lists the 2026 rumours scored by the model in use, by player–club pair.
              Headlines are not shown: they are the papers&rsquo; text. <Link href="/market-value/">The market-value page</Link> shows
              the registered run&rsquo;s test predictions as they were made.</p>
          </section>
        </article>
      </div>
    </main>
  );
}
