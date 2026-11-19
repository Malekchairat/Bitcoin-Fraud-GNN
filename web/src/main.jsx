import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const metricLabels = {
  f1: "Illicit F1",
  precision: "Precision",
  recall: "Recall",
  roc_auc: "ROC-AUC",
  pr_auc: "PR-AUC",
};

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function App() {
  const [active, setActive] = useState("Overview");
  const [modelRows, setModelRows] = useState([]);
  const [loaded, setLoaded] = useState(false);

  React.useEffect(() => {
    fetch("/model-results.json")
      .then((response) => response.json())
      .then((rows) => {
        setModelRows(rows);
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
  }, []);

  const summary = useMemo(() => {
    const names = ["MLP", "GraphSAGE", "GAT"];
    return names.map((name) => {
      const rows = modelRows.filter((row) => row.model === name);
      const metrics = {};
      Object.keys(metricLabels).forEach((key) => {
        const values = rows.map((row) => Number(row[key])).filter(Number.isFinite);
        const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
        metrics[key] = mean;
      });
      return { name, ...metrics };
    });
  }, [modelRows]);

  const best = summary.reduce(
    (winner, row) => (!winner || row.f1 > winner.f1 ? row : winner),
    null
  );

  const sections = [
    ["Overview", "01"],
    ["Dataset", "02"],
    ["Graph", "03"],
    ["Models", "04"],
  ];

  return (
    <div className="shell">
      <aside className="sidebar">
        <div>
          <div className="brand-mark">₿</div>
          <div className="brand-name">ELLIPTIC <span>LAB</span></div>
        </div>
        <nav aria-label="Dashboard sections">
          {sections.map(([label, number]) => (
            <button
              key={label}
              className={active === label ? "nav-item active" : "nav-item"}
              onClick={() => setActive(label)}
            >
              <span>{number}</span>
              {label}
            </button>
          ))}
        </nav>
        <div className="experiment-status">
          <span className="status-dot" />
          <div>
            <b>Normal experiment</b>
            <small>5 seeds · 200 epochs</small>
          </div>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div>
            <span className="eyebrow">Bitcoin transaction intelligence</span>
            <h1>{active}</h1>
          </div>
          <div className="live-label">Elliptic dataset · 49 time steps</div>
        </header>

        {active === "Overview" && (
          <>
            <section className="hero-card">
              <div>
                <span className="hero-kicker">Model study complete</span>
                <h2>Discover illicit activity through graph relationships.</h2>
                <p>
                  A temporal comparison of MLP, GraphSAGE and GAT on 203,769
                  Bitcoin transactions.
                </p>
              </div>
              <div className="hero-score">
                <span>Best mean F1</span>
                <strong>{best ? best.f1.toFixed(3) : "—"}</strong>
                <b>{best ? best.name : ""}</b>
              </div>
            </section>

            <section className="stat-grid">
              <Stat label="Transactions" value="203,769" note="49 time steps" />
              <Stat label="Graph edges" value="234,355" note="Undirected: 468,710" />
              <Stat label="Labelled nodes" value="46,564" note="22.85% of total" />
              <Stat label="Illicit nodes" value="4,545" note="9.76% of labelled" />
            </section>

            <section className="grid-two">
              <Panel title="Temporal integrity" subtitle="Chronological leakage-safe split">
                <div className="timeline">
                  <TimelineStep name="Training" range="Steps 1–29" value="26,381" width="62%" />
                  <TimelineStep name="Validation" range="Steps 30–34" value="3,513" width="14%" />
                  <TimelineStep name="Test" range="Steps 35–49" value="16,670" width="38%" />
                </div>
              </Panel>
              <Panel title="Model ranking" subtitle="Mean illicit F1 across five seeds">
                <ModelBars rows={summary} />
              </Panel>
            </section>
          </>
        )}

        {active === "Dataset" && (
          <section className="grid-two">
            <Panel title="Label distribution" subtitle="Every transaction, classified or unknown">
              <Distribution />
            </Panel>
            <Panel title="Why accuracy is insufficient" subtitle="The illicit class is the minority">
              <Insight>
                <b>77.15%</b>
                <span>of transactions are unknown</span>
              </Insight>
              <Insight>
                <b>9.76%</b>
                <span>of labelled transactions are illicit</span>
              </Insight>
            </Panel>
          </section>
        )}

        {active === "Graph" && (
          <Panel title="Transaction graph" subtitle="Elliptic node-and-edge topology">
            <div className="graph-placeholder">
              <div className="graph-orb orb-a" />
              <div className="graph-orb orb-b" />
              <div className="graph-orb orb-c" />
              <div className="graph-orb orb-d" />
              <div className="graph-lines" />
              <div className="graph-caption">
                <span>234,355</span>
                <small>connections</small>
              </div>
            </div>
            <div className="legend-row">
              <Legend color="#ef4444" label="Illicit" />
              <Legend color="#2dd4bf" label="Licit" />
              <Legend color="#64748b" label="Unknown" />
            </div>
          </Panel>
        )}

        {active === "Models" && (
          <>
            <section className="grid-three">
              {summary.map((row) => (
                <Panel key={row.name} title={row.name} subtitle="Normal-mode mean">
                  <div className="model-number">{row.f1.toFixed(3)}</div>
                  <small>Mean illicit F1</small>
                  <MetricLine label="Precision" value={row.precision} />
                  <MetricLine label="Recall" value={row.recall} />
                  <MetricLine label="PR-AUC" value={row.pr_auc} />
                </Panel>
              ))}
            </section>
            <Panel title="Metric definitions" subtitle="How to interpret the evaluation">
              <div className="definitions">
                <Definition term="F1" text="Balance between precision and recall for illicit transactions." />
                <Definition term="Recall" text="Share of actual illicit transactions detected." />
                <Definition term="PR-AUC" text="Ranking quality under severe class imbalance." />
              </div>
            </Panel>
          </>
        )}

        <footer>
          {!loaded && "Loading model evidence…"}
          {loaded && "Source: Elliptic Bitcoin Transaction Dataset · 5-seed normal run"}
        </footer>
      </main>
    </div>
  );
}

function Stat({ label, value, note }) {
  return (
    <article className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  );
}

function Panel({ title, subtitle, children }) {
  return (
    <article className="panel">
      <header>
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </header>
      {children}
    </article>
  );
}

function TimelineStep({ name, range, value, width }) {
  return (
    <div className="timeline-row">
      <div className="timeline-label"><b>{name}</b><span>{range}</span></div>
      <div className="timeline-track"><div style={{ width }} /></div>
      <strong>{value}</strong>
    </div>
  );
}

function ModelBars({ rows }) {
  return (
    <div className="model-bars">
      {rows.map((row) => (
        <div className="bar-row" key={row.name}>
          <span>{row.name}</span>
          <div><i style={{ width: `${row.f1 * 100}%` }} /></div>
          <strong>{row.f1.toFixed(3)}</strong>
        </div>
      ))}
    </div>
  );
}

function Distribution() {
  return (
    <div className="distribution">
      <div className="distribution-bar">
        <i className="unknown" /><i className="licit" /><i className="illicit" />
      </div>
      <DistributionRow color="#64748b" label="Unknown" value="157,205" percent="77.15%" />
      <DistributionRow color="#2dd4bf" label="Licit" value="42,019" percent="20.62%" />
      <DistributionRow color="#ef4444" label="Illicit" value="4,545" percent="2.23%" />
    </div>
  );
}

function DistributionRow({ color, label, value, percent }) {
  return (
    <div className="distribution-row">
      <span className="legend-dot" style={{ background: color }} />
      <b>{label}</b><strong>{value}</strong><small>{percent}</small>
    </div>
  );
}

function Insight({ children }) {
  return <div className="insight">{children}</div>;
}

function Legend({ color, label }) {
  return <div className="legend"><span style={{ background: color }} />{label}</div>;
}

function MetricLine({ label, value }) {
  return (
    <div className="metric-line">
      <span>{label}</span>
      <div><i style={{ width: `${value * 100}%` }} /></div>
      <strong>{value.toFixed(3)}</strong>
    </div>
  );
}

function Definition({ term, text }) {
  return <article><b>{term}</b><p>{text}</p></article>;
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode><App /></React.StrictMode>
);
