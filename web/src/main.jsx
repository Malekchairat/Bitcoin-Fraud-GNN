import React, { useMemo, useRef, useState } from "react";
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
  const [metadata, setMetadata] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [modelPredictions, setModelPredictions] = useState({});

  React.useEffect(() => {
    Promise.all([
      fetch("/model-results.json").then((response) => response.json()),
      fetch("/model-results-meta.json")
        .then((response) => response.json())
        .catch(() => null),
      fetch("/graph-data.json")
        .then((response) => response.json())
        .catch(() => ({ nodes: [], edges: [] })),
      fetch("/model-predictions.json")
        .then((response) => response.json())
        .catch(() => ({})),
    ])
      .then(([rows, meta, graph, predictions]) => {
        setModelRows(rows);
        setMetadata(meta);
        setGraphData(graph);
        setModelPredictions(predictions);
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
          <TransactionGraph
            nodes={graphData.nodes}
            edges={graphData.edges}
            modelPredictions={modelPredictions}
            modelRows={modelRows}
          />
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
          {loaded && metadata
            ? `Real results · ${metadata.seed_count}-seed ${metadata.mode} run · ${metadata.row_count} model records`
            : "Source unavailable · no model records loaded"}
        </footer>
      </main>
    </div>
  );
}

function TransactionGraph({ nodes, edges, modelPredictions, modelRows }) {
  const [nodeCount, setNodeCount] = useState(500);
  const [model, setModel] = useState("Unknown");
  const [selected, setSelected] = useState(null);
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: 1000, h: 650 });
  const svgRef = useRef(null);
  const pointers = useRef(new Map());
  const lastPinch = useRef(null);

  const visibleNodes = useMemo(() => nodes.slice(0, nodeCount), [nodes, nodeCount]);
  const nodeMap = useMemo(
    () => new Map(visibleNodes.map((node) => [Number(node.txId), node])),
    [visibleNodes]
  );
  const visibleEdges = useMemo(
    () =>
      edges.filter(
        (edge) =>
          nodeMap.has(Number(edge.source)) && nodeMap.has(Number(edge.target))
      ).slice(0, 5000),
    [edges, nodeMap]
  );

  const displayLabel = (node) => {
    if (node.label !== "unknown" || model === "Unknown") return node.label;
    return modelPredictions[String(node.txId)]?.[model]?.label || "unknown";
  };

  const nodeColor = (node) => {
    const label = displayLabel(node);
    if (label === "illicit") return "#ef4444";
    if (label === "licit") return "#2dd4bf";
    return "#64748b";
  };

  const zoom = (factor, pointX = 500, pointY = 325) => {
    setViewBox((box) => {
      const nextW = Math.max(180, Math.min(1000, box.w * factor));
      const nextH = nextW * 0.65;
      const ratioX = (pointX - box.x) / box.w;
      const ratioY = (pointY - box.y) / box.h;
      return {
        x: pointX - nextW * ratioX,
        y: pointY - nextH * ratioY,
        w: nextW,
        h: nextH,
      };
    });
  };

  const handleWheel = (event) => {
    event.preventDefault();
    const rect = svgRef.current.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * viewBox.w + viewBox.x;
    const y = ((event.clientY - rect.top) / rect.height) * viewBox.h + viewBox.y;
    zoom(event.deltaY > 0 ? 1.12 : 0.88, x, y);
  };

  const clientToSvg = (event) => {
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * viewBox.w + viewBox.x,
      y: ((event.clientY - rect.top) / rect.height) * viewBox.h + viewBox.y,
    };
  };

  const handlePointerDown = (event) => {
    svgRef.current.setPointerCapture(event.pointerId);
    pointers.current.set(event.pointerId, clientToSvg(event));
  };

  const handlePointerMove = (event) => {
    if (!pointers.current.has(event.pointerId)) return;
    const previous = pointers.current.get(event.pointerId);
    const current = clientToSvg(event);
    pointers.current.set(event.pointerId, current);

    if (pointers.current.size === 1) {
      setViewBox((box) => ({
        ...box,
        x: box.x - (current.x - previous.x),
        y: box.y - (current.y - previous.y),
      }));
      lastPinch.current = null;
      return;
    }

    const points = [...pointers.current.values()];
    const distance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
    if (lastPinch.current) {
      const factor = lastPinch.current / distance;
      const centerX = (points[0].x + points[1].x) / 2;
      const centerY = (points[0].y + points[1].y) / 2;
      zoom(factor, centerX, centerY);
    }
    lastPinch.current = distance;
  };

  const handlePointerUp = (event) => {
    pointers.current.delete(event.pointerId);
    lastPinch.current = null;
  };

  const resetView = () => setViewBox({ x: 0, y: 0, w: 1000, h: 650 });

  const selectedPrediction =
    selected && model !== "Unknown"
      ? modelPredictions[String(selected.txId)]?.[model]
      : null;

  return (
    <Panel title="Interactive transaction graph" subtitle="Real nodes and edges from the Elliptic dataset">
      <div className="graph-toolbar">
        <div className="model-switcher" aria-label="Choose model classification">
          {["Unknown", "MLP", "GraphSAGE", "GAT"].map((name) => (
            <button
              key={name}
              className={model === name ? "model-button active" : "model-button"}
              onClick={() => setModel(name)}
            >
              {name === "Unknown" ? "Original labels" : name}
            </button>
          ))}
        </div>
        <div className="graph-actions">
          <button className="icon-button" onClick={() => zoom(0.82)}>+</button>
          <button className="icon-button" onClick={() => zoom(1.22)}>−</button>
          <button className="icon-button wide" onClick={resetView}>Reset</button>
        </div>
      </div>

      <div className="node-count">
        <div>
          <b>Visible nodes</b>
          <span>{visibleNodes.length.toLocaleString()}</span>
        </div>
        <input
          type="range"
          min="100"
          max={Math.min(2000, nodes.length)}
          step="100"
          value={nodeCount}
          onChange={(event) => setNodeCount(Number(event.target.value))}
        />
      </div>

      <div className="interactive-svg-wrap">
        <svg
          ref={svgRef}
          className="interactive-graph"
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`}
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        >
          <rect x="-100" y="-100" width="1200" height="850" fill="#091526" />
          {visibleEdges.map((edge, index) => {
            const source = nodeMap.get(Number(edge.source));
            const target = nodeMap.get(Number(edge.target));
            if (!source || !target) return null;
            return (
              <line
                key={`${edge.source}-${edge.target}-${index}`}
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                className="graph-edge"
              />
            );
          })}
          {visibleNodes.map((node) => (
            <circle
              key={node.txId}
              cx={node.x}
              cy={node.y}
              r={selected?.txId === node.txId ? 8 : 4.2}
              fill={nodeColor(node)}
              stroke={selected?.txId === node.txId ? "#ffffff" : "none"}
              strokeWidth={selected?.txId === node.txId ? 2 : 0}
              className="graph-node"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                setSelected(node);
              }}
            >
              <title>
                {node.txId} · {displayLabel(node)} · time step {node.timeStep}
              </title>
            </circle>
          ))}
        </svg>

        {selected && (
          <div className="node-inspector">
            <button onClick={() => setSelected(null)} aria-label="Close node details">×</button>
            <span>Selected transaction</span>
            <strong>{selected.txId}</strong>
            <dl>
              <div><dt>Time step</dt><dd>{selected.timeStep}</dd></div>
              <div><dt>Source label</dt><dd>{selected.label}</dd></div>
              {selectedPrediction && (
                <>
                  <div><dt>{model} prediction</dt><dd>{selectedPrediction.label}</dd></div>
                  <div><dt>Illicit probability</dt><dd>{(selectedPrediction.probability * 100).toFixed(1)}%</dd></div>
                </>
              )}
            </dl>
          </div>
        )}
      </div>

      <div className="legend-row">
        <Legend color="#ef4444" label="Illicit" />
        <Legend color="#2dd4bf" label="Licit" />
        <Legend color="#64748b" label={model === "Unknown" ? "Unknown" : `${model} unknown predictions`} />
      </div>

      <div className="graph-explanation">
        <div>
          <b>Scroll or pinch</b>
          <span>Zoom into clusters</span>
        </div>
        <div>
          <b>Drag</b>
          <span>Move around the graph</span>
        </div>
        <div>
          <b>Select a node</b>
          <span>Inspect its transaction details</span>
        </div>
        <div>
          <b>Choose a model</b>
          <span>Reclassify unknown nodes only</span>
        </div>
      </div>
    </Panel>
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
