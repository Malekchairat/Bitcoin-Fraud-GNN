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

function TransactionGraph({ nodes, edges, modelPredictions }) {
  const canvasRef = useRef(null);
  const [nodeCount, setNodeCount] = useState(500);
  const [model, setModel] = useState("Unknown");
  const [selected, setSelected] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [rotation, setRotation] = useState({ x: -0.25, y: 0.55 });
  const pointers = useRef(new Map());
  const pinchRef = useRef(null);

  const visibleNodes = useMemo(() => nodes.slice(0, nodeCount), [nodes, nodeCount]);
  const nodeMap = useMemo(
    () => new Map(visibleNodes.map((node) => [Number(node.txId), node])),
    [visibleNodes]
  );
  const visibleEdges = useMemo(
    () =>
      edges
        .filter((edge) =>
          nodeMap.has(Number(edge.source)) && nodeMap.has(Number(edge.target))
        )
        .slice(0, 14000),
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

  const getPosition = (node, width, height) => {
    const id = Number(node.txId);
    const bx = ((id * 2654435761) % 1000003) / 1000003 - 0.5;
    const by = ((id * 2246822519) % 1000033) / 1000033 - 0.5;
    const bz = ((id * 3266489917) % 10000019) / 10000019 - 0.5;
    const x = bx * 520;
    const y = by * 520;
    const z = bz * 520;
    const cy = Math.cos(rotation.y);
    const sy = Math.sin(rotation.y);
    const cx = Math.cos(rotation.x);
    const sx = Math.sin(rotation.x);
    const rx = x * cy - z * sy;
    const rz = x * sy + z * cy;
    const ry = y * cx - rz * sx;
    const rz2 = y * sx + rz * cx;
    const scale = (720 / (820 + rz2)) * zoom;
    return {
      x: width / 2 + rx * scale,
      y: height / 2 + ry * scale,
      z: rz2,
      scale,
    };
  };

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    let frameId;

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
        canvas.width = Math.round(rect.width * dpr);
        canvas.height = Math.round(rect.height * dpr);
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      const width = rect.width;
      const height = rect.height;
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#091526";
      context.fillRect(0, 0, width, height);

      const cube = 300;
      const corners = [
        [-cube,-cube,-cube],[cube,-cube,-cube],[cube,cube,-cube],[-cube,cube,-cube],
        [-cube,-cube,cube],[cube,-cube,cube],[cube,cube,cube],[-cube,cube,cube],
      ];
      const projectedCorners = corners.map(([x,y,z]) => getPosition({txId: 1 + ((x+cube)/600)*1000 + ((y+cube)/600)*10000 + ((z+cube)/600)*100000}, width, height));
      context.strokeStyle = "rgba(45,212,191,.13)";
      context.lineWidth = 1;
      [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a,b]) => {
        context.beginPath();
        context.moveTo(projectedCorners[a].x, projectedCorners[a].y);
        context.lineTo(projectedCorners[b].x, projectedCorners[b].y);
        context.stroke();
      });

      const projectedNodes = new Map();
      visibleNodes.forEach((node) => {
        projectedNodes.set(Number(node.txId), getPosition(node, width, height));
      });

      context.lineWidth = 0.65;
      visibleEdges.forEach((edge) => {
        const source = projectedNodes.get(Number(edge.source));
        const target = projectedNodes.get(Number(edge.target));
        if (!source || !target) return;
        context.strokeStyle = "rgba(121,151,176,.18)";
        context.beginPath();
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.stroke();
      });

      [...visibleNodes]
        .map((node) => ({node, position: projectedNodes.get(Number(node.txId))}))
        .sort((a,b) => b.position.z - a.position.z)
        .forEach(({node, position}) => {
          const radius = Math.max(2.2, 4.5 * position.scale);
          context.beginPath();
          context.arc(position.x, position.y, selected?.txId === node.txId ? radius + 3 : radius, 0, Math.PI * 2);
          context.fillStyle = nodeColor(node);
          context.globalAlpha = 0.68 + Math.max(0, 1 - position.z / 900) * 0.32;
          context.fill();
          context.globalAlpha = 1;
          if (selected?.txId === node.txId) {
            context.strokeStyle = "#ffffff";
            context.lineWidth = 2;
            context.stroke();
          }
        });

      frameId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(frameId);
  }, [visibleNodes, visibleEdges, model, modelPredictions, selected, zoom, rotation]);

  const handlePointerDown = (event) => {
    canvasRef.current.setPointerCapture(event.pointerId);
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
  };

  const handlePointerMove = (event) => {
    if (!pointers.current.has(event.pointerId)) return;
    const previous = pointers.current.get(event.pointerId);
    pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.current.size === 1) {
      setRotation((value) => ({
        x: Math.max(-1.1, Math.min(1.1, value.x - (event.clientY - previous.y) * 0.006)),
        y: value.y + (event.clientX - previous.x) * 0.006,
      }));
    } else {
      const points = [...pointers.current.values()];
      const distance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
      if (pinchRef.current) {
        setZoom((value) => Math.max(0.55, Math.min(2.4, value * pinchRef.current / distance)));
      }
      pinchRef.current = distance;
    }
  };

  const handlePointerUp = (event) => {
    pointers.current.delete(event.pointerId);
    pinchRef.current = null;
  };

  const handleCanvasClick = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let nearest = null;
    let nearestDistance = 16;
    visibleNodes.forEach((node) => {
      const position = getPosition(node, rect.width, rect.height);
      const distance = Math.hypot(position.x - x, position.y - y);
      if (distance < nearestDistance) {
        nearest = node;
        nearestDistance = distance;
      }
    });
    setSelected(nearest);
  };

  const selectedPrediction =
    selected && model !== "Unknown"
      ? modelPredictions[String(selected.txId)]?.[model]
      : null;

  return (
    <Panel title="Network cube" subtitle="Rotate a real transaction network inside a spatial cube">
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
          <button className="icon-button" onClick={() => setZoom((value) => Math.min(2.4, value + 0.12))}>+</button>
          <button className="icon-button" onClick={() => setZoom((value) => Math.max(0.55, value - 0.12))}>−</button>
          <button className="icon-button wide" onClick={() => { setZoom(1); setRotation({x:-0.25,y:0.55}); }}>Reset</button>
        </div>
      </div>

      <div className="node-count">
        <div><b>Visible nodes</b><span>{visibleNodes.length.toLocaleString()}</span></div>
        <input
          type="range"
          min="100"
          max={Math.min(2000, nodes.length)}
          step="100"
          value={nodeCount}
          onChange={(event) => setNodeCount(Number(event.target.value))}
        />
      </div>

      <div className="network-cube-wrap">
        <canvas
          ref={canvasRef}
          className="network-cube"
          onWheel={(event) => {
            event.preventDefault();
            setZoom((value) => Math.max(0.55, Math.min(2.4, value + (event.deltaY > 0 ? -0.08 : 0.08))));
          }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onClick={handleCanvasClick}
        />
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
        <div><b>Scroll or pinch</b><span>Zoom through the cube</span></div>
        <div><b>Drag</b><span>Rotate the network</span></div>
        <div><b>Select a node</b><span>Inspect transaction details</span></div>
        <div><b>Choose a model</b><span>Reclassify unknown nodes only</span></div>
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
