import { useState } from 'react';
import './index.css';

function App() {
  const [feed, setFeed] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [epochs, setEpochs] = useState(50);

  const safeCount = feed.filter(t => t.result === 'SAFE').length;
  const fraudCount = feed.filter(t => t.result === 'FRAUD').length;

  const createTransaction = (type) => {
    let amount, time, features, merchant;
    time = Math.floor(Math.random() * 100000);

    if (type === 'fraud') {
      amount = (Math.random() * 2500 + 500).toFixed(2);
      merchant = 'Transferencia Offshore (Desconocida)';
      const realFraudPattern = [
        -2.31, 1.95, -1.61, 3.99, -0.52, -1.42, -2.53, 1.39, -2.77, -2.77,
         3.20, -2.89, -0.59, -4.28,  0.38, -1.14, -2.83, -0.01,  0.41,  0.12,
         0.51, -0.03, -0.46,  0.32,  0.04,  0.17,  0.26, -0.14,
      ];
      features = Array(28).fill(0).map((_, i) =>
        (realFraudPattern[i] + (Math.random() * 0.5 - 0.25)).toFixed(3)
      );
    } else if (type === 'borderline') {
      amount = (200 + Math.random() * 50).toFixed(2);
      merchant = 'Compra Internacional Inusual';
      const borderlinePattern = [
        1.8, -1.9, 1.5, 1.9, -1.5, 1.4, -1.6, 1.2, -1.8, 1.4,
       -1.5,  1.8, -1.3, 1.2,  1.7, -1.1, 1.4,  1.2,  1.3, -1.4,
        1.2, -1.1,  1.5, 1.3, -1.2,  1.4, -1.3,  1.1,
      ];
      features = Array(28).fill(0).map((_, i) =>
        (borderlinePattern[i] + (Math.random() * 0.1 - 0.05)).toFixed(3)
      );
    } else {
      amount = (Math.random() * 50 + 5).toFixed(2);
      merchant = ['Amazon', 'Netflix', 'Starbucks', 'Uber', 'Supermercado'][Math.floor(Math.random() * 5)];
      features = Array(28).fill(0).map(() => (Math.random() * 0.05 - 0.025).toFixed(3));
    }

    return {
      id: 'TRX-' + Math.floor(100000 + Math.random() * 900000),
      amount,
      time,
      merchant,
      raw_features: [time, ...features, amount],
    };
  };

  const analyzeSingle = async (trx) => {
    try {
      const response = await fetch('http://localhost:3001/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: trx.raw_features, epochs }),
      });
      const data = await response.json();
      return data.prediction === 1 ? 'FRAUD' : 'SAFE';
    } catch {
      return 'ERROR';
    }
  };

  const handleManualScan = async (type) => {
    setAnalyzing(true);
    const trx = createTransaction(type);
    setTimeout(async () => {
      const result = await analyzeSingle(trx);
      trx.result = result;
      setFeed(prev => [trx, ...prev].slice(0, 10));
      setAnalyzing(false);
    }, 800);
  };

  return (
    <div className="app-layout">

      {/* ── SIDEBAR ───────────────────────────────────── */}
      <aside className="sidebar">

        <div className="brand">
          <div className="brand-name">FraudNet</div>
          <div className="brand-sub">Intelligence · v2.0</div>
        </div>

        <div className="status-row">
          <div className="pulse-dot" />
          <span className="status-label">Neural Engine</span>
          <span className="status-value">ONLINE</span>
        </div>

        <div className="control-panel">
          <div className="section-label">Simulation Engine</div>
          <p className="control-desc" style={{ marginTop: '0.75rem' }}>
            Inject transaction vectors into the backend neural network for real-time tensor evaluation.
          </p>

          <div className="select-group">
            <label className="field-label">Model weights</label>
            <select
              className="styled-select"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
            >
              <option value={50}>Fast Check — 50 Epochs</option>
              <option value={100}>Standard — 100 Epochs</option>
              <option value={200}>Deep Analysis — 200 Epochs</option>
            </select>
          </div>

          <div className="btn-stack">
            <button className="scan-btn btn-normal" onClick={() => handleManualScan('normal')} disabled={analyzing}>
              <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="9"/><path d="M9 12l2 2 4-4"/>
              </svg>
              Standard Transaction
            </button>
            <button className="scan-btn btn-warn" onClick={() => handleManualScan('borderline')} disabled={analyzing}>
              <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
              </svg>
              Borderline Vector
            </button>
            <button className="scan-btn btn-danger" onClick={() => handleManualScan('fraud')} disabled={analyzing}>
              <svg className="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              Inject Fraud Signature
            </button>
          </div>
        </div>

        <div className="tech-stack">
          <div className="section-label" style={{ marginBottom: '0.75rem' }}>System Architecture</div>
          {[
            ['Frontend', 'React · Vite'],
            ['API', 'Node.js'],
            ['Inference', 'C++ MLP'],
          ].map(([k, v]) => (
            <div className="tech-row" key={k}>
              <span className="tech-key">{k}</span>
              <span className="tech-val">{v}</span>
            </div>
          ))}
        </div>

      </aside>

      {/* ── MAIN PANEL ────────────────────────────────── */}
      <main className="main-content">

        <div className="top-bar">
          <div>
            <div className="page-title">Global Monitoring</div>
            <div className="page-subtitle">Real-time neural network evaluation · MLP inference</div>
          </div>

          <div className="stats-row">
            <div className="stat-chip">
              <span className="stat-num">{feed.length}</span>
              <span className="stat-label">Total</span>
            </div>
            <div className="stat-chip">
              <span className="stat-num safe-color">{safeCount}</span>
              <span className="stat-label">Approved</span>
            </div>
            <div className="stat-chip">
              <span className="stat-num fraud-color">{fraudCount}</span>
              <span className="stat-label">Flagged</span>
            </div>
          </div>
        </div>

        <div className="feed-wrapper">
          <div className="feed-container">

            {analyzing && (
              <div className="scanning-card">
                <div className="scan-ring" />
                <div className="scan-info">
                  <strong>Processing vector…</strong>
                  <span>Propagating through 30 hidden dense layers</span>
                </div>
              </div>
            )}

            {feed.length === 0 && !analyzing && (
              <div className="empty-state">
                <svg className="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                  <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>
                </svg>
                <p>AWAITING TRANSACTION INJECTION</p>
              </div>
            )}

            {feed.map((trx, idx) => (
              <div
                key={idx}
                className={`trx-card ${trx.result === 'FRAUD' ? 'card-fraud' : trx.result === 'SAFE' ? 'card-safe' : 'card-error'}`}
              >
                <div className="trx-header">
                  <span className="trx-id">{trx.id}</span>
                  <span className={`badge ${trx.result}`}>
                    <span className="badge-dot" />
                    {trx.result === 'FRAUD' ? 'Fraud Detected' :
                     trx.result === 'SAFE' ? 'Approved' : 'Network Error'}
                  </span>
                </div>

                <div className="trx-body">
                  <div className="info-block">
                    <span className="info-label">Destination</span>
                    <span className="info-value">{trx.merchant}</span>
                  </div>
                  <div className="info-block">
                    <span className="info-label">Amount</span>
                    <span className="info-value amount-value">${trx.amount}</span>
                  </div>
                </div>

                <div className="trx-footer">
                  [ T:{trx.time} · V1:{trx.raw_features[1]} · V2:{trx.raw_features[2]} · V3:{trx.raw_features[3]} … ]
                </div>
              </div>
            ))}

          </div>
        </div>
      </main>

    </div>
  );
}

export default App;
