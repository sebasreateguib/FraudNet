import { useState } from 'react';
import './index.css';

function App() {
  const [feed, setFeed] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);

  const [epochs, setEpochs] = useState(50);
  
  // Genera una transacción falsa pero con estructura matemática realista para engañar/probar a la red
  const createTransaction = (type) => {
    let amount, time, features, merchant;
    time = Math.floor(Math.random() * 100000);

    if (type === 'fraud') {
        amount = (Math.random() * 2500 + 500).toFixed(2);
        merchant = "Transferencia Offshore (Desconocida)";
        const realFraudPattern = [
          -2.31, 1.95, -1.61, 3.99, -0.52, -1.42, -2.53, 1.39, -2.77, -2.77,
          3.20, -2.89, -0.59, -4.28, 0.38, -1.14, -2.83, -0.01, 0.41, 0.12,
          0.51, -0.03, -0.46, 0.32, 0.04, 0.17, 0.26, -0.14
        ];
        features = Array(28).fill(0).map((_, i) => (realFraudPattern[i] + (Math.random() * 0.5 - 0.25)).toFixed(3));
    } else if (type === 'borderline') {
        // Esta transacción engaña al modelo de 50 epochs (Falso Positivo) pero el de 200 epochs sabe que es legal.
        amount = (200 + Math.random() * 50).toFixed(2);
        merchant = "Compra Internacional Inusual";
        const borderlinePattern = [
          1.8, -1.9, 1.5, 1.9, -1.5, 1.4, -1.6, 1.2, -1.8, 1.4, -1.5, 1.8, -1.3, 1.2, 
          1.7, -1.1, 1.4, 1.2, 1.3, -1.4, 1.2, -1.1, 1.5, 1.3, -1.2, 1.4, -1.3, 1.1
        ];
        features = Array(28).fill(0).map((_, i) => (borderlinePattern[i] + (Math.random() * 0.1 - 0.05)).toFixed(3));
    } else {
        amount = (Math.random() * 50 + 5).toFixed(2);
        merchant = ["Amazon", "Netflix", "Starbucks", "Uber", "Supermercado"][Math.floor(Math.random()*5)];
        features = Array(28).fill(0).map(() => (Math.random() * 0.05 - 0.025).toFixed(3));
    }

    return {
      id: "TRX-" + Math.floor(100000 + Math.random() * 900000),
      amount,
      time,
      merchant,
      raw_features: [time, ...features, amount]
    };
  };

  const analyzeSingle = async (trx) => {
    try {
      const response = await fetch('http://localhost:3001/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: trx.raw_features, epochs })
      });
      const data = await response.json();
      return data.prediction === 1 ? 'FRAUD' : 'SAFE';
    } catch (e) {
      console.error(e);
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
       {/* PANEL LATERAL - CONTROLES */}
       <div className="sidebar">
          <h2>
             <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
             FraudNet Intelligence
          </h2>
          
          <div className="status-indicator">
            <div className="dot pulse"></div>
            <span>System Active</span>
          </div>
          
          <div className="control-panel">
             <h3>Simulation Engine</h3>
             <p>Inject transaction vectors into the backend. The neural network will evaluate the tensor parameters in real-time.</p>
             
             <div className="select-wrapper">
                <label>Core Weights Configuration</label>
                <select 
                   value={epochs} 
                   onChange={(e) => setEpochs(Number(e.target.value))}
                >
                   <option value={50}>Fast Check (50 Epochs)</option>
                   <option value={100}>Standard (100 Epochs)</option>
                   <option value={200}>Deep Analysis (200 Epochs)</option>
                </select>
             </div>

             <button className="btn-normal" onClick={() => handleManualScan('normal')} disabled={analyzing}>
                Evaluate Standard Transaction
             </button>
             <button className="btn-secondary" onClick={() => handleManualScan('borderline')} disabled={analyzing}>
                Evaluate Borderline Vector
             </button>
             <button className="btn-fraud" onClick={() => handleManualScan('fraud')} disabled={analyzing}>
                Inject Fraudulent Signature
             </button>
          </div>

          <div className="tech-stack">
             <b>System Architecture</b>
             Frontend: React (Vite)<br/>
             Backend API: Node.js<br/>
             Inference: Custom C++ MLP
          </div>
       </div>

       {/* PANEL PRINCIPAL - FEED DE TRANSACCIONES */}
       <div className="main-content">
          <div className="header">
             <h1>Global Monitoring</h1>
             <p>Real-time neural network evaluation</p>
          </div>

          <div className="feed-container">
             {analyzing && (
               <div className="scanning-card">
                  <div className="spinner"></div>
                  <div className="scan-text">
                    <strong>Processing vector...</strong>
                    <span>Propagating 30 hidden tensors through dense layers</span>
                  </div>
               </div>
             )}
             
             {feed.length === 0 && !analyzing && (
                <div className="empty-state">
                   Awaiting transaction injection...
                </div>
             )}

             {feed.map((trx, idx) => (
                <div key={idx} className={`trx-card ${trx.result === 'FRAUD' ? 'card-fraud' : trx.result === 'SAFE' ? 'card-safe' : 'card-error'}`}>
                   <div className="trx-header">
                      <span className="trx-id">{trx.id}</span>
                      <span className={`badge ${trx.result}`}>
                        {trx.result === 'FRAUD' ? 'Fraud Detected' : 
                         trx.result === 'SAFE' ? 'Transaction Approved' : 'Network Error'}
                      </span>
                   </div>
                   
                   <div className="trx-body">
                      <div className="info">
                         <span className="label">Destination</span>
                         <span className="value">{trx.merchant}</span>
                      </div>
                      <div className="info">
                         <span className="label">Requested Amount</span>
                         <span className="value amount">${trx.amount}</span>
                      </div>
                   </div>

                   <div className="trx-footer">
                      Vector parameters: [ Time: {trx.time}, V1: {trx.raw_features[1]}, V2: {trx.raw_features[2]} ... ]
                   </div>
                </div>
             ))}
          </div>
       </div>
    </div>
  )
}

export default App;
