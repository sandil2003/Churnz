import React, { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

const PRESETS = {
  highRisk: {
    customer_id: "CUST-HIGH",
    gender: "Female",
    SeniorCitizen: 0,
    Partner: "No",
    Dependents: "No",
    tenure: 2,
    PhoneService: "Yes",
    MultipleLines: "No",
    InternetService: "Fiber optic",
    OnlineSecurity: "No",
    OnlineBackup: "No",
    DeviceProtection: "No",
    TechSupport: "No",
    StreamingTV: "Yes",
    StreamingMovies: "Yes",
    Contract: "Month-to-month",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check",
    MonthlyCharges: 95.5,
    TotalCharges: 191.0,
  },
  loyal: {
    customer_id: "CUST-LOW",
    gender: "Male",
    SeniorCitizen: 0,
    Partner: "Yes",
    Dependents: "Yes",
    tenure: 54,
    PhoneService: "Yes",
    MultipleLines: "Yes",
    InternetService: "DSL",
    OnlineSecurity: "Yes",
    OnlineBackup: "Yes",
    DeviceProtection: "Yes",
    TechSupport: "Yes",
    StreamingTV: "No",
    StreamingMovies: "No",
    Contract: "Two year",
    PaperlessBilling: "No",
    PaymentMethod: "Credit card (automatic)",
    MonthlyCharges: 65.0,
    TotalCharges: 3510.0,
  },
};

export default function App() {
  const [activeTab, setActiveTab] = useState("predict");
  const [apiOnline, setApiOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);

  const [form, setForm] = useState(PRESETS.highRisk);

  useEffect(() => {
    fetchHealth();
    fetchStats();
    fetchHistory();
    // Auto-score the initial profile on mount so the UI is immediately populated for screenshots
    executePrediction(PRESETS.highRisk, false);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiOnline(true);
        setModelLoaded(data.model_loaded);
      }
    } catch {
      setApiOnline(false);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      if (res.ok) setStats(await res.json());
    } catch {}
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history?limit=10`);
      if (res.ok) setHistory(await res.json());
    } catch {}
  };

  const executePrediction = async (payload, showAlert = true) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        const data = await res.json();
        setPrediction(data);
        fetchHistory();
      } else if (showAlert) {
        alert("Prediction failed. Ensure model is trained.");
      }
    } catch {
      if (showAlert) {
        alert("Cannot connect to backend.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "number" ? parseFloat(value) || 0 : value,
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    await executePrediction(form, true);
  };

  return (
    <div className="app-layout">
      {/* Minimal Header */}
      <header className="top-nav">
        <div className="top-nav-inner">
          <div className="brand-group">
            <div className="brand-title">Churnz</div>
            <span className="brand-tag">MLOps Platform</span>
          </div>
          <div className="nav-links">
            <span className={`badge ${apiOnline ? "online" : ""}`}>
              API: {apiOnline ? "Ready" : "Offline"}
            </span>
            <span className={`badge ${modelLoaded ? "online" : ""}`}>
              Model: {modelLoaded ? "Active" : "Untrained"}
            </span>
            <a href="http://localhost:5001" target="_blank" rel="noreferrer" className="link-btn">
              MLflow ↗
            </a>
            <a href={`${API_BASE}/reports/drift`} target="_blank" rel="noreferrer" className="link-btn">
              Drift Report ↗
            </a>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="main-container">
        {/* Minimal Segmented Tabs */}
        <div className="tabs-bar">
          <button
            className={`tab-item ${activeTab === "predict" ? "active" : ""}`}
            onClick={() => setActiveTab("predict")}
          >
            Predict
          </button>
          <button
            className={`tab-item ${activeTab === "metrics" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("metrics");
              fetchStats();
            }}
          >
            Metrics
          </button>
          <button
            className={`tab-item ${activeTab === "drift" ? "active" : ""}`}
            onClick={() => setActiveTab("drift")}
          >
            Drift
          </button>
          <button
            className={`tab-item ${activeTab === "logs" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("logs");
              fetchHistory();
            }}
          >
            Logs
          </button>
        </div>

        {/* TAB: PREDICT */}
        {activeTab === "predict" && (
          <div className="grid-layout">
            {/* Left Card: Customer Profile Input */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Customer Profile</h2>
                <div className="preset-buttons">
                  {/*<button
                    type="button"
                    className="preset-btn"
                    onClick={() => {
                      setForm(PRESETS.highRisk);
                      setPrediction(null);
                    }}
                  >
                    High Risk Preset
                  </button>*/}
                  {/*<button
                    type="button"
                    className="preset-btn"
                    onClick={() => {
                      setForm(PRESETS.loyal);
                      setPrediction(null);
                    }}
                  >
                    Low Risk Preset
                  </button>*/}
                </div>
              </div>

              <form onSubmit={handlePredict}>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Customer ID</label>
                    <input
                      name="customer_id"
                      value={form.customer_id}
                      onChange={handleInputChange}
                    />
                  </div>

                  <div className="form-group">
                    <label>Contract</label>
                    <select
                      name="Contract"
                      value={form.Contract}
                      onChange={handleInputChange}
                    >
                      <option value="Month-to-month">Month-to-month</option>
                      <option value="One year">One year</option>
                      <option value="Two year">Two year</option>
                    </select>
                  </div>

                  <div className="range-box">
                    <div className="range-header">
                      <span>Tenure</span>
                      <strong>{form.tenure} months</strong>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="72"
                      name="tenure"
                      value={form.tenure}
                      onChange={handleInputChange}
                    />
                  </div>

                  <div className="form-group">
                    <label>Internet Service</label>
                    <select
                      name="InternetService"
                      value={form.InternetService}
                      onChange={handleInputChange}
                    >
                      <option value="Fiber optic">Fiber optic</option>
                      <option value="DSL">DSL</option>
                      <option value="No">No</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Tech Support</label>
                    <select
                      name="TechSupport"
                      value={form.TechSupport}
                      onChange={handleInputChange}
                    >
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                      <option value="No internet service">No internet</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Monthly Charges ($)</label>
                    <input
                      type="number"
                      step="0.05"
                      name="MonthlyCharges"
                      value={form.MonthlyCharges}
                      onChange={handleInputChange}
                    />
                  </div>

                  <div className="form-group">
                    <label>Total Charges ($)</label>
                    <input
                      type="number"
                      step="0.1"
                      name="TotalCharges"
                      value={form.TotalCharges}
                      onChange={handleInputChange}
                    />
                  </div>

                  <div className="form-group">
                    <label>Payment Method</label>
                    <select
                      name="PaymentMethod"
                      value={form.PaymentMethod}
                      onChange={handleInputChange}
                    >
                      <option value="Electronic check">Electronic check</option>
                      <option value="Mailed check">Mailed check</option>
                      <option value="Bank transfer (automatic)">Bank transfer (auto)</option>
                      <option value="Credit card (automatic)">Credit card (auto)</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Online Security</label>
                    <select
                      name="OnlineSecurity"
                      value={form.OnlineSecurity}
                      onChange={handleInputChange}
                    >
                      <option value="No">No</option>
                      <option value="Yes">Yes</option>
                      <option value="No internet service">No internet</option>
                    </select>
                  </div>
                </div>

                <button type="submit" disabled={loading} className="btn-primary">
                  {loading ? "Scoring Profile..." : "Predict Churn Risk"}
                </button>
              </form>
            </div>

            {/* Right Card: Prediction Result & TreeSHAP Drivers */}
            <div className="card result-card">
              <div className="card-header">
                <h2 className="card-title">Prediction Result</h2>
                {prediction && (
                  <span className="model-chip">
                    {prediction.model_version || "Champion LightGBM"}
                  </span>
                )}
              </div>

              {!prediction ? (
                <div className="result-placeholder">
                  <div className="placeholder-text">Click "Predict Churn Risk" to evaluate profile</div>
                </div>
              ) : (
                <div className="result-content">
                  {/* Score KPI Banner */}
                  <div className={`score-banner ${prediction.risk_level.toLowerCase()}`}>
                    <div>
                      <div className="score-label">Churn Probability</div>
                      <div className="score-percentage">
                        {(prediction.churn_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="score-badge-wrap">
                      <span className={`risk-pill ${prediction.risk_level.toLowerCase()}`}>
                        {prediction.risk_level} Risk
                      </span>
                      <div className="prediction-outcome">
                        {prediction.churn_prediction === 1 ? "Likely Churn" : "Likely Retained"}
                      </div>
                    </div>
                  </div>

                  {/* Probability Meter with Threshold Ticks */}
                  <div className="progress-section">
                    <div className="progress-bar">
                      <div
                        className={`progress-fill ${prediction.risk_level.toLowerCase()}`}
                        style={{ width: `${Math.min(100, Math.max(0, prediction.churn_probability * 100))}%` }}
                      ></div>
                    </div>
                    <div className="progress-ticks">
                      <span>0% Safe</span>
                      <span>50% Threshold</span>
                      <span>100% Critical</span>
                    </div>
                  </div>

                  {/* Top TreeSHAP Impact Factors */}
                  <div className="factors-section">
                    <div className="factors-heading">Key Risk Drivers</div>
                    <div className="factors-list">
                      {(prediction.risk_factors && prediction.risk_factors.length > 0
                        ? prediction.risk_factors.slice(0, 4)
                        : (prediction.explanation || []).slice(0, 4)
                      ).map((item, idx) => {
                        const isRisk = typeof item === "object" ? item.direction === "increases_risk" : true;
                        const text = typeof item === "object" ? item.display_text : item;
                        return (
                          <div key={idx} className={`factor-item ${isRisk ? "risk-up" : "risk-down"}`}>
                            <span className="factor-indicator">{isRisk ? "▲" : "▼"}</span>
                            <span className="factor-text">{text}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB: METRICS */}
        {activeTab === "metrics" && (
          <div>
            {stats && stats.metrics ? (
              <>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <div className="metric-title">ROC-AUC</div>
                    <div className="metric-num">{stats.metrics.roc_auc}</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-title">PR-AUC</div>
                    <div className="metric-num">{stats.metrics.pr_auc}</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-title">Accuracy</div>
                    <div className="metric-num">{stats.metrics.accuracy}</div>
                  </div>
                  <div className="metric-item">
                    <div className="metric-title">Recall</div>
                    <div className="metric-num">{(stats.metrics.recall * 100).toFixed(1)}%</div>
                  </div>
                </div>

                <div className="card">
                  <h3 className="card-title" style={{ marginBottom: "16px" }}>
                    Top Features
                  </h3>
                  {stats.top_features.map((f, idx) => (
                    <div key={idx} className="bar-row">
                      <span className="bar-name">{f.feature}</span>
                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{
                            width: `${(f.importance / stats.top_features[0].importance) * 100}%`,
                          }}
                        ></div>
                      </div>
                      <span className="bar-val">{f.importance}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="card result-placeholder">No metrics available.</div>
            )}
          </div>
        )}

        {/* TAB: DRIFT */}
        {activeTab === "drift" && (
          <div className="card" style={{ textAlign: "center", padding: "48px 24px" }}>
            <h3 className="card-title" style={{ marginBottom: "8px", fontSize: "16px" }}>
              Evidently AI Drift Monitoring
            </h3>
            <p style={{ color: "var(--text-secondary)", marginBottom: "24px", fontSize: "13px" }}>
              Statistical distribution tests comparing production traffic against training baseline.
            </p>
            <a
              href={`${API_BASE}/reports/drift`}
              target="_blank"
              rel="noreferrer"
              className="btn-primary"
              style={{ display: "inline-block", width: "auto", padding: "10px 24px", textDecoration: "none" }}
            >
              Open Interactive Report ↗
            </a>
          </div>
        )}

        {/* TAB: LOGS */}
        {activeTab === "logs" && (
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Recent Predictions</h3>
              <button onClick={fetchHistory} className="link-btn">
                Refresh
              </button>
            </div>
            {history.length === 0 ? (
              <div className="result-placeholder">No logs found.</div>
            ) : (
              <div className="table-wrap">
                <table className="simple-table">
                  <thead>
                    <tr>
                      <th>Customer</th>
                      <th>Risk</th>
                      <th>Probability</th>
                      <th>Outcome</th>
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((h, i) => (
                      <tr key={i}>
                        <td style={{ fontWeight: 600 }}>{h.customer_id}</td>
                        <td>
                          <span className={`pill ${h.risk_level.toLowerCase()}`}>
                            {h.risk_level}
                          </span>
                        </td>
                        <td>{(h.churn_probability * 100).toFixed(1)}%</td>
                        <td>{h.churn_prediction === 1 ? "Churn" : "Retain"}</td>
                        <td>{h.created_at ? new Date(h.created_at).toLocaleTimeString() : "N/A"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
