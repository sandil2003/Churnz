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

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "number" ? parseFloat(value) || 0 : value,
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (res.ok) {
        setPrediction(await res.json());
        fetchHistory();
      } else {
        alert("Prediction failed. Ensure model is trained.");
      }
    } catch {
      alert("Cannot connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Minimal Navbar */}
      <header className="top-nav">
        <div className="top-nav-inner">
          <div className="brand-title">
            <span className="brand-dot"></span>
            Churnz
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
        {/* Minimal Tabs */}
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
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Customer Profile</h2>
                <div className="preset-buttons">
                  <button
                    type="button"
                    className="preset-btn"
                    onClick={() => {
                      setForm(PRESETS.highRisk);
                      setPrediction(null);
                    }}
                  >
                    High Risk Preset
                  </button>
                  <button
                    type="button"
                    className="preset-btn"
                    onClick={() => {
                      setForm(PRESETS.loyal);
                      setPrediction(null);
                    }}
                  >
                    Low Risk Preset
                  </button>
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
                  {loading ? "Calculating..." : "Predict Churn"}
                </button>
              </form>
            </div>

            {/* Result Card */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Prediction Result</h2>
              </div>

              {!prediction ? (
                <div className="result-placeholder">
                  Fill the form and click "Predict Churn" to see score.
                </div>
              ) : (
                <div>
                  <div className={`score-banner ${prediction.risk_level.toLowerCase()}`}>
                    <div className="score-tier">{prediction.risk_level} Risk</div>
                    <div className="score-status">
                      {prediction.churn_prediction === 1 ? "Likely Churn" : "Likely Retained"}
                    </div>
                  </div>

                  <div className="progress-section">
                    <div className="progress-label">
                      <span>Probability</span>
                      <strong>{(prediction.churn_probability * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="progress-bar">
                      <div
                        className={`progress-fill ${prediction.risk_level.toLowerCase()}`}
                        style={{ width: `${prediction.churn_probability * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="factors-list">
                    {prediction.explanation.map((item, idx) => (
                      <div key={idx} className="factor-item">
                        • {item}
                      </div>
                    ))}
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
                  <h3 className="card-title" style={{ marginBottom: "14px" }}>
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
          <div className="card" style={{ textAlign: "center", padding: "40px 20px" }}>
            <h3 className="card-title" style={{ marginBottom: "8px" }}>
              Evidently AI Drift Report
            </h3>
            <p style={{ color: "var(--text-secondary)", marginBottom: "20px", fontSize: "13px" }}>
              Compares live prediction logs against training baseline to detect data drift.
            </p>
            <a
              href={`${API_BASE}/reports/drift`}
              target="_blank"
              rel="noreferrer"
              className="link-btn"
              style={{ display: "inline-block", background: "var(--brand)", color: "#fff", padding: "8px 16px" }}
            >
              Open HTML Report ↗
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
