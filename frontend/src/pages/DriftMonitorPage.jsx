import React, { useState, useEffect } from 'react';

export default function DriftMonitorPage() {
  const [scenario, setScenario] = useState('economic_downturn');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(true);

  const fetchDriftReport = (selectedScenario) => {
    setLoading(true);
    fetch(`http://127.0.0.1:8008/api/drift/simulate?shift_type=${selectedScenario}`)
      .then((res) => {
        if (!res.ok) throw new Error('API request failed');
        return res.json();
      })
      .then((data) => {
        setReport(data);
        setApiOnline(true);
        setLoading(false);
      })
      .catch(() => {
        // Fallback simulation report in case backend is offline
        setApiOnline(false);
        const fallback = generateFallbackReport(selectedScenario);
        setReport(fallback);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchDriftReport(scenario);
  }, [scenario]);

  return (
    <div className="drift-page">
      <div className="page-header">
        <div>
          <h2>Statistical Data Drift & Covariate Shift Detector</h2>
          <p className="subtitle">
            Automated quality assurance: Detecting distribution shift between training baseline and incoming customer cohorts
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className={`badge ${apiOnline ? 'badge-success' : 'badge-accent'}`}>
            {apiOnline ? '🟢 FastAPI Connected (:8008)' : '🟡 Client-Side Demo Mode'}
          </span>
          <span className="badge badge-accent">Week 4 Milestone</span>
        </div>
      </div>

      {/* Scenario Selector */}
      <div className="card control-panel">
        <div className="controls-row">
          <div className="control-group">
            <label className="control-label">Select Simulated Market Regime / Customer Shift:</label>
            <div className="budget-preset-buttons">
              <button
                className={`btn-preset ${scenario === 'economic_downturn' ? 'active' : ''}`}
                onClick={() => setScenario('economic_downturn')}
              >
                📉 Economic Downturn (Spend Drops, Recency Lags)
              </button>
              <button
                className={`btn-preset ${scenario === 'web_channel_surge' ? 'active' : ''}`}
                onClick={() => setScenario('web_channel_surge')}
              >
                🌐 Web Channel Surge (Newbie Influx)
              </button>
              <button
                className={`btn-preset ${scenario === 'normal_stable' ? 'active' : ''}`}
                onClick={() => setScenario('normal_stable')}
              >
                ✅ Normal Stable Cohort (No Drift)
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Status Summary Banner */}
      {report && (
        <div className="kpi-grid">
          <div className="kpi-card highlight">
            <div className="kpi-label">Drift Status</div>
            <div
              className={`kpi-value ${
                report.overall_status === 'CRITICAL_DRIFT'
                  ? 'text-danger'
                  : report.overall_status === 'MODERATE_DRIFT'
                  ? 'text-warning'
                  : 'text-success'
              }`}
            >
              {report.overall_status.replace('_', ' ')}
            </div>
            <div className="kpi-subtext">
              {report.drift_detected_features_count} of {report.total_features_tested} features drifted
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Baseline Population</div>
            <div className="kpi-value">{report.baseline_sample_size.toLocaleString()}</div>
            <div className="kpi-subtext">Training reference sample</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Incoming Test Batch</div>
            <div className="kpi-value">{report.incoming_sample_size.toLocaleString()}</div>
            <div className="kpi-subtext">Inference batch evaluated</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Statistical Testing Suite</div>
            <div className="kpi-value text-accent">KS + PSI</div>
            <div className="kpi-subtext">$\alpha = 0.05$ significance cutoff</div>
          </div>
        </div>
      )}

      {/* Detailed Feature Drift Audit Table */}
      <div className="card table-card">
        <div className="chart-header">
          <h3>Per-Feature Statistical Diagnostic Matrix</h3>
          <p className="chart-desc">
            Continuous variables tested via 2-Sample Kolmogorov-Smirnov (KS). Categorical features tested via Population Stability Index (PSI).
          </p>
        </div>

        {loading ? (
          <div className="loading-card">
            <div className="spinner"></div>
            <p>Running Kolmogorov-Smirnov & PSI tests...</p>
          </div>
        ) : (
          <div className="table-responsive">
            <table className="custom-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Data Type</th>
                  <th>Statistical Test</th>
                  <th>Test Statistic</th>
                  <th>P-Value / Metric</th>
                  <th>PSI Score</th>
                  <th>Baseline vs Current</th>
                  <th>Drift Flag</th>
                </tr>
              </thead>
              <tbody>
                {report?.features.map((feat) => (
                  <tr key={feat.feature} className={feat.drift_detected ? 'row-drift-alert' : ''}>
                    <td>
                      <strong className="font-mono">{feat.feature}</strong>
                    </td>
                    <td>
                      <span className="segment-tag">{feat.type}</span>
                    </td>
                    <td className="text-muted">{feat.test}</td>
                    <td className="font-mono">{feat.statistic.toFixed(4)}</td>
                    <td className="font-mono">
                      {feat.p_value !== null ? (
                        <span className={feat.p_value < 0.05 ? 'text-danger' : 'text-success'}>
                          $p = {feat.p_value.toFixed(4)}$
                        </span>
                      ) : (
                        <span className="text-muted">Categorical</span>
                      )}
                    </td>
                    <td className="font-mono">
                      <span className={feat.psi > 0.1 ? 'text-danger font-bold' : 'text-muted'}>
                        {feat.psi.toFixed(4)}
                      </span>
                    </td>
                    <td className="font-mono text-muted">
                      {feat.type === 'numerical'
                        ? `${feat.baseline_mean} → ${feat.current_mean}`
                        : 'Distribution shift'}
                    </td>
                    <td>
                      {feat.drift_detected ? (
                        <span className="badge badge-danger">⚠️ DRIFT DETECTED</span>
                      ) : (
                        <span className="badge badge-success">✓ STABLE</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Explanation Box */}
      <div className="insight-card">
        <div className="insight-icon">🛡️</div>
        <div className="insight-content">
          <h4>Why Drift Detection is critical for Causal ML</h4>
          <p>
            Standard predictive models degrade quietly when customer behavior shifts; however, <strong>causal inference policies suffer severe allocation mispricing</strong> if the confounder distribution changes (violating positivity or unconfoundedness assumptions).
            When a feature triggers a <code>CRITICAL DRIFT</code> alert ($p &lt; 0.05, \text{PSI} &gt; 0.10$), the automated pipeline flags the model for retraining before marketing budget is misallocated.
          </p>
        </div>
      </div>
    </div>
  );
}

function generateFallbackReport(scenario) {
  if (scenario === 'economic_downturn') {
    return {
      overall_status: 'CRITICAL_DRIFT',
      drift_detected_features_count: 2,
      total_features_tested: 7,
      baseline_sample_size: 64000,
      incoming_sample_size: 3000,
      features: [
        { feature: 'history', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.205, p_value: 0.0, psi: 0.1975, drift_detected: true, baseline_mean: 242.09, current_mean: 148.95 },
        { feature: 'recency', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.3499, p_value: 0.0, psi: 2.7191, drift_detected: true, baseline_mean: 5.76, current_mean: 8.44 },
        { feature: 'mens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0002, p_value: null, psi: 0.0002, drift_detected: false },
        { feature: 'womens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false },
        { feature: 'newbie', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0008, p_value: null, psi: 0.0008, drift_detected: false },
        { feature: 'channel', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false },
        { feature: 'zip_code', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false },
      ],
    };
  }
  return {
    overall_status: 'STABLE',
    drift_detected_features_count: 0,
    total_features_tested: 7,
    baseline_sample_size: 64000,
    incoming_sample_size: 3000,
    features: [
      { feature: 'history', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.012, p_value: 0.82, psi: 0.0014, drift_detected: false, baseline_mean: 242.09, current_mean: 243.12 },
      { feature: 'recency', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.015, p_value: 0.74, psi: 0.0021, drift_detected: false, baseline_mean: 5.76, current_mean: 5.74 },
      { feature: 'mens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false },
      { feature: 'womens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false },
      { feature: 'newbie', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0002, p_value: null, psi: 0.0002, drift_detected: false },
      { feature: 'channel', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0002, p_value: null, psi: 0.0002, drift_detected: false },
      { feature: 'zip_code', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false },
    ],
  };
}
