import React, { useState, useEffect } from 'react';

export default function DriftMonitorPage() {
  const [scenario, setScenario] = useState('economic_downturn');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);

  const fetchDriftReport = (selectedScenario) => {
    setLoading(true);
    fetch(`http://127.0.0.1:8008/api/drift/simulate?shift_type=${selectedScenario}`)
      .then((res) => {
        if (!res.ok) throw new Error('Backend API offline');
        return res.json();
      })
      .then((data) => {
        setReport(data);
        setApiOnline(true);
        setLoading(false);
      })
      .catch(() => {
        // Fallback simulation report in case FastAPI backend is offline
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
          <h2>Statistical Data Drift &amp; Covariate Shift Detector</h2>
          <p className="subtitle">
            Automated quality assurance: Detecting distribution shift between training baseline and incoming customer cohorts
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className={`badge ${apiOnline ? 'badge-success' : 'badge-accent'}`}>
            {apiOnline ? '🟢 FastAPI Connected (:8008)' : '🟡 Client-Side Demo Mode'}
          </span>
          <span className="badge badge-accent">Quality Control</span>
        </div>
      </div>

      {/* Scenario Selector */}
      <div className="card control-panel">
        <div className="controls-row">
          <div className="control-group">
            <label className="control-label">Select Simulated Market Regime / Customer Shift:</label>
            <div className="budget-preset-buttons" style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
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
              {(report.overall_status || 'STABLE').replaceAll('_', ' ')}
            </div>
            <div className="kpi-subtext">
              {report.drift_detected_features_count} of {report.total_features_tested} features drifted
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Baseline Population</div>
            <div className="kpi-value">{(report.baseline_sample_size || 64000).toLocaleString()}</div>
            <div className="kpi-subtext">Training reference sample</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Incoming Test Batch</div>
            <div className="kpi-value">{(report.incoming_sample_size || 3000).toLocaleString()}</div>
            <div className="kpi-subtext">Inference batch evaluated</div>
          </div>

          <div className="kpi-card">
            <div className="kpi-label">Statistical Testing Suite</div>
            <div className="kpi-value text-accent">KS + PSI</div>
            <div className="kpi-subtext">α = 0.05 significance cutoff</div>
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
          <div className="loading-card" style={{ padding: '3rem', textAlign: 'center' }}>
            <div className="spinner"></div>
            <p>Running Kolmogorov-Smirnov &amp; PSI statistical tests...</p>
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
                  <th>P-Value / Significance</th>
                  <th>PSI Score</th>
                  <th>Baseline vs Current</th>
                  <th>Drift Status</th>
                </tr>
              </thead>
              <tbody>
                {report?.features?.map((feat) => (
                  <tr key={feat.feature} className={feat.drift_detected ? 'row-drift-alert' : ''}>
                    <td>
                      <strong className="font-mono">{feat.feature}</strong>
                    </td>
                    <td>
                      <span className="segment-tag">{feat.type}</span>
                    </td>
                    <td className="text-muted">{feat.test}</td>
                    <td className="font-mono">{typeof feat.statistic === 'number' ? feat.statistic.toFixed(4) : '—'}</td>
                    <td className="font-mono">
                      {feat.p_value !== null && feat.p_value !== undefined ? (
                        <span className={feat.p_value < 0.05 ? 'text-danger font-semibold' : 'text-success'}>
                          {feat.p_value < 0.0001 ? 'p < 0.0001' : `p = ${feat.p_value.toFixed(4)}`}
                        </span>
                      ) : (
                        <span className="text-muted">Categorical (PSI)</span>
                      )}
                    </td>
                    <td className="font-mono">
                      <span className={feat.psi > 0.1 ? 'text-danger font-bold' : 'text-muted'}>
                        {typeof feat.psi === 'number' ? feat.psi.toFixed(4) : '—'}
                      </span>
                    </td>
                    <td className="font-mono text-muted">
                      {feat.type === 'numerical'
                        ? `${feat.baseline_mean} → ${feat.current_mean}`
                        : feat.shift_summary || 'Stable distribution'}
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

      {/* Explanation Card */}
      <div className="insight-card" style={{ marginTop: '1.5rem', display: 'flex', gap: '1rem', padding: '1.25rem', borderRadius: '12px', background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
        <div className="insight-icon" style={{ fontSize: '1.8rem' }}>🛡️</div>
        <div className="insight-content">
          <h4 style={{ margin: '0 0 0.4rem 0', fontSize: '1.05rem', color: '#f8fafc' }}>Why Drift Detection is Critical for Causal Machine Learning</h4>
          <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.9rem', lineHeight: '1.5' }}>
            Standard predictive ML models degrade quietly when customer behavior shifts; however, <strong>causal inference policies suffer severe allocation mispricing</strong> if the confounder distribution changes (violating positivity or unconfoundedness assumptions).
            When a feature triggers a <code>CRITICAL DRIFT</code> alert (p &lt; 0.05 or PSI &gt; 0.10), the automated pipeline flags the model for retraining before marketing budget is misallocated.
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
        { feature: 'history', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.2050, p_value: 0.0, psi: 0.1975, drift_detected: true, baseline_mean: 242.09, current_mean: 148.95 },
        { feature: 'recency', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.3499, p_value: 0.0, psi: 2.7191, drift_detected: true, baseline_mean: 5.76, current_mean: 8.44 },
        { feature: 'mens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0002, p_value: null, psi: 0.0002, drift_detected: false, shift_summary: '48% → 48%' },
        { feature: 'womens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false, shift_summary: '52% → 52%' },
        { feature: 'newbie', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0008, p_value: null, psi: 0.0008, drift_detected: false, shift_summary: '50% → 51%' },
        { feature: 'channel', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false, shift_summary: 'Multichannel stable' },
        { feature: 'zip_code', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false, shift_summary: 'Geo stable' },
      ],
    };
  }

  if (scenario === 'web_channel_surge') {
    return {
      overall_status: 'MODERATE_DRIFT',
      drift_detected_features_count: 2,
      total_features_tested: 7,
      baseline_sample_size: 64000,
      incoming_sample_size: 3000,
      features: [
        { feature: 'channel', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.1420, p_value: null, psi: 0.1850, drift_detected: true, shift_summary: 'Web channel surge (+35%)' },
        { feature: 'newbie', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.1180, p_value: null, psi: 0.1420, drift_detected: true, shift_summary: 'Newbie influx (50% → 72%)' },
        { feature: 'history', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.0320, p_value: 0.18, psi: 0.0150, drift_detected: false, baseline_mean: 242.09, current_mean: 238.40 },
        { feature: 'recency', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.0210, p_value: 0.45, psi: 0.0080, drift_detected: false, baseline_mean: 5.76, current_mean: 5.65 },
        { feature: 'mens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0004, p_value: null, psi: 0.0004, drift_detected: false, shift_summary: '48% → 47%' },
        { feature: 'womens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0003, p_value: null, psi: 0.0003, drift_detected: false, shift_summary: '52% → 53%' },
        { feature: 'zip_code', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0002, p_value: null, psi: 0.0002, drift_detected: false, shift_summary: 'Geo stable' },
      ],
    };
  }

  // Normal Stable Cohort
  return {
    overall_status: 'STABLE',
    drift_detected_features_count: 0,
    total_features_tested: 7,
    baseline_sample_size: 64000,
    incoming_sample_size: 3000,
    features: [
      { feature: 'history', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.0120, p_value: 0.82, psi: 0.0014, drift_detected: false, baseline_mean: 242.09, current_mean: 243.12 },
      { feature: 'recency', type: 'numerical', test: 'Kolmogorov-Smirnov + PSI', statistic: 0.0150, p_value: 0.74, psi: 0.0021, drift_detected: false, baseline_mean: 5.76, current_mean: 5.74 },
      { feature: 'mens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false, shift_summary: '48% → 48%' },
      { feature: 'womens', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false, shift_summary: '52% → 52%' },
      { feature: 'newbie', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0002, p_value: null, psi: 0.0002, drift_detected: false, shift_summary: '50% → 50%' },
      { feature: 'channel', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0002, p_value: null, psi: 0.0002, drift_detected: false, shift_summary: 'Multichannel stable' },
      { feature: 'zip_code', type: 'categorical', test: 'Population Stability Index (PSI)', statistic: 0.0001, p_value: null, psi: 0.0001, drift_detected: false, shift_summary: 'Geo stable' },
    ],
  };
}
