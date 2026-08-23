import React, { useEffect, useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import Papa from 'papaparse';

export default function QiniCurvePage() {
  const [data, setData] = useState({ ranks: [], qinis: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [samplePct, setSamplePct] = useState(25); // Top 25% cutoff slider for inspection

  useEffect(() => {
    fetch('/data/qini_curve.csv')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.text();
      })
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const ranks = [];
            const qinis = [];
            // Downsample slightly for ultra-smooth 60fps rendering in browser (1 every 8 points)
            const step = Math.max(1, Math.floor(results.data.length / 4000));
            
            for (let i = 0; i < results.data.length; i += step) {
              const row = results.data[i];
              if (row && typeof row.rank === 'number' && typeof row.qini === 'number') {
                ranks.push(row.rank);
                qinis.push(row.qini);
              }
            }
            // Always include last endpoint
            const lastRow = results.data[results.data.length - 1];
            if (lastRow && ranks[ranks.length - 1] !== lastRow.rank) {
              ranks.push(lastRow.rank);
              qinis.push(lastRow.qini);
            }

            setData({ ranks, qinis, rawTotal: results.data.length });
            setLoading(false);
          },
          error: (err) => {
            setError(err.message);
            setLoading(false);
          },
        });
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const metrics = useMemo(() => {
    if (!data.ranks.length) return null;
    const maxRank = data.ranks[data.ranks.length - 1];
    const maxQini = Math.max(...data.qinis);
    const finalQini = data.qinis[data.qinis.length - 1];
    
    // Calculate targeted stats at current cutoff percentage
    const targetRank = Math.floor((samplePct / 100) * maxRank);
    let modelUpliftAtCutoff = 0;
    for (let i = 0; i < data.ranks.length; i++) {
      if (data.ranks[i] >= targetRank) {
        modelUpliftAtCutoff = data.qinis[i];
        break;
      }
    }
    const randomUpliftAtCutoff = (targetRank / maxRank) * finalQini;
    const upliftGain = modelUpliftAtCutoff - randomUpliftAtCutoff;

    return {
      maxRank,
      maxQini,
      finalQini,
      targetRank,
      modelUpliftAtCutoff,
      randomUpliftAtCutoff,
      upliftGain,
    };
  }, [data, samplePct]);

  if (loading) {
    return (
      <div className="card loading-card">
        <div className="spinner"></div>
        <p>Loading Qini Curve dataset from Double Machine Learning model...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card error-card">
        <h3>Error Loading Qini Data</h3>
        <p>{error}</p>
      </div>
    );
  }

  const maxRank = metrics?.maxRank || 64000;
  const finalQini = metrics?.finalQini || 0;

  return (
    <div className="qini-page">
      <div className="page-header">
        <div>
          <h2>Uplift & Qini Evaluation</h2>
          <p className="subtitle">
            Validating EconML Double Machine Learning (DML) targeting efficiency vs. random marketing policy
          </p>
        </div>
        <span className="badge badge-accent">Week 2 Milestone</span>
      </div>

      {/* Summary KPI cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Total Population</div>
          <div className="kpi-value">{maxRank.toLocaleString()}</div>
          <div className="kpi-subtext">Hillstrom test sample</div>
        </div>

        <div className="kpi-card highlight">
          <div className="kpi-label">Max Cumulative Uplift</div>
          <div className="kpi-value text-accent">{metrics ? metrics.maxQini.toFixed(1) : '--'}</div>
          <div className="kpi-subtext">Peak incremental conversions</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Total Experiment Lift</div>
          <div className="kpi-value">{metrics ? metrics.finalQini.toFixed(1) : '--'}</div>
          <div className="kpi-subtext">If 100% population emailed</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Uplift Gain @ {samplePct}% Targeted</div>
          <div className="kpi-value text-success">
            {metrics ? `+${metrics.upliftGain.toFixed(1)}` : '--'}
          </div>
          <div className="kpi-subtext">
            Conversions saved vs Random ({metrics ? metrics.modelUpliftAtCutoff.toFixed(1) : 0} vs{' '}
            {metrics ? metrics.randomUpliftAtCutoff.toFixed(1) : 0})
          </div>
        </div>
      </div>

      {/* Chart container */}
      <div className="card chart-card">
        <div className="chart-header">
          <div>
            <h3>Uplift: Model Targeting vs Random Targeting</h3>
            <p className="chart-desc">
              The gap between the blue curve and the gray dashed line represents the incremental conversion benefit of DML causal ranking.
            </p>
          </div>
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: data.ranks,
                y: data.qinis,
                type: 'scatter',
                mode: 'lines',
                name: 'EconML DML Model (Ranked by ITE)',
                line: { color: '#6366f1', width: 3 },
                hoverinfo: 'x+y',
                hovertemplate: '<b>Rank:</b> %{x:,}<br><b>Cumulative Incremental:</b> %{y:.2f}<extra></extra>',
              },
              {
                x: [0, maxRank],
                y: [0, finalQini],
                type: 'scatter',
                mode: 'lines',
                name: 'Random Targeting Baseline',
                line: { color: '#94a3b8', width: 2, dash: 'dash' },
                hoverinfo: 'x+y',
                hovertemplate: '<b>Rank:</b> %{x:,}<br><b>Random Expected:</b> %{y:.2f}<extra></extra>',
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 60, r: 30, t: 30, b: 60 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              font: { family: 'Inter, system-ui, sans-serif', color: '#cbd5e1' },
              xaxis: {
                title: { text: 'Customers Ranked (Sorted by Predicted ITE Descending)', font: { size: 13, color: '#94a3b8' } },
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                zerolinecolor: 'rgba(148, 163, 184, 0.2)',
                tickformat: ',',
              },
              yaxis: {
                title: { text: 'Cumulative Incremental Conversions (Qini)', font: { size: 13, color: '#94a3b8' } },
                gridcolor: 'rgba(148, 163, 184, 0.1)',
                zerolinecolor: 'rgba(148, 163, 184, 0.2)',
              },
              legend: {
                orientation: 'h',
                y: 1.12,
                x: 0,
                font: { color: '#e2e8f0', size: 12 },
              },
              hovermode: 'x unified',
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '480px' }}
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>

        {/* Interactive targeting slider */}
        <div className="slider-box">
          <div className="slider-header">
            <span>Targeting Budget / Fraction Slider: <strong>Top {samplePct}%</strong> ({metrics?.targetRank.toLocaleString()} customers)</span>
            <span className="slider-gain">
              Model captures <strong>{((metrics?.modelUpliftAtCutoff / (finalQini || 1)) * 100).toFixed(1)}%</strong> of all total conversions in the first {samplePct}% of spend
            </span>
          </div>
          <input
            type="range"
            min="1"
            max="100"
            value={samplePct}
            onChange={(e) => setSamplePct(Number(e.target.value))}
            className="styled-slider"
          />
        </div>
      </div>

      {/* Proof & Business Insight Callout */}
      <div className="insight-card">
        <div className="insight-icon">💡</div>
        <div className="insight-content">
          <h4>Why this curve proves causal model superiority</h4>
          <p>
            Standard response models target users based on overall propensity $P(Y=1)$, which often selects &ldquo;Sure Things&rdquo; (customers who would buy anyway without any email). 
            Our <strong>Double Machine Learning (DML) model</strong> specifically identifies the <em>Individual Treatment Effect ($\tau_i$)</em>, prioritizing <strong>Persuadables</strong>.
            Notice how the model curve steepens aggressively in the first 20&ndash;30% of the customer base, capturing the vast majority of incremental conversions before flattening out.
          </p>
        </div>
      </div>
    </div>
  );
}
