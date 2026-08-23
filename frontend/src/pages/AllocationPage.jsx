import React, { useEffect, useState, useMemo } from 'react';
import Papa from 'papaparse';

export default function AllocationPage() {
  const [customers, setCustomers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Interactive optimization parameters
  const [budget, setBudget] = useState(5000);
  const [costPerTreatment, setCostPerTreatment] = useState(10);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterAllocation, setFilterAllocation] = useState('allocated'); // 'all', 'allocated', 'unallocated'
  const [page, setPage] = useState(1);
  const pageSize = 20;

  useEffect(() => {
    fetch('/data/allocation_table.csv')
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
            // Sort by ITE descending
            const sorted = results.data
              .filter((c) => typeof c.ite === 'number')
              .sort((a, b) => b.ite - a.ite)
              .map((c, idx) => ({ ...c, rank: idx + 1 }));

            setCustomers(sorted);
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

  // Compute live budget allocation based on dynamic budget and cost per treatment
  const { allocatedCustomers, targetedCount, totalExpectedConversions, costPerConversion, blanketComparison } = useMemo(() => {
    if (!customers.length) {
      return {
        allocatedCustomers: [],
        targetedCount: 0,
        totalExpectedConversions: 0,
        costPerConversion: 0,
        blanketComparison: { blanketConversions: 0, efficiencyMultiplier: 1 },
      };
    }

    const maxTargets = Math.floor(budget / costPerTreatment);
    let expectedConv = 0;

    const computed = customers.map((c, idx) => {
      const isAllocated = idx < maxTargets && c.ite > 0;
      if (isAllocated) {
        expectedConv += c.ite;
      }
      return {
        ...c,
        isAllocated,
        assignedCost: isAllocated ? costPerTreatment : 0,
        runningCost: (idx + 1) * costPerTreatment,
      };
    });

    const numTargeted = Math.min(maxTargets, computed.filter((c) => c.ite > 0).length);
    const avgEffectAll = customers.reduce((sum, c) => sum + c.ite, 0) / customers.length;
    const blanketExpected = numTargeted * avgEffectAll;
    const efficiencyMultiplier = blanketExpected > 0 ? expectedConv / blanketExpected : 1;

    return {
      allocatedCustomers: computed,
      targetedCount: numTargeted,
      totalExpectedConversions: expectedConv,
      costPerConversion: expectedConv > 0 ? (numTargeted * costPerTreatment) / expectedConv : 0,
      blanketComparison: {
        blanketConversions: blanketExpected,
        efficiencyMultiplier,
      },
    };
  }, [customers, budget, costPerTreatment]);

  // Filter and paginate table rows
  const filteredRows = useMemo(() => {
    return allocatedCustomers.filter((c) => {
      if (filterAllocation === 'allocated' && !c.isAllocated) return false;
      if (filterAllocation === 'unallocated' && c.isAllocated) return false;
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        const segMatch = String(c.segment || '').toLowerCase().includes(term);
        const rankMatch = String(c.rank).includes(term);
        const channelMatch = String(c.channel || '').toLowerCase().includes(term);
        return segMatch || rankMatch || channelMatch;
      }
      return true;
    });
  }, [allocatedCustomers, filterAllocation, searchTerm]);

  const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
  const paginatedRows = filteredRows.slice((page - 1) * pageSize, page * pageSize);

  if (loading) {
    return (
      <div className="card loading-card">
        <div className="spinner"></div>
        <p>Computing prescriptive budget optimization across 64,000 customers...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card error-card">
        <h3>Error Loading Allocation Data</h3>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="allocation-page">
      <div className="page-header">
        <div>
          <h2>Prescriptive Budget Allocation Matrix</h2>
          <p className="subtitle">
            Optimal budget dispatching: Maximizing expected incremental conversions subject to discount constraints
          </p>
        </div>
        <span className="badge badge-accent">Week 3 Milestone</span>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card highlight">
          <div className="kpi-label">Assigned Budget</div>
          <div className="kpi-value text-accent">${budget.toLocaleString()}</div>
          <div className="kpi-subtext">${costPerTreatment} discount per customer</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Customers Prescribed</div>
          <div className="kpi-value">{targetedCount.toLocaleString()}</div>
          <div className="kpi-subtext">{((targetedCount / customers.length) * 100).toFixed(1)}% of user base</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Expected Incremental Conversions</div>
          <div className="kpi-value text-success">+{totalExpectedConversions.toFixed(2)}</div>
          <div className="kpi-subtext">Net additional purchases</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Causal Targeting Efficiency</div>
          <div className="kpi-value text-accent">{blanketComparison.efficiencyMultiplier.toFixed(2)}x</div>
          <div className="kpi-subtext">vs. blanket / random campaign (+{(totalExpectedConversions - blanketComparison.blanketConversions).toFixed(1)} lift)</div>
        </div>
      </div>

      {/* Interactive Controls Bar */}
      <div className="card control-panel">
        <div className="controls-row">
          <div className="control-group">
            <label className="control-label">Total Campaign Budget ($):</label>
            <div className="budget-preset-buttons">
              {[1000, 2500, 5000, 10000, 25000].map((b) => (
                <button
                  key={b}
                  className={`btn-preset ${budget === b ? 'active' : ''}`}
                  onClick={() => {
                    setBudget(b);
                    setPage(1);
                  }}
                >
                  ${(b / 1000).toFixed(0)}k
                </button>
              ))}
            </div>
            <input
              type="range"
              min="500"
              max="50000"
              step="500"
              value={budget}
              onChange={(e) => {
                setBudget(Number(e.target.value));
                setPage(1);
              }}
              className="styled-slider"
              style={{ marginTop: '0.75rem' }}
            />
          </div>

          <div className="control-group" style={{ maxWidth: '240px' }}>
            <label className="control-label">Discount Cost / Offer ($):</label>
            <input
              type="number"
              min="1"
              max="100"
              value={costPerTreatment}
              onChange={(e) => {
                setCostPerTreatment(Math.max(1, Number(e.target.value)));
                setPage(1);
              }}
              className="number-input"
            />
          </div>
        </div>
      </div>

      {/* Customer Allocation Matrix Table */}
      <div className="card table-card">
        <div className="table-header-controls">
          <div>
            <h3>Customer Prescriptions ({filteredRows.length.toLocaleString()} matching)</h3>
            <p className="chart-desc">Sorted by predicted Individual Treatment Effect (ITE)</p>
          </div>

          <div className="table-filters">
            <input
              type="text"
              placeholder="Search segment or rank..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setPage(1);
              }}
              className="search-input"
            />
            <div className="filter-buttons">
              <button
                className={`btn-filter ${filterAllocation === 'all' ? 'active' : ''}`}
                onClick={() => {
                  setFilterAllocation('all');
                  setPage(1);
                }}
              >
                All (64k)
              </button>
              <button
                className={`btn-filter ${filterAllocation === 'allocated' ? 'active' : ''}`}
                onClick={() => {
                  setFilterAllocation('allocated');
                  setPage(1);
                }}
              >
                Targeted ({targetedCount})
              </button>
              <button
                className={`btn-filter ${filterAllocation === 'unallocated' ? 'active' : ''}`}
                onClick={() => {
                  setFilterAllocation('unallocated');
                  setPage(1);
                }}
              >
                Skipped
              </button>
            </div>
          </div>
        </div>

        <div className="table-responsive">
          <table className="custom-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Past Spend ($)</th>
                <th>Recency (Mo)</th>
                <th>Segment</th>
                <th>Predicted ITE ($\tau_i$)</th>
                <th>Lift per $</th>
                <th>Prescription</th>
              </tr>
            </thead>
            <tbody>
              {paginatedRows.map((row) => (
                <tr key={row.rank} className={row.isAllocated ? 'row-allocated' : ''}>
                  <td className="font-mono">#{row.rank.toLocaleString()}</td>
                  <td className="font-mono">${(row.history || 0).toFixed(2)}</td>
                  <td>{row.recency} mo</td>
                  <td>
                    <span className="segment-tag">{row.segment || 'Customer'}</span>
                  </td>
                  <td className="font-mono text-accent">
                    +{(row.ite * 100).toFixed(3)}%
                  </td>
                  <td className="font-mono text-muted">
                    {(row.ite / costPerTreatment).toFixed(6)}
                  </td>
                  <td>
                    {row.isAllocated ? (
                      <span className="badge badge-success">🎯 Send ${costPerTreatment} Discount</span>
                    ) : (
                      <span className="badge badge-muted">Do Not Target</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="pagination-bar">
          <span className="pagination-info">
            Showing {(page - 1) * pageSize + 1}&ndash;{Math.min(page * pageSize, filteredRows.length)} of {filteredRows.length.toLocaleString()}
          </span>
          <div className="pagination-buttons">
            <button
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="btn-page"
            >
              &larr; Prev
            </button>
            <span className="page-indicator">
              Page {page} of {totalPages}
            </span>
            <button
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="btn-page"
            >
              Next &rarr;
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
