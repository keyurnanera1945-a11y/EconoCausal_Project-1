import React, { useEffect, useState, useMemo } from 'react';
import Papa from 'papaparse';

export default function AllocationPage() {
  const [rawCustomers, setRawCustomers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Budget slider & filter parameters (STEP 4: default $5000, $1000–$20000)
  const [budget, setBudget] = useState(5000);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterAllocation, setFilterAllocation] = useState('allocated'); // 'allocated', 'all', 'unallocated'

  // Sorting state (STEP 3: Clickable headers to sort asc/desc)
  const [sortField, setSortField] = useState('rank');
  const [sortDirection, setSortDirection] = useState('asc'); // 'asc' | 'desc'

  // Pagination state (STEP 3: 25 rows per page)
  const [page, setPage] = useState(1);
  const pageSize = 25;

  useEffect(() => {
    fetch('/data/allocation_table_tiered.csv')
      .then((res) => {
        if (!res.ok) {
          return fetch('/data/allocation_table.csv');
        }
        return res;
      })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error loading allocation table: ${res.status}`);
        return res.text();
      })
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const parsed = results.data
              .filter((c) => c && (typeof c.ite === 'number' || typeof c.expected_incremental_conversion === 'number'))
              .map((c) => {
                const discountTier = c.discount_tier || 'medium';
                let tierCost = c.tier_cost;
                if (!tierCost) {
                  if (discountTier === 'low') tierCost = 5.0;
                  else if (discountTier === 'high') tierCost = 20.0;
                  else tierCost = 10.0;
                }
                const multiplier = c.ite_multiplier || (discountTier === 'low' ? 1.0 : discountTier === 'high' ? 1.6 : 1.3);
                const baseIte = typeof c.ite === 'number' ? c.ite : 0.008;
                const expectedConv = typeof c.expected_incremental_conversion === 'number'
                  ? c.expected_incremental_conversion
                  : baseIte * multiplier;

                return {
                  ...c,
                  ite: baseIte,
                  discount_tier: discountTier,
                  tier_cost: tierCost,
                  ite_multiplier: multiplier,
                  expected_incremental_conversion: expectedConv,
                };
              });

            // Pre-sort by expected incremental conversion descending
            parsed.sort((a, b) => b.expected_incremental_conversion - a.expected_incremental_conversion);
            setRawCustomers(parsed);
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

  // Compute live budget allocation client-side based on active budget slider value
  const { customersWithAllocation, targetedCount, totalSpend, totalExpectedConversions, tierBreakdown } = useMemo(() => {
    if (!rawCustomers.length) {
      return {
        customersWithAllocation: [],
        targetedCount: 0,
        totalSpend: 0,
        totalExpectedConversions: 0,
        tierBreakdown: { low: 0, medium: 0, high: 0 },
      };
    }

    let currentSpend = 0;
    let totalConv = 0;
    let count = 0;
    const breakdown = { low: 0, medium: 0, high: 0 };

    const processed = rawCustomers.map((c, idx) => {
      const cost = c.tier_cost || 10.0;
      const canFit = (currentSpend + cost) <= budget + 1e-6;

      let isAllocated = false;
      if (canFit && c.expected_incremental_conversion > 0) {
        isAllocated = true;
        currentSpend += cost;
        totalConv += c.expected_incremental_conversion;
        count++;
        if (breakdown[c.discount_tier] !== undefined) {
          breakdown[c.discount_tier]++;
        }
      }

      return {
        ...c,
        rank: idx + 1,
        isAllocated,
        assignedCost: isAllocated ? cost : 0,
        runningSpend: currentSpend,
      };
    });

    return {
      customersWithAllocation: processed,
      targetedCount: count,
      totalSpend: currentSpend,
      totalExpectedConversions: totalConv,
      tierBreakdown: breakdown,
    };
  }, [rawCustomers, budget]);

  // Filter customers by search term and allocation status
  const filteredRows = useMemo(() => {
    return customersWithAllocation.filter((c) => {
      if (filterAllocation === 'allocated' && !c.isAllocated) return false;
      if (filterAllocation === 'unallocated' && c.isAllocated) return false;
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        const segMatch = String(c.segment || '').toLowerCase().includes(term);
        const rankMatch = String(c.rank).includes(term);
        const tierMatch = String(c.discount_tier || '').toLowerCase().includes(term);
        return segMatch || rankMatch || tierMatch;
      }
      return true;
    });
  }, [customersWithAllocation, filterAllocation, searchTerm]);

  // Column header sort toggle handler
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Sort rows dynamically by sortField and sortDirection
  const sortedRows = useMemo(() => {
    return [...filteredRows].sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      if (typeof aVal === 'string') {
        aVal = aVal.toLowerCase();
        bVal = String(bVal || '').toLowerCase();
      }

      if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  }, [filteredRows, sortField, sortDirection]);

  // Pagination slice (25 rows per page)
  const totalPages = Math.ceil(sortedRows.length / pageSize) || 1;
  const paginatedRows = useMemo(() => {
    const start = (page - 1) * pageSize;
    return sortedRows.slice(start, start + pageSize);
  }, [sortedRows, page, pageSize]);

  // Budget change handler (constrained between $1,000 and $20,000)
  const handleBudgetChange = (val) => {
    const b = Math.min(20000, Math.max(1000, Number(val)));
    setBudget(b);
    setPage(1);
  };

  if (loading) {
    return (
      <div className="card loading-card">
        <div className="spinner"></div>
        <p>Loading multi-tier prescriptive allocation matrix...</p>
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
            SciPy LP multi-tier optimization: Maximizing expected incremental conversions across dynamic discount tiers
          </p>
        </div>
        <span className="badge badge-accent">Week 3 LP Milestone</span>
      </div>

      {/* KPI Cards (Live Recalculation) */}
      <div className="kpi-grid">
        <div className="kpi-card highlight">
          <div className="kpi-label">Active Budget Cap</div>
          <div className="kpi-value text-accent">${budget.toLocaleString()}</div>
          <div className="kpi-subtext">Total Spent: ${totalSpend.toLocaleString()}</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Customers Prescribed</div>
          <div className="kpi-value">{targetedCount.toLocaleString()}</div>
          <div className="kpi-subtext">Out of {rawCustomers.length.toLocaleString()} total candidates</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Expected Incremental Conversions</div>
          <div className="kpi-value text-success">+{totalExpectedConversions.toFixed(4)}</div>
          <div className="kpi-subtext">Net conversion gain</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Tier Breakdown ($5 / $10 / $20)</div>
          <div className="kpi-value text-accent">
            {tierBreakdown.low} Low &bull; {tierBreakdown.medium} Mid &bull; {tierBreakdown.high} High
          </div>
          <div className="kpi-subtext">Multipliers: $5 (1.0x), $10 (1.3x), $20 (1.6x)</div>
        </div>
      </div>

      {/* STEP 4: Budget Slider & Number Input with Live Recalculation */}
      <div className="card control-panel" style={{ padding: '1.25rem 1.5rem', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <label className="control-label" style={{ fontSize: '1.05rem', fontWeight: 600 }}>
                Dynamic Budget Slider ($1,000 – $20,000):
              </label>
              <p className="chart-desc" style={{ margin: '0.2rem 0 0 0' }}>
                Adjust slider or type budget to instantly re-rank &amp; re-slice targeted customers live client-side.
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontWeight: 600, color: '#94a3b8' }}>$</span>
              <input
                type="number"
                min="1000"
                max="20000"
                step="500"
                value={budget}
                onChange={(e) => handleBudgetChange(e.target.value)}
                className="number-input"
                style={{ width: '120px', fontSize: '1rem', padding: '0.4rem 0.6rem', textAlign: 'right' }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontSize: '0.85rem', color: '#94a3b8', fontWeight: 500 }}>$1,000</span>
            <input
              type="range"
              min="1000"
              max="20000"
              step="500"
              value={budget}
              onChange={(e) => handleBudgetChange(e.target.value)}
              className="styled-slider"
              style={{ flex: 1, cursor: 'pointer' }}
            />
            <span style={{ fontSize: '0.85rem', color: '#94a3b8', fontWeight: 500 }}>$20,000</span>
          </div>

          <div className="budget-preset-buttons" style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            {[1000, 2500, 5000, 7500, 10000, 15000, 20000].map((preset) => (
              <button
                key={preset}
                className={`btn-preset ${budget === preset ? 'active' : ''}`}
                onClick={() => handleBudgetChange(preset)}
              >
                ${(preset / 1000).toFixed(preset % 1000 === 0 ? 0 : 1)}k
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* STEP 3: Full Allocation Matrix UI (Paginated 25/page, Sortable Click Headers) */}
      <div className="card table-card">
        <div className="table-header-controls">
          <div>
            <h3>Customer Prescriptions ({filteredRows.length.toLocaleString()} matching)</h3>
            <p className="chart-desc">Click headers to sort ascending / descending &bull; 25 customers per page</p>
          </div>

          <div className="table-filters">
            <input
              type="text"
              placeholder="Search segment, rank, tier..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setPage(1);
              }}
              className="search-input"
            />
            <div className="filter-buttons">
              <button
                className={`btn-filter ${filterAllocation === 'allocated' ? 'active' : ''}`}
                onClick={() => {
                  setFilterAllocation('allocated');
                  setPage(1);
                }}
              >
                Targeted ({targetedCount.toLocaleString()})
              </button>
              <button
                className={`btn-filter ${filterAllocation === 'all' ? 'active' : ''}`}
                onClick={() => {
                  setFilterAllocation('all');
                  setPage(1);
                }}
              >
                All Candidates
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
                <th onClick={() => handleSort('rank')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                  Rank {sortField === 'rank' ? (sortDirection === 'asc' ? '▲' : '▼') : '↕'}
                </th>
                <th onClick={() => handleSort('ite')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                  Customer ITE Score {sortField === 'ite' ? (sortDirection === 'asc' ? '▲' : '▼') : '↕'}
                </th>
                <th onClick={() => handleSort('discount_tier')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                  Assigned Discount Tier {sortField === 'discount_tier' ? (sortDirection === 'asc' ? '▲' : '▼') : '↕'}
                </th>
                <th onClick={() => handleSort('expected_incremental_conversion')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                  Expected Incremental Conversion {sortField === 'expected_incremental_conversion' ? (sortDirection === 'asc' ? '▲' : '▼') : '↕'}
                </th>
                <th onClick={() => handleSort('history')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                  Past Spend ($) {sortField === 'history' ? (sortDirection === 'asc' ? '▲' : '▼') : '↕'}
                </th>
                <th onClick={() => handleSort('segment')} style={{ cursor: 'pointer', userSelect: 'none' }}>
                  Segment {sortField === 'segment' ? (sortDirection === 'asc' ? '▲' : '▼') : '↕'}
                </th>
              </tr>
            </thead>
            <tbody>
              {paginatedRows.length === 0 ? (
                <tr>
                  <td colSpan="6" style={{ textAlign: 'center', padding: '2rem' }}>
                    No matching customer prescriptions found.
                  </td>
                </tr>
              ) : (
                paginatedRows.map((row) => (
                  <tr key={row.rank} className={row.isAllocated ? 'row-allocated' : ''}>
                    <td className="font-mono">#{row.rank}</td>
                    <td className="font-mono text-accent">
                      +{(row.ite * 100).toFixed(3)}%
                    </td>
                    <td>
                      {row.isAllocated ? (
                        <span className={`badge ${row.discount_tier === 'high' ? 'badge-success' : row.discount_tier === 'low' ? 'badge-accent' : 'badge-primary'}`}>
                          ${row.tier_cost} ({row.discount_tier.toUpperCase()}) &bull; {row.ite_multiplier}x
                        </span>
                      ) : (
                        <span className="badge badge-muted">Do Not Target</span>
                      )}
                    </td>
                    <td className="font-mono text-success font-semibold">
                      +{row.expected_incremental_conversion.toFixed(5)}
                    </td>
                    <td className="font-mono">${(row.history || 0).toFixed(2)}</td>
                    <td>
                      <span className="segment-tag">{row.segment || 'Customer'}</span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* STEP 3: Pagination Controls (Next/Prev, 25 rows/page) */}
        <div className="pagination-bar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem' }}>
          <span className="pagination-info">
            Showing {(page - 1) * pageSize + 1}&ndash;{Math.min(page * pageSize, sortedRows.length)} of {sortedRows.length.toLocaleString()} customers
          </span>
          <div className="pagination-buttons" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
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
