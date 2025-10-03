import React, { useState } from 'react';
import GraphVisualization from './GraphVisualization';

interface QueryResult {
  id: string;
  query: string;
  timestamp: Date;
  data?: any;
  error?: string;
}

const MainContent: React.FC = () => {
  const [query, setQuery] = useState('');
  const [queryResults, setQueryResults] = useState<QueryResult[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);

  const handleExecuteQuery = async () => {
    if (!query.trim() || isExecuting) return;

    const newResult: QueryResult = {
      id: `query-${Date.now()}`,
      query: query.trim(),
      timestamp: new Date(),
    };

    setIsExecuting(true);
    setQueryResults(prev => [newResult, ...prev]);

    try {
      // Simulate query execution - in real implementation, this would call your backend
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data for demonstration - replace with actual API call
      const mockData = {
        title: query,
        data: {
          result: [
            {
              metric: { __name__: query, instance: "localhost:9090" },
              values: Array.from({ length: 20 }, (_, i) => [
                Date.now() / 1000 - (20 - i) * 60,
                (Math.random() * 100).toFixed(2)
              ])
            }
          ]
        },
        query: query,
        metadata: {
          timeRange: "1h",
          step: "1m"
        }
      };

      setQueryResults(prev => 
        prev.map(result => 
          result.id === newResult.id 
            ? { ...result, data: mockData }
            : result
        )
      );
    } catch (error) {
      setQueryResults(prev => 
        prev.map(result => 
          result.id === newResult.id 
            ? { ...result, error: 'Failed to execute query' }
            : result
        )
      );
    } finally {
      setIsExecuting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleExecuteQuery();
    }
  };

  const clearResults = () => {
    setQueryResults([]);
  };

  return (
    <div className="observability-platform">
      <div className="platform-header">
        <div className="header-content">
          <h1>HolmesGPT Observability</h1>
          <p>Query and visualize your metrics, logs, and traces</p>
        </div>
      </div>

      <div className="query-section">
        <div className="query-input-container">
          <label htmlFor="query-input" className="query-label">
            Query
          </label>
          <div className="query-input-wrapper">
            <textarea
              id="query-input"
              className="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter your query (e.g., cpu_usage, memory_usage, http_requests_total)..."
              rows={3}
            />
            <div className="query-actions">
              <button
                className="execute-button"
                onClick={handleExecuteQuery}
                disabled={!query.trim() || isExecuting}
              >
                {isExecuting ? 'Executing...' : 'Execute'}
              </button>
              {queryResults.length > 0 && (
                <button
                  className="clear-button"
                  onClick={clearResults}
                >
                  Clear Results
                </button>
              )}
            </div>
          </div>
          <div className="query-hint">
            Press Cmd/Ctrl + Enter to execute
          </div>
        </div>
      </div>

      <div className="results-section">
        {queryResults.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <h3>No queries executed yet</h3>
            <p>Enter a query above and click Execute to see visualizations</p>
          </div>
        ) : (
          <div className="results-list">
            {queryResults.map((result) => (
              <div key={result.id} className="result-item">
                <div className="result-header">
                  <div className="result-info">
                    <h4 className="result-query">{result.query}</h4>
                    <span className="result-timestamp">
                      {result.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="result-status">
                    {result.error ? (
                      <span className="status-error">Error</span>
                    ) : result.data ? (
                      <span className="status-success">Success</span>
                    ) : (
                      <span className="status-loading">Loading...</span>
                    )}
                  </div>
                </div>
                
                <div className="result-content">
                  {result.error ? (
                    <div className="error-message">
                      <span className="error-icon">⚠️</span>
                      {result.error}
                    </div>
                  ) : result.data ? (
                    <div className="graph-container">
                      <GraphVisualization data={result.data} />
                    </div>
                  ) : (
                    <div className="loading-placeholder">
                      <div className="loading-spinner"></div>
                      <span>Executing query...</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default MainContent;