import React, { useState } from 'react';
import GraphVisualization from './GraphVisualization';
type ObservabilityPage = 'metrics' | 'logs' | 'traces';

interface QueryResult {
  id: string;
  query: string;
  timestamp: Date;
  data?: any;
  error?: string;
  errorDetails?: any;
}

interface ContextItem {
  description: string;
  value: string;
}

interface MainContentProps {
  selectedPage: ObservabilityPage;
  initialQuery?: string;
  triggerQuery?: string | null;
  onContextChange?: (context: ContextItem[]) => void;
  onQueryTriggered?: () => void;
}

const MainContent: React.FC<MainContentProps> = ({ 
  selectedPage, 
  initialQuery = '', 
  triggerQuery,
  onContextChange,
  onQueryTriggered 
}) => {
  const [query, setQuery] = useState(initialQuery);
  const [currentResult, setCurrentResult] = useState<QueryResult | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [prometheusStatus, setPrometheusStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [prometheusUrl] = useState(process.env.PROMETHEUS_URL || 'http://localhost:9090');
  const [isMaximized, setIsMaximized] = useState(false);

  // Update context for ChatAssistant
  const updateContext = React.useCallback(() => {
    if (!onContextChange) return;

    const context: ContextItem[] = [];
    
    // Add current page info
    context.push({
      description: "Current page",
      value: selectedPage
    });

    // Add current query if exists
    if (query.trim()) {
      context.push({
        description: `Current ${selectedPage} query`,
        value: query.trim()
      });
    }

    // Add current result info if exists
    if (currentResult) {
      if (currentResult.error) {
        context.push({
          description: `${selectedPage} query error`,
          value: currentResult.error
        });
        
        // Add detailed error response if available
        if (currentResult.errorDetails) {
          context.push({
            description: `${selectedPage} error response`,
            value: JSON.stringify(currentResult.errorDetails)
          });
        }
      } else if (currentResult.data) {
        context.push({
          description: `${selectedPage} query status`,
          value: "Success - data available for visualization"
        });
      }
    }

    // Add Prometheus connection status for metrics page
    if (selectedPage === 'metrics') {
      context.push({
        description: "Prometheus connection status",
        value: `${prometheusStatus} (${prometheusUrl})`
      });
    }

    onContextChange(context);
  }, [selectedPage, query, currentResult, prometheusStatus, prometheusUrl, onContextChange]);

  // Update context whenever relevant state changes
  React.useEffect(() => {
    updateContext();
  }, [updateContext]);

  // Check Prometheus connection status
  const checkPrometheusConnection = React.useCallback(async () => {
    if (selectedPage !== 'metrics') {
      setPrometheusStatus('connected'); // Don't check for non-metrics pages
      return;
    }

    try {
      setPrometheusStatus('checking');
      const response = await fetch(`${prometheusUrl}/api/v1/label/__name__/values?limit=1`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000), // 5 second timeout
      });
      
      if (response.ok) {
        setPrometheusStatus('connected');
      } else {
        setPrometheusStatus('disconnected');
      }
    } catch (error) {
      console.warn('Prometheus connection check failed:', error);
      setPrometheusStatus('disconnected');
    }
  }, [prometheusUrl, selectedPage]);

  // Check connection on mount and when page changes
  React.useEffect(() => {
    checkPrometheusConnection();
    
    // Check connection every 30 seconds for metrics page
    if (selectedPage === 'metrics') {
      const interval = setInterval(checkPrometheusConnection, 30000);
      return () => clearInterval(interval);
    }
  }, [checkPrometheusConnection, selectedPage]);

  // Update query when initialQuery changes
  React.useEffect(() => {
    if (initialQuery && initialQuery !== query) {
      setQuery(initialQuery);
    }
  }, [initialQuery]);

  // Update URL when query changes (debounced)
  React.useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (query.trim()) {
        const urlParams = new URLSearchParams(window.location.search);
        urlParams.set('query', encodeURIComponent(query.trim()));
        const newUrl = `${window.location.pathname}?${urlParams.toString()}`;
        window.history.replaceState({}, '', newUrl);
      } else {
        // Remove query parameter if empty
        const urlParams = new URLSearchParams(window.location.search);
        urlParams.delete('query');
        const newUrl = urlParams.toString() 
          ? `${window.location.pathname}?${urlParams.toString()}`
          : window.location.pathname;
        window.history.replaceState({}, '', newUrl);
      }
    }, 500); // 500ms debounce

    return () => clearTimeout(timeoutId);
  }, [query]);

  const queryPrometheus = async (promqlQuery: string) => {
    const prometheusUrl = process.env.PROMETHEUS_URL || 'http://localhost:9090';
    const endTime = Math.floor(Date.now() / 1000);
    const startTime = endTime - 3600; // 1 hour ago
    const step = 60; // 1 minute step

    try {
      const url = `${prometheusUrl}/api/v1/query_range?query=${encodeURIComponent(promqlQuery)}&start=${startTime}&end=${endTime}&step=${step}`;
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });

      const result = await response.json();

      if (!response.ok) {
        // Create detailed error with response data
        const error = new Error(`Prometheus query failed: ${response.status} ${response.statusText}`);
        (error as any).responseData = result;
        (error as any).statusCode = response.status;
        throw error;
      }
      
      if (result.status !== 'success') {
        // Create detailed error with Prometheus error response
        const error = new Error(`Prometheus query error: ${result.error || 'Unknown error'}`);
        (error as any).responseData = result;
        (error as any).errorType = result.errorType;
        throw error;
      }

      return {
        title: "Metrics Visualization",
        data: result.data,
        query: promqlQuery,
        metadata: {
          timeRange: "1h",
          step: "1m",
          resultType: result.data.resultType
        }
      };
    } catch (error) {
      console.error('Prometheus query error:', error);
      throw error;
    }
  };

  const handleExecuteQuery = async () => {
    if (!query.trim() || isExecuting) return;

    const newResult: QueryResult = {
      id: `query-${Date.now()}`,
      query: query.trim(),
      timestamp: new Date(),
    };

    setIsExecuting(true);
    setCurrentResult(newResult);

    try {
      let responseData: any;
      
      if (selectedPage === 'metrics') {
        // Query Prometheus for metrics
        responseData = await queryPrometheus(query.trim());
      } else {
        // For logs and traces, use mock data for now
        responseData = {
          title: `${selectedPage.charAt(0).toUpperCase() + selectedPage.slice(1)} Visualization`,
          data: {
            result: [
              {
                metric: { __name__: query, service: selectedPage },
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
            step: "1m",
            type: selectedPage
          }
        };
      }

      setCurrentResult(prev => prev ? { ...prev, data: responseData } : null);
    } catch (error: any) {
      console.error('Query execution error:', error);
      const errorMessage = selectedPage === 'metrics' 
        ? `Prometheus query failed: ${error.message || 'Unknown error'}`
        : `${selectedPage} query failed: ${error.message || 'Unknown error'}`;
        
      setCurrentResult(prev => prev ? { 
        ...prev, 
        error: errorMessage,
        errorDetails: error.responseData || null
      } : null);
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
    setCurrentResult(null);
  };

  // Handle trigger query execution from ChatAssistant
  React.useEffect(() => {
    if (triggerQuery && triggerQuery.trim()) {
      setQuery(triggerQuery);
      // Execute the query automatically after a short delay to ensure state is updated
      setTimeout(() => {
        handleExecuteQuery();
        if (onQueryTriggered) {
          onQueryTriggered();
        }
      }, 100);
    }
  }, [triggerQuery, onQueryTriggered]);

  // Handle keyboard shortcuts for modal
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMaximized) {
        setIsMaximized(false);
      }
    };

    if (isMaximized) {
      document.addEventListener('keydown', handleKeyDown);
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isMaximized]);

  return (
    <div className="observability-platform">
      <div className="platform-header">
        <div className="header-content">
          <h1>ExampleOps Platform - {selectedPage.charAt(0).toUpperCase() + selectedPage.slice(1)}</h1>
          <p>
            {selectedPage === 'metrics' && 'Query and visualize your application metrics and performance data'}
            {selectedPage === 'logs' && 'Search and analyze your application logs and events'}
            {selectedPage === 'traces' && 'Explore distributed traces and request flows'}
          </p>
        </div>
      </div>

      {selectedPage === 'metrics' && (
        <div className="connection-status-bar">
          <div className="connection-info">
            <span className="connection-label">Prometheus:</span>
            <span className="connection-url">{prometheusUrl}</span>
            <div className={`connection-indicator ${prometheusStatus}`}>
              <span className="status-dot"></span>
              <span className="status-text">
                {prometheusStatus === 'checking' && 'Checking...'}
                {prometheusStatus === 'connected' && 'Connected'}
                {prometheusStatus === 'disconnected' && 'Disconnected'}
              </span>
            </div>
          </div>
          {prometheusStatus === 'disconnected' && (
            <button 
              className="retry-connection-btn"
              onClick={checkPrometheusConnection}
            >
              Retry
            </button>
          )}
        </div>
      )}

      <div className="query-section">
        <div className="query-input-container">
          <label htmlFor="query-input" className="query-label">
            {selectedPage === 'metrics' && 'Metrics Query'}
            {selectedPage === 'logs' && 'Log Query'}
            {selectedPage === 'traces' && 'Trace Query'}
          </label>
          <div className="query-input-wrapper">
            <textarea
              id="query-input"
              className="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                selectedPage === 'metrics' 
                  ? "Enter PromQL query (e.g., cpu_usage, memory_usage, http_requests_total)..."
                  : selectedPage === 'logs'
                  ? "Enter log search query (e.g., level:error, service:api, message:timeout)..."
                  : "Enter trace query (e.g., service:checkout, operation:payment, duration:>1s)..."
              }
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
              {currentResult && (
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
        {!currentResult ? (
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <h3>No queries executed yet</h3>
            <p>Enter a query above and click Execute to see visualizations</p>
          </div>
        ) : (
          <div className="result-item">
            <div className="result-header">
              <div className="result-info">
                <h4 className="result-title">
                  {selectedPage === 'metrics' ? '📊 Metrics Result' : 
                   selectedPage === 'logs' ? '📝 Logs Result' : 
                   '🔍 Traces Result'}
                </h4>
                <span className="result-timestamp">
                  {currentResult.timestamp.toLocaleTimeString()}
                </span>
              </div>
              <div className="result-actions">
                <div className="result-status">
                  {currentResult.error ? (
                    <span className="status-error">Error</span>
                  ) : currentResult.data ? (
                    <span className="status-success">Success</span>
                  ) : (
                    <span className="status-loading">Loading...</span>
                  )}
                </div>
                {currentResult.data && !currentResult.error && (
                  <button
                    className="maximize-button"
                    onClick={() => setIsMaximized(true)}
                    title="Maximize visualization"
                  >
                    ⛶
                  </button>
                )}
              </div>
            </div>
            
            <div className="result-content">
              {currentResult.error ? (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  <div className="error-text">{currentResult.error}</div>
                  {currentResult.errorDetails && (
                    <div className="error-details">
                      <div className="error-details-label">Response Details:</div>
                      <div className="error-response-container">
                        <pre className="error-response">
                          {JSON.stringify(currentResult.errorDetails, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              ) : currentResult.data ? (
                <div className="graph-container">
                  <GraphVisualization data={currentResult.data} />
                </div>
              ) : (
                <div className="loading-placeholder">
                  <div className="loading-spinner"></div>
                  <span>Executing query...</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Maximized Graph Modal */}
      {isMaximized && currentResult?.data && (
        <div className="graph-modal-overlay" onClick={() => setIsMaximized(false)}>
          <div className="graph-modal" onClick={(e) => e.stopPropagation()}>
            <div className="graph-modal-header">
              <div className="graph-modal-title">
                <h3>
                  {selectedPage === 'metrics' ? '📊 Metrics Visualization' : 
                   selectedPage === 'logs' ? '📝 Logs Visualization' : 
                   '🔍 Traces Visualization'}
                </h3>
                <div className="graph-modal-query">
                  Query: <code>{currentResult.query}</code>
                </div>
              </div>
              <button
                className="close-modal-button"
                onClick={() => setIsMaximized(false)}
                title="Close maximized view"
              >
                ✕
              </button>
            </div>
            <div className="graph-modal-content">
              <GraphVisualization data={currentResult.data} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MainContent;