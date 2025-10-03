import React, { useState } from 'react';
import GraphVisualization from './GraphVisualization';
import LogsVisualization from './LogsVisualization';
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
  onQueryUpdate?: (page: ObservabilityPage, query: string) => void;
}

const MainContent: React.FC<MainContentProps> = ({ 
  selectedPage, 
  initialQuery = '', 
  triggerQuery,
  onContextChange,
  onQueryTriggered,
  onQueryUpdate
}) => {
  const [query, setQuery] = useState(initialQuery);
  const [isExecuting, setIsExecuting] = useState(false);
  
  // Track current query execution to prevent race conditions
  const currentQueryRef = React.useRef<string>('');
  const abortControllerRef = React.useRef<AbortController | null>(null);
  
  // Store separate results for each page
  const [pageResults, setPageResults] = useState<Record<ObservabilityPage, QueryResult | null>>({
    metrics: null,
    logs: null,
    traces: null
  });
  
  // Get current page's result
  const currentResult = pageResults[selectedPage];
  const [prometheusStatus, setPrometheusStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [prometheusUrl] = useState(process.env.REACT_APP_PROMETHEUS_URL || 'http://localhost:9090');
  const [opensearchStatus, setOpensearchStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [opensearchUrl] = useState(process.env.REACT_APP_OPENSEARCH_URL || 'http://localhost:9200');
  const [opensearchUser] = useState(process.env.REACT_APP_OPENSEARCH_USER);
  const [opensearchPassword] = useState(process.env.REACT_APP_OPENSEARCH_PASSWORD);
  
  // Indices discovery state
  const [availableIndices, setAvailableIndices] = useState<string[]>([]);
  const [showIndices, setShowIndices] = useState(false);
  const [loadingIndices, setLoadingIndices] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);

  // Helper function to create OpenSearch auth headers
  const getOpensearchHeaders = React.useCallback(() => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (opensearchUser && opensearchPassword) {
      const credentials = btoa(`${opensearchUser}:${opensearchPassword}`);
      headers['Authorization'] = `Basic ${credentials}`;
    }

    return headers;
  }, [opensearchUser, opensearchPassword]);

  // Fetch indices count automatically when connected
  const fetchIndicesCount = React.useCallback(async () => {
    if (selectedPage !== 'logs' || opensearchStatus !== 'connected') return;

    try {
      const response = await fetch(`${opensearchUrl}/_cat/indices?format=json&h=index`, {
        method: 'GET',
        headers: getOpensearchHeaders(),
        signal: AbortSignal.timeout(10000), // 10 second timeout
      });

      if (response.ok) {
        const indices = await response.json();
        const indexNames = indices
          .map((idx: any) => idx.index)
          .filter((name: string) => !name.startsWith('.')) // Filter out system indices
          .sort();
        
        setAvailableIndices(indexNames);
      } else {
        console.error('Failed to fetch indices count:', response.status, response.statusText);
        setAvailableIndices([]);
      }
    } catch (error) {
      console.error('Error fetching indices count:', error);
      setAvailableIndices([]);
    }
  }, [opensearchUrl, selectedPage, opensearchStatus, getOpensearchHeaders]);

  // Toggle available indices display
  const toggleAvailableIndices = React.useCallback(async () => {
    if (selectedPage !== 'logs' || opensearchStatus !== 'connected') return;

    // If indices are currently shown, just hide them
    if (showIndices) {
      setShowIndices(false);
      return;
    }

    // If we already have indices cached, just show them
    if (availableIndices.length > 0) {
      setShowIndices(true);
      return;
    }

    // Otherwise, fetch indices from OpenSearch (this shouldn't happen often now)
    setLoadingIndices(true);
    try {
      await fetchIndicesCount();
      setShowIndices(true);
    } finally {
      setLoadingIndices(false);
    }
  }, [selectedPage, opensearchStatus, showIndices, availableIndices.length, fetchIndicesCount]);

  const isUpdatingFromParent = React.useRef(false);
  
  // Retry delay state
  const prometheusRetryTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);
  const opensearchRetryTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);

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
        let errorValue = currentResult.error;
        
        // For logs, include detailed error response in the same entry
        if (selectedPage === 'logs' && currentResult.errorDetails) {
          errorValue += `\n\nDetailed error response: ${JSON.stringify(currentResult.errorDetails, null, 2)}`;
        }
        
        context.push({
          description: `${selectedPage} query error`,
          value: errorValue
        });
        
        // Add detailed error response for non-logs pages only
        if (selectedPage !== 'logs' && currentResult.errorDetails) {
          context.push({
            description: `${selectedPage} error response`,
            value: JSON.stringify(currentResult.errorDetails)
          });
        }
      } else if (currentResult.data && selectedPage !== 'logs') {
        // Only add success status for non-logs pages
        context.push({
          description: `${selectedPage} query status`,
          value: "Success - data available for visualization"
        });
      }
    }

    // Add connection status for metrics page only
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

  // Auto-fetch indices count when OpenSearch connection becomes healthy
  React.useEffect(() => {
    if (selectedPage === 'logs') {
      if (opensearchStatus === 'connected' && availableIndices.length === 0) {
        fetchIndicesCount();
      } else if (opensearchStatus === 'disconnected') {
        // Clear indices when connection is lost
        setAvailableIndices([]);
        setShowIndices(false);
      }
    }
  }, [selectedPage, opensearchStatus, availableIndices.length, fetchIndicesCount]);

  // Check Prometheus connection status
  const checkPrometheusConnection = React.useCallback(async (isRetry = false) => {
    if (selectedPage !== 'metrics') {
      setPrometheusStatus('connected'); // Don't check for non-metrics pages
      return;
    }

    // Prevent multiple concurrent checks
    if (prometheusStatus === 'checking' && !isRetry) {
      console.log('Prometheus check already in progress, skipping');
      return;
    }

    // Clear any existing retry timeout
    if (prometheusRetryTimeoutRef.current) {
      clearTimeout(prometheusRetryTimeoutRef.current);
      prometheusRetryTimeoutRef.current = null;
    }

    // Add delay for retries to prevent overwhelming the server
    if (isRetry) {
      await new Promise(resolve => setTimeout(resolve, 2000)); // 2 second delay for retries
    }

    try {
      setPrometheusStatus('checking');
      console.log('Checking Prometheus connection...', prometheusUrl);
      const response = await fetch(`${prometheusUrl}/api/v1/label/__name__/values?limit=1`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000), // 5 second timeout
      });
      
      console.log('Prometheus response:', response.status, response.ok);
      if (response.ok) {
        console.log('Setting Prometheus status to connected');
        setPrometheusStatus('connected');
      } else {
        console.log('Setting Prometheus status to disconnected');
        setPrometheusStatus('disconnected');
      }
    } catch (error) {
      console.warn('Prometheus connection check failed:', error);
      setPrometheusStatus('disconnected');
    }
  }, [prometheusUrl, selectedPage]);

  // Check OpenSearch connection status
  const checkOpensearchConnection = React.useCallback(async (isRetry = false) => {
    if (selectedPage !== 'logs') {
      setOpensearchStatus('connected'); // Don't check for non-logs pages
      return;
    }

    // Clear any existing retry timeout
    if (opensearchRetryTimeoutRef.current) {
      clearTimeout(opensearchRetryTimeoutRef.current);
      opensearchRetryTimeoutRef.current = null;
    }

    // Add delay for retries to prevent overwhelming the server
    if (isRetry) {
      await new Promise(resolve => setTimeout(resolve, 2000)); // 2 second delay for retries
    }

    try {
      setOpensearchStatus('checking');
      // Try basic cluster info first, then health endpoint
      let response = await fetch(`${opensearchUrl}/`, {
        method: 'GET',
        headers: getOpensearchHeaders(),
        signal: AbortSignal.timeout(5000), // 5 second timeout
      });
      
      // If root endpoint fails, try cluster health
      if (!response.ok) {
        response = await fetch(`${opensearchUrl}/_cluster/health`, {
          method: 'GET',
          headers: getOpensearchHeaders(),
          signal: AbortSignal.timeout(5000), // 5 second timeout
        });
      }
      
      if (response.ok) {
        setOpensearchStatus('connected');
      } else {
        setOpensearchStatus('disconnected');
      }
    } catch (error) {
      console.warn('OpenSearch connection check failed:', error);
      setOpensearchStatus('disconnected');
    }
  }, [opensearchUrl, selectedPage, getOpensearchHeaders]);

  // Check connection on mount and when page changes
  React.useEffect(() => {
    // Initial connection checks
    if (selectedPage === 'metrics') {
      checkPrometheusConnection();
    } else if (selectedPage === 'logs') {
      checkOpensearchConnection();
    }
    
    // Set up interval for active page only
    let interval: NodeJS.Timeout | null = null;
    if (selectedPage === 'metrics') {
      interval = setInterval(() => checkPrometheusConnection(), 30000);
    } else if (selectedPage === 'logs') {
      interval = setInterval(() => checkOpensearchConnection(), 30000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [selectedPage]); // Only depend on selectedPage

  // Update query when initialQuery changes
  React.useEffect(() => {
    if (initialQuery !== undefined) {
      isUpdatingFromParent.current = true;
      setQuery(initialQuery);
      // Reset flag after state update
      setTimeout(() => {
        isUpdatingFromParent.current = false;
      }, 0);
    }
  }, [initialQuery]); // Only depend on initialQuery, not query

  // Notify parent when query changes (but only from user input)
  React.useEffect(() => {
    if (onQueryUpdate && !isUpdatingFromParent.current) {
      onQueryUpdate(selectedPage, query);
    }
  }, [query, selectedPage]); // Remove onQueryUpdate from dependencies to prevent loop

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
    const prometheusUrl = process.env.REACT_APP_PROMETHEUS_URL || 'http://localhost:9090';
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

  const queryOpensearch = async (pplQuery: string, signal?: AbortSignal) => {
    const opensearchUrl = process.env.REACT_APP_OPENSEARCH_URL || 'http://localhost:9200';

    try {
      // First try PPL endpoint
      const response = await fetch(`${opensearchUrl}/_plugins/_ppl`, {
        method: 'POST',
        headers: getOpensearchHeaders(),
        body: JSON.stringify({
          query: pplQuery
        }),
        signal: signal
      });

      const result = await response.json();

      if (!response.ok) {
        // Check if it's a PPL plugin not available error (404 or 500)
        if (response.status === 404 || response.status === 500) {
          const error = new Error(`PPL plugin not available on this OpenSearch cluster (${response.status}). This AWS OpenSearch cluster may not have PPL enabled.`);
          (error as any).responseData = { 
            suggestion: "AWS OpenSearch clusters may not have PPL plugin enabled by default. Consider using OpenSearch Query DSL or enabling PPL plugin.",
            pplQuery: pplQuery,
            alternativeEndpoint: `${opensearchUrl}/_search`,
            statusCode: response.status
          };
          throw error;
        }
        
        // Create detailed error with response data
        const error = new Error(`OpenSearch query failed: ${response.status} ${response.statusText}`);
        (error as any).responseData = result;
        (error as any).statusCode = response.status;
        throw error;
      }
      
      if (result.error) {
        // Handle specific PPL errors
        if (result.error.type === 'NoSuchElementException') {
          const error = new Error(`PPL query error: No data found. Check your index name and query syntax.`);
          (error as any).responseData = result;
          (error as any).errorType = result.error.type;
          (error as any).suggestion = "Try: source=your_actual_index_name | head 10 (replace 'your_actual_index_name' with an actual index)";
          throw error;
        }
        
        // Create detailed error with OpenSearch error response
        const error = new Error(`OpenSearch PPL error: ${result.error.reason || 'Unknown error'}`);
        (error as any).responseData = result;
        (error as any).errorType = result.error.type;
        throw error;
      }

      return {
        title: "Logs Visualization",
        data: result,
        query: pplQuery,
        metadata: {
          timeRange: "query-dependent",
          source: "OpenSearch PPL",
          resultType: "logs"
        }
      };
    } catch (error) {
      console.error('OpenSearch query error:', error);
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
    setPageResults(prev => ({
      ...prev,
      [selectedPage]: newResult
    }));

    try {
      let responseData: any;
      
      if (selectedPage === 'metrics') {
        // Query Prometheus for metrics
        responseData = await queryPrometheus(query.trim());
      } else if (selectedPage === 'logs') {
        // Query OpenSearch for logs
        responseData = await queryOpensearch(query.trim());
      } else {
        // For traces, use mock data for now
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

      setPageResults(prev => ({
        ...prev,
        [selectedPage]: prev[selectedPage] ? { ...prev[selectedPage], data: responseData } : null
      }));
    } catch (error: any) {
      console.error('Query execution error:', error);
      const errorMessage = selectedPage === 'metrics' 
        ? `Prometheus query failed: ${error.message || 'Unknown error'}`
        : `${selectedPage} query failed: ${error.message || 'Unknown error'}`;
        
      setPageResults(prev => ({
        ...prev,
        [selectedPage]: prev[selectedPage] ? { 
          ...prev[selectedPage], 
          error: errorMessage,
          errorDetails: error.responseData || null
        } : null
      }));
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
    setPageResults(prev => ({
      ...prev,
      [selectedPage]: null
    }));
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

  // Cleanup timeout refs on unmount
  React.useEffect(() => {
    return () => {
      if (prometheusRetryTimeoutRef.current) {
        clearTimeout(prometheusRetryTimeoutRef.current);
      }
      if (opensearchRetryTimeoutRef.current) {
        clearTimeout(opensearchRetryTimeoutRef.current);
      }
    };
  }, []);

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
              onClick={() => checkPrometheusConnection(true)}
            >
              Retry
            </button>
          )}
        </div>
      )}

      {selectedPage === 'logs' && (
        <div className="connection-status-bar">
          <div className="connection-info">
            <span className="connection-label">OpenSearch:</span>
            <span className="connection-url">{opensearchUrl}</span>
            <div className={`connection-indicator ${opensearchStatus}`}>
              <span className="status-dot"></span>
              <span className="status-text">
                {opensearchStatus === 'checking' && 'Checking...'}
                {opensearchStatus === 'connected' && 'Connected'}
                {opensearchStatus === 'disconnected' && 'Disconnected'}
              </span>
            </div>
          </div>
          {opensearchStatus === 'disconnected' && (
            <button 
              className="retry-connection-btn"
              onClick={() => checkOpensearchConnection(true)}
            >
              Retry
            </button>
          )}
        </div>
      )}

      {/* Indices Discovery Section - Only show for logs page when connected */}
      {selectedPage === 'logs' && opensearchStatus === 'connected' && (
        <div className="indices-discovery-section">
          <button 
            className="show-indices-btn"
            onClick={toggleAvailableIndices}
            disabled={loadingIndices}
          >
            {loadingIndices ? (
              <>
                <span className="loading-spinner"></span>
                Loading Indices...
              </>
            ) : showIndices ? (
              <>
                📂 Hide Indices
                {availableIndices.length > 0 && (
                  <span className="indices-count-badge">{availableIndices.length}</span>
                )}
                <span className="dropdown-arrow up">▲</span>
              </>
            ) : (
              <>
                📂 {availableIndices.length > 0 ? `Show ${availableIndices.length} Indices` : 'Show Available Indices'}
                <span className="dropdown-arrow">▼</span>
              </>
            )}
          </button>
          <div className="indices-discovery-header">
            <p>
              {availableIndices.length > 0 
                ? `Found ${availableIndices.length} ${availableIndices.length === 1 ? 'index' : 'indices'} in your OpenSearch cluster`
                : 'Discover what indices are available in your OpenSearch cluster'
              }
            </p>
          </div>
        </div>
      )}

      {/* Available Indices Display */}
      {selectedPage === 'logs' && showIndices && availableIndices.length > 0 && (
        <div className="indices-display">
          <div className="indices-header">
            <h4>Select an Index</h4>
            <button 
              className="close-indices-btn"
              onClick={() => setShowIndices(false)}
            >
              ✕
            </button>
          </div>
          <div className="indices-help">
            💡 Click on an index name to use it in a PPL query
          </div>
          <div className="indices-list">
            {availableIndices.map((index, i) => (
              <div 
                key={i} 
                className="index-item"
                onClick={() => {
                  setQuery(`source=${index} | head 10`);
                  setShowIndices(false);
                }}
                title={`Click to use in query: source=${index} | head 10`}
              >
                {index}
              </div>
            ))}
          </div>
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
                  ? "Enter PPL query (e.g., source=logs-* | head 10) - Note: PPL plugin may not be available on all AWS clusters..."
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
                <div className="visualization-container">
                  {/* Detect data type and render appropriate visualization */}
                  {/* Check if data is already structured (has title, data, query) or raw (has schema, datarows) */}
                  {(currentResult.data.schema && currentResult.data.datarows) || 
                   (currentResult.data.data && currentResult.data.data.schema && currentResult.data.data.datarows) ? (
                    <LogsVisualization 
                      data={
                        currentResult.data.title ? 
                          // Data is already structured
                          currentResult.data :
                          // Data is raw, need to structure it
                          {
                            title: selectedPage === 'logs' ? 'Logs Visualization' : 'Data Visualization',
                            query: currentResult.query,
                            data: currentResult.data,
                            metadata: {
                              timestamp: Date.now() / 1000,
                              source: 'OpenSearch PPL'
                            }
                          }
                      } 
                    />
                  ) : (currentResult.data.result !== undefined) || 
                       (currentResult.data.data && currentResult.data.data.result !== undefined) ? (
                    <GraphVisualization 
                      data={
                        currentResult.data.title ? 
                          // Data is already structured
                          currentResult.data :
                          // Data is raw, need to structure it
                          {
                            title: selectedPage === 'metrics' ? 'Metrics Visualization' : 'Data Visualization',
                            query: currentResult.query,
                            data: currentResult.data,
                            metadata: {
                              timestamp: Date.now() / 1000,
                              source: 'Prometheus'
                            }
                          }
                      }
                    />
                  ) : (
                    <div className="unsupported-data">
                      <span className="error-icon">⚠️</span>
                      <div className="error-text">Unsupported data format</div>
                      <div className="error-details">
                        <pre>{JSON.stringify(currentResult.data, null, 2)}</pre>
                      </div>
                    </div>
                  )}
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
              {/* Detect data type and render appropriate visualization */}
              {(currentResult.data.schema && currentResult.data.datarows) || 
               (currentResult.data.data && currentResult.data.data.schema && currentResult.data.data.datarows) ? (
                <LogsVisualization 
                  data={
                    currentResult.data.title ? 
                      // Data is already structured
                      currentResult.data :
                      // Data is raw, need to structure it
                      {
                        title: selectedPage === 'logs' ? 'Logs Visualization' : 'Data Visualization',
                        query: currentResult.query,
                        data: currentResult.data,
                        metadata: {
                          timestamp: Date.now() / 1000,
                          source: 'OpenSearch PPL'
                        }
                      }
                  } 
                />
              ) : (currentResult.data.result !== undefined) || 
                   (currentResult.data.data && currentResult.data.data.result !== undefined) ? (
                <GraphVisualization 
                  data={
                    currentResult.data.title ? 
                      // Data is already structured
                      currentResult.data :
                      // Data is raw, need to structure it
                      {
                        title: selectedPage === 'metrics' ? 'Metrics Visualization' : 'Data Visualization',
                        query: currentResult.query,
                        data: currentResult.data,
                        metadata: {
                          timestamp: Date.now() / 1000,
                          source: 'Prometheus'
                        }
                      }
                  }
                />
              ) : (
                <div className="unsupported-data">
                  <span className="error-icon">⚠️</span>
                  <div className="error-text">Unsupported data format</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MainContent;