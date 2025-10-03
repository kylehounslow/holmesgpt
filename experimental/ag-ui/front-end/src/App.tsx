import React, { useState, useEffect } from 'react';
import './App.css';
import MainContent from './components/MainContent';
import ChatAssistant from './components/ChatAssistant';
import ErrorBoundary from './components/ErrorBoundary';

export type ObservabilityPage = 'metrics' | 'logs' | 'traces';

const App: React.FC = () => {
  const [selectedPage, setSelectedPage] = useState<ObservabilityPage>('metrics');
  const [initialQuery, setInitialQuery] = useState<string>('');

  // Read URL parameters on mount
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const appParam = urlParams.get('app');
    const queryParam = urlParams.get('query');

    // Set page from URL parameter or default to metrics
    if (appParam && ['metrics', 'logs', 'traces'].includes(appParam)) {
      setSelectedPage(appParam as ObservabilityPage);
    } else {
      // Default to metrics and update URL
      setSelectedPage('metrics');
      urlParams.set('app', 'metrics');
      const newUrl = `${window.location.pathname}?${urlParams.toString()}`;
      window.history.replaceState({}, '', newUrl);
    }

    // Set initial query from URL parameter
    if (queryParam) {
      setInitialQuery(decodeURIComponent(queryParam));
    }
  }, []);

  // Update URL when page changes
  const handlePageChange = (page: ObservabilityPage) => {
    setSelectedPage(page);
    
    // Update URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    urlParams.set('app', page);
    
    // Preserve existing query parameter if present
    const newUrl = `${window.location.pathname}?${urlParams.toString()}`;
    window.history.replaceState({}, '', newUrl);
  };

  return (
    <div className="app">
      <div className="left-sidebar">
        <div className="sidebar-header">
          <h2>HolmesGPT</h2>
        </div>
        <nav className="sidebar-nav">
          <button
            className={`nav-item ${selectedPage === 'metrics' ? 'active' : ''}`}
            onClick={() => handlePageChange('metrics')}
          >
            <span className="nav-icon">📊</span>
            Metrics
          </button>
          <button
            className={`nav-item ${selectedPage === 'logs' ? 'active' : ''}`}
            onClick={() => handlePageChange('logs')}
          >
            <span className="nav-icon">📝</span>
            Logs
          </button>
          <button
            className={`nav-item ${selectedPage === 'traces' ? 'active' : ''}`}
            onClick={() => handlePageChange('traces')}
          >
            <span className="nav-icon">🔍</span>
            Traces
          </button>
        </nav>
      </div>
      <MainContent selectedPage={selectedPage} initialQuery={initialQuery} />
      <ErrorBoundary>
        <ChatAssistant />
      </ErrorBoundary>
    </div>
  );
};

export default App;