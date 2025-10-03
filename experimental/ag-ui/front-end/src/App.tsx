import React from 'react';
import './App.css';
import MainContent from './components/MainContent';
import ChatAssistant from './components/ChatAssistant';
import ErrorBoundary from './components/ErrorBoundary';

const App: React.FC = () => {
  return (
    <div className="app">
      <MainContent />
      <ErrorBoundary>
        <ChatAssistant />
      </ErrorBoundary>
    </div>
  );
};

export default App;