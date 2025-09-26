import React from 'react';
import Diagnostics from './Diagnostics';

const MainContent: React.FC = () => {
  return (
    <div className="main-content">
      <h1>Welcome to AG UI App</h1>
      <p>This is the main content area. You can add your application content here.</p>
      
      <Diagnostics />
      
      <div style={{ marginTop: '2rem' }}>
        <h2>Features</h2>
        <ul>
          <li>Responsive layout with chat assistant</li>
          <li>TypeScript support</li>
          <li>Modern React setup</li>
          <li>AG-UI Protocol diagnostics panel</li>
          <li>Real-time event monitoring</li>
        </ul>
      </div>
    </div>
  );
};

export default MainContent;