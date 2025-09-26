import React, { useState, useEffect, useRef } from 'react';
import { HttpAgent } from '@ag-ui/client';
import './Diagnostics.css';

interface FlowEvent {
  id: string;
  timestamp: Date;
  type: string;
  data: any;
  category: 'run' | 'message' | 'tool';
  phase: 'start' | 'content' | 'end' | 'error';
}

interface FlowStats {
  totalEvents: number;
  activeRuns: number;
  activeMessages: number;
  activeTools: number;
  lastEventTime: Date | null;
}

const Diagnostics: React.FC = () => {
  const [events, setEvents] = useState<FlowEvent[]>([]);
  const [stats, setStats] = useState<FlowStats>({
    totalEvents: 0,
    activeRuns: 0,
    activeMessages: 0,
    activeTools: 0,
    lastEventTime: null
  });
  const [isExpanded, setIsExpanded] = useState(true);
  const [maxEvents, setMaxEvents] = useState(100);
  const [showContentEvents, setShowContentEvents] = useState(false);
  const agentRef = useRef<HttpAgent | null>(null);

  useEffect(() => {
    // Initialize diagnostic agent (separate from chat)
    agentRef.current = new HttpAgent({
      url: process.env.REACT_APP_AGENT_URL || 'http://localhost:5050/api/agui/chat',
      threadId: 'diagnostics-' + Date.now(),
      headers: {
        'Content-Type': 'application/json',
      }
    });

    const addFlowEvent = (type: string, data: any, category: 'run' | 'message' | 'tool', phase: 'start' | 'content' | 'end' | 'error') => {
      // Skip content events if not enabled (they're too noisy)
      if (phase === 'content' && !showContentEvents) return;

      const flowEvent: FlowEvent = {
        id: Date.now() + '-' + Math.random(),
        timestamp: new Date(),
        type,
        data,
        category,
        phase
      };
      
      setEvents(prev => {
        const newEvents = [flowEvent, ...prev];
        return newEvents.slice(0, maxEvents);
      });
      
      setStats(prev => {
        const newStats = { ...prev };
        newStats.totalEvents = prev.totalEvents + 1;
        newStats.lastEventTime = new Date();
        
        // Track active states
        if (phase === 'start') {
          if (category === 'run') newStats.activeRuns++;
          if (category === 'message') newStats.activeMessages++;
          if (category === 'tool') newStats.activeTools++;
        } else if (phase === 'end' || phase === 'error') {
          if (category === 'run') newStats.activeRuns = Math.max(0, newStats.activeRuns - 1);
          if (category === 'message') newStats.activeMessages = Math.max(0, newStats.activeMessages - 1);
          if (category === 'tool') newStats.activeTools = Math.max(0, newStats.activeTools - 1);
        }
        
        return newStats;
      });
    };

    // Set up flow event subscriber
    const subscriber = {
      onRunStartedEvent: (params: { event: any }) => {
        addFlowEvent('RUN_STARTED', { runId: params.event.runId }, 'run', 'start');
      },

      onRunFinishedEvent: (params: { event: any }) => {
        addFlowEvent('RUN_FINISHED', { runId: params.event.runId }, 'run', 'end');
      },

      onRunErrorEvent: (params: { event: any }) => {
        addFlowEvent('RUN_ERROR', { runId: params.event.runId, message: params.event.message }, 'run', 'error');
      },

      onTextMessageStartEvent: (params: { event: any }) => {
        addFlowEvent('TEXT_MESSAGE_START', { messageId: params.event.messageId }, 'message', 'start');
      },

      onTextMessageContentEvent: (params: { event: any }) => {
        addFlowEvent('TEXT_MESSAGE_CONTENT', { 
          messageId: params.event.messageId, 
          deltaLength: params.event.delta?.length || 0,
          delta: params.event.delta?.substring(0, 50) + (params.event.delta?.length > 50 ? '...' : '')
        }, 'message', 'content');
      },

      onTextMessageEndEvent: (params: { event: any }) => {
        addFlowEvent('TEXT_MESSAGE_END', { messageId: params.event.messageId }, 'message', 'end');
      },

      onToolCallStartEvent: (params: { event: any }) => {
        addFlowEvent('TOOL_CALL_START', { 
          toolCallId: params.event.toolCallId,
          toolName: params.event.toolCallName
        }, 'tool', 'start');
      },

      onToolCallEndEvent: (params: { event: any, toolCallArgs: any }) => {
        addFlowEvent('TOOL_CALL_END', { 
          toolCallId: params.event.toolCallId,
          argsSize: JSON.stringify(params.toolCallArgs || {}).length,
          args: params.toolCallArgs
        }, 'tool', 'end');
      }
    };

    agentRef.current.subscribe(subscriber);

    return () => {
      // Cleanup if needed
    };
  }, [maxEvents, showContentEvents]);

  const clearEvents = () => {
    setEvents([]);
    setStats({
      totalEvents: 0,
      activeRuns: 0,
      activeMessages: 0,
      activeTools: 0,
      lastEventTime: null
    });
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'run': return '🏃';
      case 'message': return '💬';
      case 'tool': return '🔧';
      default: return '📋';
    }
  };

  const getPhaseColor = (phase: string) => {
    switch (phase) {
      case 'start': return '#17a2b8';
      case 'content': return '#6c757d';
      case 'end': return '#28a745';
      case 'error': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const formatEventData = (data: any) => {
    if (typeof data === 'object') {
      return JSON.stringify(data, null, 2);
    }
    return String(data);
  };

  return (
    <div className="diagnostics-panel">
      <div className="diagnostics-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="diagnostics-title">
          <span className="flow-icon">🔄</span>
          AG-UI Protocol Flow
          <span className="diagnostics-toggle">{isExpanded ? '▼' : '▶'}</span>
        </div>
        <div className="flow-summary">
          <span className="flow-stat">🏃 Runs: {stats.activeRuns}</span>
          <span className="flow-stat">💬 Messages: {stats.activeMessages}</span>
          <span className="flow-stat">🔧 Tools: {stats.activeTools}</span>
          <span className="flow-stat">📊 Total: {stats.totalEvents}</span>
        </div>
      </div>

      {isExpanded && (
        <div className="diagnostics-content">
          <div className="flow-controls">
            <div className="flow-options">
              <label className="flow-checkbox">
                <input
                  type="checkbox"
                  checked={showContentEvents}
                  onChange={(e) => setShowContentEvents(e.target.checked)}
                />
                Show content events
              </label>
            </div>
            <div className="flow-actions">
              <button onClick={clearEvents} className="clear-btn">Clear Flow</button>
              <select 
                value={maxEvents} 
                onChange={(e) => setMaxEvents(Number(e.target.value))}
                className="max-events-select"
              >
                <option value={50}>50 events</option>
                <option value={100}>100 events</option>
                <option value={200}>200 events</option>
                <option value={500}>500 events</option>
              </select>
            </div>
          </div>

          <div className="flow-events">
            {events.length === 0 ? (
              <div className="no-events">
                <div className="flow-waiting">Waiting for AG-UI protocol events...</div>
                <div className="flow-hint">Send a message in the chat to see the data flow</div>
              </div>
            ) : (
              events.map((event, index) => (
                <div key={event.id} className={`flow-event ${event.phase}`}>
                  <div className="flow-event-header">
                    <span className="flow-category">{getCategoryIcon(event.category)}</span>
                    <span className="flow-type">{event.type}</span>
                    <span className="flow-phase" style={{ color: getPhaseColor(event.phase) }}>
                      {event.phase.toUpperCase()}
                    </span>
                    <span className="flow-timestamp">
                      {event.timestamp.toLocaleTimeString()}.{event.timestamp.getMilliseconds().toString().padStart(3, '0')}
                    </span>
                  </div>
                  <div className="flow-data">
                    <pre>{formatEventData(event.data)}</pre>
                  </div>
                  {index < events.length - 1 && <div className="flow-connector">↓</div>}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Diagnostics;