import React, { useState, useRef, useEffect } from 'react';
import { HttpAgent } from '@ag-ui/client';
import ReactMarkdown from 'react-markdown';
import GraphVisualization from './GraphVisualization';
import './ChatAssistant.css';

interface ChatMessage {
  id: string;
  text?: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  type?: 'text' | 'graph' | 'error';
  graphData?: any;
  error?: {
    title: string;
    description: string;
    retryable?: boolean;
  };
}

type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';

const ChatAssistant: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome-1',
      text: "Hello! I'm your AI assistant. How can I help you today?",
      sender: 'assistant',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [width, setWidth] = useState(350);
  const [isResizing, setIsResizing] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [isThinking, setIsThinking] = useState(false);
  const [messageHistory, setMessageHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [retryCount, setRetryCount] = useState(0);
  const [lastFailedMessage, setLastFailedMessage] = useState<string | null>(null);
  const agentRef = useRef<HttpAgent | null>(null);
  const threadIdRef = useRef<string>('thread-' + Date.now());
  const currentMessageRef = useRef<string>('');
  const toolCallsRef = useRef<Map<string, { name: string, args?: any }>>(new Map());
  const chatAssistantRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize the HttpAgent with error handling
  const initializeAgent = React.useCallback(() => {
    try {
      setConnectionStatus('connecting');
      
      agentRef.current = new HttpAgent({
        url: process.env.REACT_APP_AGENT_URL || 'http://localhost:5050/api/agui/chat',
        threadId: threadIdRef.current,
        headers: {
          'Content-Type': 'application/json',
        }
      });

      // Set up event subscriber with error handling
      const subscriber: Object = {
      onRunStartedEvent: (params: { event: any; }) => {
        setIsLoading(true);
        setIsThinking(true);
        setConnectionStatus('connected');
        setRetryCount(0);
        console.log('Agent run started:', params.event);
      },

      onRunFinishedEvent: (params: { event: any; }) => {
        setIsLoading(false);
        setIsThinking(false);
        setConnectionStatus('connected');
        console.log('Agent run finished:', params.event);
      },

      onTextMessageStartEvent: (params: { event: { messageId: any; }; }) => {
        currentMessageRef.current = '';
        const newMessage: ChatMessage = {
          id: params.event.messageId,
          text: '',
          sender: 'assistant',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, newMessage]);
      },

      onTextMessageContentEvent: (params: { event: { delta: string; messageId: string; }; }) => {
        currentMessageRef.current += params.event.delta;
        setMessages(prev => 
          prev.map(msg => 
            msg.id === params.event.messageId 
              ? { ...msg, text: currentMessageRef.current }
              : msg
          )
        );
      },

      onTextMessageEndEvent: () => {
        setIsLoading(false);
      },

      onToolCallStartEvent: (params: { event: { toolCallName: string; toolCallId: string; }; }) => {
        console.log('Tool call started:', params.event);
        
        // Store tool call metadata for later use
        toolCallsRef.current.set(params.event.toolCallId, { 
          name: params.event.toolCallName 
        });
        
        if (params.event.toolCallName === 'graph_timeseries_data') {
          const toolMessage: ChatMessage = {
            id: 'tool-' + params.event.toolCallId,
            text: '🔧 Preparing to display graph...',
            sender: 'assistant',
            timestamp: new Date()
          };
          setMessages(prev => [...prev, toolMessage]);
        }
      },

      onToolCallEndEvent: (params: { event: { toolCallId: string; }; toolCallArgs: any; }) => {
        console.log('Tool call ended:', params.event, params.toolCallArgs);
        
        // Retrieve tool call info from our stored metadata
        const toolCallInfo = toolCallsRef.current.get(params.event.toolCallId);
        
        if (toolCallInfo?.name === 'graph_timeseries_data') {
          try {
            // toolCallArgs is already an object, no need to parse
            const args = params.toolCallArgs || {};
            console.log('Graph tool args:', args);
            
            // Handle the case where data might be a JSON string
            let processedArgs = { ...args };
            if (typeof args.data === 'string') {
              try {
                processedArgs.data = JSON.parse(args.data);
              } catch (parseError) {
                console.warn('Could not parse data as JSON, using as-is:', parseError);
              }
            }
            
            // Create a graph visualization message
            const graphMessage: ChatMessage = {
              id: 'graph-' + params.event.toolCallId,
              sender: 'assistant',
              timestamp: new Date(),
              type: 'graph',
              graphData: processedArgs
            };
            
            setMessages(prev => 
              prev.map(msg => 
                msg.id === 'tool-' + params.event.toolCallId 
                  ? graphMessage 
                  : msg
              )
            );
          } catch (error) {
            console.error('Error processing tool arguments:', error);
          }
        }
        
        // Clean up the stored tool call info
        toolCallsRef.current.delete(params.event.toolCallId);
      },

      onRunErrorEvent: (params: { event: { message: any; }; }) => {
        setIsLoading(false);
        setIsThinking(false);
        setConnectionStatus('error');
        
        const errorMessage: ChatMessage = {
          id: 'error-' + Date.now(),
          sender: 'assistant',
          timestamp: new Date(),
          type: 'error',
          error: {
            title: 'Connection Error',
            description: typeof params.event.message === 'string' 
              ? params.event.message 
              : 'Failed to communicate with the AI service',
            retryable: true
          }
        };
        setMessages(prev => [...prev, errorMessage]);
        
        // Schedule reconnection attempt
        scheduleReconnect();
      },

      // Add network error handler
      onNetworkError: (error: any) => {
        console.error('Network error:', error);
        setIsLoading(false);
        setIsThinking(false);
        setConnectionStatus('disconnected');
        
        const errorMessage: ChatMessage = {
          id: 'network-error-' + Date.now(),
          sender: 'assistant',
          timestamp: new Date(),
          type: 'error',
          error: {
            title: 'Network Error',
            description: 'Unable to connect to the AI service. Please check your internet connection.',
            retryable: true
          }
        };
        setMessages(prev => [...prev, errorMessage]);
        
        scheduleReconnect();
      }
    };

      agentRef.current.subscribe(subscriber);
      setConnectionStatus('connected');
      
    } catch (error) {
      console.error('Failed to initialize agent:', error);
      setConnectionStatus('error');
      scheduleReconnect();
    }
  }, []);

  // Connection health check
  const checkConnection = React.useCallback(async () => {
    try {
      const baseUrl = process.env.REACT_APP_AGENT_URL || 'http://localhost:5050/api/agui/chat';
      const healthUrl = baseUrl.replace('/chat', '/chat/health');
      
      const response = await fetch(healthUrl, {
        method: 'GET',
        timeout: 5000
      } as any);
      return response.ok;
    } catch {
      return false;
    }
  }, []);

  // Reconnection logic
  const scheduleReconnect = React.useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    const delay = Math.min(1000 * Math.pow(2, retryCount), 30000); // Exponential backoff, max 30s
    
    reconnectTimeoutRef.current = setTimeout(async () => {
      // Check if server is reachable before attempting reconnection
      const isServerReachable = await checkConnection();
      
      if (isServerReachable) {
        setRetryCount(prev => prev + 1);
        initializeAgent();
      } else {
        // Server still not reachable, schedule another check
        setConnectionStatus('disconnected');
        scheduleReconnect();
      }
    }, delay);
  }, [retryCount, initializeAgent, checkConnection]);

  // Global error handlers to prevent crashes
  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      console.error('Unhandled promise rejection:', event.reason);
      event.preventDefault(); // Prevent the default browser behavior
      
      // Handle network errors specifically
      if (event.reason?.message?.includes('network error') || 
          event.reason?.message?.includes('ERR_INCOMPLETE_CHUNKED_ENCODING')) {
        setIsLoading(false);
        setIsThinking(false);
        setConnectionStatus('error');
        
        const errorMessage: ChatMessage = {
          id: 'unhandled-error-' + Date.now(),
          sender: 'assistant',
          timestamp: new Date(),
          type: 'error',
          error: {
            title: 'Connection Lost',
            description: 'Lost connection to AI service. Attempting to reconnect...',
            retryable: true
          }
        };
        setMessages(prev => [...prev, errorMessage]);
        scheduleReconnect();
      }
    };

    const handleError = (event: ErrorEvent) => {
      console.error('Global error:', event.error);
      // Don't prevent default for general errors, just log them
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    window.addEventListener('error', handleError);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
      window.removeEventListener('error', handleError);
    };
  }, [scheduleReconnect]);

  // Initialize agent on mount
  useEffect(() => {
    initializeAgent();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [initializeAgent]);

  // Auto-scroll functionality
  useEffect(() => {
    if (autoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScroll]);

  // Resize functionality
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsResizing(true);
    e.preventDefault();
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      
      const newWidth = window.innerWidth - e.clientX;
      const minWidth = 250;
      const maxWidth = 600;
      
      if (newWidth >= minWidth && newWidth <= maxWidth) {
        setWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    if (isResizing) {
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  const handleSendMessage = async (messageText?: string) => {
    const textToSend = messageText || inputValue.trim();
    if (!textToSend || !agentRef.current || isLoading) return;

    const userMessage: ChatMessage = {
      id: 'user-' + Date.now(),
      text: textToSend,
      sender: 'user',
      timestamp: new Date()
    };
    
    // Add to message history if it's not already the last message
    if (messageHistory[messageHistory.length - 1] !== textToSend) {
      setMessageHistory(prev => [...prev, textToSend]);
    }
    
    setMessages(prev => [...prev, userMessage]);
    if (!messageText) {
      setInputValue('');
    }
    setHistoryIndex(-1);
    setIsLoading(true);
    setLastFailedMessage(null);

    // Wrap everything in a Promise to catch all possible rejections
    const sendOperation = async () => {
      try {
        // Check connection status
        if (connectionStatus === 'disconnected' || connectionStatus === 'error') {
          throw new Error('Not connected to AI service');
        }

        // Add the user message to the agent's message history
        if (agentRef.current) {
          agentRef.current.addMessage({
            id: userMessage.id,
            role: 'user',
            content: userMessage.text || ''
          });

          // Wrap runAgent in a timeout to prevent hanging
          const runAgentPromise = agentRef.current.runAgent({
            runId: 'run-' + Date.now(),
            tools: [
              {
                name: 'graph_timeseries_data',
                description: 'Display time series data as an interactive graph. Use this when you have prometheus data or any time series data that should be visualized.',
                parameters: {
                  type: 'object',
                  properties: {
                    title: {
                      type: 'string',
                      description: 'Title for the graph'
                    },
                    data: {
                      type: 'object',
                      description: 'Prometheus-style data with result array containing metric and values'
                    },
                    query: {
                      type: 'string',
                      description: 'The original query used to generate this data'
                    },
                    metadata: {
                      type: 'object',
                      description: 'Additional metadata like time range, step, etc.'
                    }
                  },
                  required: ['title', 'data']
                }
              }
            ],
            context: []
          });

          // Add timeout to prevent hanging requests
          const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Request timeout')), 30000);
          });

          await Promise.race([runAgentPromise, timeoutPromise]);
        }
      } catch (error: any) {
        console.error('Send message error:', error);
        setIsLoading(false);
        setIsThinking(false);
        setConnectionStatus('error');
        setLastFailedMessage(textToSend);
        
        // Determine error type for better user messaging
        let errorTitle = 'Message Failed';
        let errorDescription = 'Failed to send message to AI service';
        
        if (error.message?.includes('network error') || 
            error.message?.includes('ERR_INCOMPLETE_CHUNKED_ENCODING')) {
          errorTitle = 'Network Error';
          errorDescription = 'Connection interrupted. The AI service may be temporarily unavailable.';
        } else if (error.message?.includes('timeout')) {
          errorTitle = 'Request Timeout';
          errorDescription = 'The request took too long to complete. Please try again.';
        } else if (error.message?.includes('Not connected')) {
          errorTitle = 'Connection Error';
          errorDescription = 'Not connected to AI service. Attempting to reconnect...';
        }
        
        const errorMessage: ChatMessage = {
          id: 'send-error-' + Date.now(),
          sender: 'assistant',
          timestamp: new Date(),
          type: 'error',
          error: {
            title: errorTitle,
            description: errorDescription,
            retryable: true
          }
        };
        setMessages(prev => [...prev, errorMessage]);
        
        // Try to reconnect
        scheduleReconnect();
      }
    };

    // Execute with additional error boundary
    try {
      await sendOperation();
    } catch (error) {
      // This should catch any remaining unhandled errors
      console.error('Outer catch - unexpected error:', error);
      setIsLoading(false);
      setIsThinking(false);
      setConnectionStatus('error');
    }
  };

  const retryLastMessage = () => {
    if (lastFailedMessage) {
      handleSendMessage(lastFailedMessage);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSendMessage();
      return;
    }

    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (messageHistory.length === 0) return;
      
      const newIndex = historyIndex === -1 
        ? messageHistory.length - 1 
        : Math.max(0, historyIndex - 1);
      
      setHistoryIndex(newIndex);
      setInputValue(messageHistory[newIndex]);
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex === -1) return;
      
      const newIndex = historyIndex + 1;
      if (newIndex >= messageHistory.length) {
        setHistoryIndex(-1);
        setInputValue('');
      } else {
        setHistoryIndex(newIndex);
        setInputValue(messageHistory[newIndex]);
      }
      return;
    }
  };

  return (
    <div 
      ref={chatAssistantRef}
      className="chat-assistant" 
      style={{ width: `${width}px` }}
    >
      <div 
        className={`resize-handle ${isResizing ? 'resizing' : ''}`}
        onMouseDown={handleMouseDown}
      />
      <div className="chat-header">
        <h3>AI Assistant</h3>
        <div className="connection-status">
          <div className={`connection-indicator ${connectionStatus}`}></div>
          <span>
            {connectionStatus === 'connected' && 'Connected'}
            {connectionStatus === 'connecting' && 'Connecting...'}
            {connectionStatus === 'disconnected' && 'Disconnected'}
            {connectionStatus === 'error' && 'Connection Error'}
          </span>
        </div>
      </div>
      
      <div className="chat-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender === 'user' ? 'user-message' : 'assistant-message'}`}
          >
            {message.type === 'graph' && message.graphData ? (
              <GraphVisualization data={message.graphData} />
            ) : message.type === 'error' && message.error ? (
              <div className="error-message">
                <div className="error-title">{message.error.title}</div>
                <div className="error-description">{message.error.description}</div>
                {message.error.retryable && (
                  <button 
                    className="retry-message-button"
                    onClick={retryLastMessage}
                    disabled={isLoading}
                  >
                    Retry Last Message
                  </button>
                )}
              </div>
            ) : (
              <div className="message-text">
                <ReactMarkdown>{message.text || ''}</ReactMarkdown>
              </div>
            )}
            <div className="message-time">
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        ))}
        {isThinking && (
          <div className="thinking-indicator">
            <div className="thinking-message">
              <div className="thinking-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
              <div className="thinking-text">AI is thinking...</div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chat-controls">
        <label className="auto-scroll-checkbox">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          Auto-scroll
        </label>
      </div>
      
      <div className="chat-input">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => {
            setInputValue(e.target.value);
            // Reset history navigation when user starts typing
            if (historyIndex !== -1) {
              setHistoryIndex(-1);
            }
          }}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button 
          onClick={() => handleSendMessage()} 
          disabled={isLoading || !inputValue.trim() || connectionStatus === 'disconnected'}
        >
          {isLoading ? 'Sending...' : connectionStatus === 'disconnected' ? 'Disconnected' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default ChatAssistant;