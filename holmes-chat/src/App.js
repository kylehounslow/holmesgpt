import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import './App.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [tasks, setTasks] = useState([]);
  const [maximizedGraph, setMaximizedGraph] = useState(null);
  const [messageHistory, setMessageHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const threadIdRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to parse and render graphs from message content
  const parseAndRenderContent = (content) => {
    // Look for GRAPH_DATA, TASK_UPDATE, and Holmes promql embedding markers
    const graphDataRegex = /📊 \*\*GRAPH_DATA:\*\* ```json\n([\s\S]*?)\n```/g;
    const taskUpdateRegex = /📋 \*\*TASK_UPDATE:\*\* ```json\n([\s\S]*?)\n```/g;
    const promqlEmbedRegex = /<<\s*(\{[^}]*"type"\s*:\s*"promql"[^}]*\})\s*>>/g;
    
    const parts = [];
    let lastIndex = 0;
    const allMatches = [];
    
    // Find graph matches
    let graphMatch;
    while ((graphMatch = graphDataRegex.exec(content)) !== null) {
      allMatches.push({ ...graphMatch, type: 'graph' });
    }
    
    // Find task matches
    let taskMatch;
    while ((taskMatch = taskUpdateRegex.exec(content)) !== null) {
      allMatches.push({ ...taskMatch, type: 'task' });
    }
    
    // Find Holmes promql embedding matches
    let promqlMatch;
    while ((promqlMatch = promqlEmbedRegex.exec(content)) !== null) {
      allMatches.push({ ...promqlMatch, type: 'promql_embed' });
    }
    
    // Sort by index
    allMatches.sort((a, b) => a.index - b.index);
    
    // Process matches
    for (const match of allMatches) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push({
          type: 'text',
          content: content.slice(lastIndex, match.index)
        });
      }

      // Parse and add data
      try {
        if (match.type === 'promql_embed') {
          // Handle Holmes promql embedding format
          const embedData = JSON.parse(match[1]);
          const randomKey = embedData.random_key;
          
          if (randomKey) {
            // Try to fetch real data from server
            fetch(`http://localhost:5051/api/prometheus-data/${randomKey}`)
              .then(response => response.json())
              .then(realGraphData => {
                // Update the part with real data
                const partIndex = parts.length;
                parts.push({
                  type: 'graph',
                  data: realGraphData
                });
                // Force re-render by updating a dummy state or using a callback
              })
              .catch(error => {
                console.error('Failed to fetch Prometheus data:', error);
                // Fallback to placeholder
                parts.push({
                  type: 'graph',
                  data: {
                    type: 'prometheus_graph',
                    tool_name: embedData.tool_name || 'Prometheus Query',
                    query: `${embedData.tool_name} (${randomKey}) - Data not available`,
                    data: { result: [] }
                  }
                });
              });
          } else {
            // No random key, use placeholder
            parts.push({
              type: 'graph',
              data: {
                type: 'prometheus_graph',
                tool_name: embedData.tool_name || 'Prometheus Query',
                query: `${embedData.tool_name} - No key available`,
                data: { result: [] }
              }
            });
          }
        } else {
          const data = JSON.parse(match[1]);
          if (match.type === 'graph' && data.type === 'prometheus_graph') {
            parts.push({
              type: 'graph',
              data: data
            });
          } else if (match.type === 'task' && data.type === 'task_update') {
            // Update global tasks state
            setTasks(data.tasks || []);
            // Don't add to parts - tasks are shown in separate panel
          }
        }
      } catch (e) {
        console.error('Error parsing data:', e);
        parts.push({
          type: 'text',
          content: match[0]
        });
      }

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < content.length) {
      parts.push({
        type: 'text',
        content: content.slice(lastIndex)
      });
    }

    return parts.length > 0 ? parts : [{ type: 'text', content }];
  };

  // Function to render a Prometheus graph
  const renderPrometheusGraph = (graphData) => {
    const { data, query, metadata } = graphData;
    
    if (!data.result || !Array.isArray(data.result)) {
      return <div>No graph data available</div>;
    }

    const datasets = data.result.map((series, index) => {
      const label = Object.entries(series.metric || {})
        .map(([key, value]) => `${key}=${value}`)
        .join(', ') || `Series ${index + 1}`;

      const dataPoints = (series.values || []).map(([timestamp, value]) => ({
        x: new Date(timestamp * 1000),
        y: parseFloat(value)
      }));

      return {
        label,
        data: dataPoints,
        borderColor: `hsl(${index * 137.5 % 360}, 70%, 50%)`,
        backgroundColor: `hsla(${index * 137.5 % 360}, 70%, 50%, 0.1)`,
        tension: 0.1
      };
    });

    const chartData = {
      datasets
    };

    // Get date range for title
    const timestamps = data.result.flatMap(series => 
      (series.values || []).map(([timestamp]) => timestamp * 1000)
    );
    const minDate = timestamps.length > 0 ? new Date(Math.min(...timestamps)) : new Date();
    const maxDate = timestamps.length > 0 ? new Date(Math.max(...timestamps)) : new Date();
    
    const dateRange = minDate.toDateString() === maxDate.toDateString() 
      ? minDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      : `${minDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${maxDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          align: 'start',
          maxHeight: 80,
          labels: {
            usePointStyle: true,
            pointStyle: 'rect',
            boxWidth: 12,
            boxHeight: 12,
            padding: 8,
            textAlign: 'left'
          },
          onClick: (e, legendItem, legend) => {
            const chart = legend.chart;
            const index = legendItem.datasetIndex;
            
            if (e.metaKey || e.ctrlKey) {
              // CMD/Ctrl+click: Show only this dataset, hide all others
              chart.data.datasets.forEach((dataset, i) => {
                const meta = chart.getDatasetMeta(i);
                meta.hidden = i !== index;
              });
            } else {
              // Regular click: Toggle this dataset
              const meta = chart.getDatasetMeta(index);
              meta.hidden = meta.hidden === null ? !chart.data.datasets[index].hidden : null;
            }
            
            chart.update();
          }
        },
        tooltip: {
          callbacks: {
            title: function(context) {
              return new Date(context[0].parsed.x).toLocaleString();
            },
            label: function(context) {
              const label = context.dataset.label || '';
              const value = context.parsed.y;
              
              // If label is longer than 50 characters, organize vertically
              if (label.length > 50) {
                const parts = label.split(', ');
                return [`Value: ${value}`, ...parts];
              }
              
              return `${label}: ${value}`;
            }
          },
          displayColors: true,
          multiKeyBackground: '#fff'
        },
        title: {
          display: true,
          text: query || 'Prometheus Query Result'
        },
      },
      scales: {
        x: {
          type: 'time',
          time: {
            displayFormats: {
              minute: 'HH:mm',
              hour: 'HH:mm',
              day: 'HH:mm'
            },
            tooltipFormat: 'MMM dd, yyyy HH:mm:ss z'
          },
          ticks: {
            callback: function(value, index) {
              const date = new Date(value);
              const time = date.toLocaleString('en-US', {
                hour: '2-digit',
                minute: '2-digit'
              });
              
              // Show date only on first tick
              if (index === 0) {
                const dateStr = date.toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric'
                });
                return [time, dateStr];
              }
              
              return time;
            },
            maxRotation: 0
          },
          title: {
            display: true,
            text: `Time (${new Date().toLocaleString('en-US', { timeZoneName: 'short' }).split(', ')[1].split(' ').pop()})`
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Value'
          }
        }
      }
    };

    // Use stable key based on data content, not random
    const stableKey = `chart-${JSON.stringify(data.result).slice(0, 50)}`;

    // Extract the actual Prometheus query from the graph data
    const prometheusQuery = graphData.query || query || '';
    const prometheusUrl = `http://localhost:9090/query?g0.expr=${encodeURIComponent(prometheusQuery)}&g0.show_tree=0&g0.tab=graph&g0.range_input=1h&g0.res_type=auto&g0.res_density=medium&g0.display_mode=lines&g0.show_exemplars=0`;

    return (
      <div 
        style={{ 
          margin: '10px 0', 
          padding: '10px', 
          backgroundColor: 'white', 
          borderRadius: '8px',
          border: '1px solid #ddd',
          height: '400px',
          width: '750px',
          minWidth: '500px',
          maxWidth: '100%',
          position: 'relative',
          overflow: 'hidden',
          boxSizing: 'border-box',
          display: 'block'
        }}
      >
        {/* Prometheus UI Link Button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            window.open(prometheusUrl, '_blank');
          }}
          style={{
            position: 'absolute',
            top: '15px',
            right: '15px',
            background: '#e6522c',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            padding: '8px 12px',
            fontSize: '13px',
            cursor: 'pointer',
            zIndex: 10,
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            e.target.style.background = '#d63031';
            e.target.style.transform = 'scale(1.05)';
          }}
          onMouseLeave={(e) => {
            e.target.style.background = '#e6522c';
            e.target.style.transform = 'scale(1)';
          }}
          title="Open query in Prometheus UI (new window)"
        >
          🔥 Open in Prometheus
        </button>
        
        {/* Clickable chart area */}
        <div
          style={{
            height: '100%',
            cursor: 'pointer'
          }}
          onClick={(e) => {
            // Don't maximize if clicking on legend area
            if (e.target.closest('canvas')) {
              setMaximizedGraph({ data: chartData, options, title: query || 'Prometheus Query Result', dateRange, prometheusUrl });
            }
          }}
          title="Click to maximize"
        >
          <Line 
            key={stableKey}
            data={chartData} 
            options={options}
          />
        </div>
      </div>
    );
  };

  // Generate thread_id on first load
  useEffect(() => {
    threadIdRef.current = `thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  // Handle ESC key to close maximized graph
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && maximizedGraph) {
        setMaximizedGraph(null);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [maximizedGraph]);

  // Handle keyboard navigation for message history
  const handleKeyDown = (e) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (messageHistory.length > 0) {
        const newIndex = historyIndex === -1 ? messageHistory.length - 1 : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setInput(messageHistory[newIndex]);
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex >= 0) {
        const newIndex = historyIndex + 1;
        if (newIndex >= messageHistory.length) {
          setHistoryIndex(-1);
          setInput('');
        } else {
          setHistoryIndex(newIndex);
          setInput(messageHistory[newIndex]);
        }
      }
    } else if (e.key === 'Enter') {
      sendMessage();
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    
    // Add to message history
    setMessageHistory(prev => {
      const newHistory = [...prev, userMessage];
      // Keep only last 50 messages
      return newHistory.slice(-50);
    });
    setHistoryIndex(-1);
    
    setInput('');
    setIsLoading(true);

    // Add user message to chat
    setMessages((prev) => [...prev, { text: userMessage, sender: 'user' }]);

    try {
      // Prepare AG-UI compatible request
      const runId = `run_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const agUIRequest = {
        thread_id: threadIdRef.current,
        run_id: runId,
        state: {},
        context: [],
        forwardedProps: {},
        messages: [
          ...messages.map((msg, index) => ({
            id: `msg_${index}_${Date.now()}`,
            role: msg.sender === 'user' ? 'user' : 'assistant',
            content: msg.text
          })),
          {
            id: `msg_${messages.length}_${Date.now()}`,
            role: 'user',
            content: userMessage
          }
        ],
        tools: []
      };

      // Send request to your AG-UI endpoint
      const response = await fetch('http://localhost:5051/api/agui/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(agUIRequest),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server response:', response.status, errorText);
        throw new Error(`HTTP error! status: ${response.status}, details: ${errorText}`);
      }

      // Handle streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let currentMessage = '';
      let messageId = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              switch (data.type) {
                case 'RUN_STARTED':
                  console.log('Run started:', data);
                  break;
                  
                case 'TEXT_MESSAGE_CHUNK':
                  if (!messageId) {
                    messageId = data.message_id;
                    setMessages((prev) => [...prev, { 
                      text: data.delta, 
                      sender: 'agent',
                      messageId: data.message_id
                    }]);
                  } else {
                    setMessages((prev) => 
                      prev.map((msg, i) => 
                        i === prev.length - 1 && msg.sender === 'agent' && msg.messageId === data.message_id
                          ? { ...msg, text: msg.text + data.delta }
                          : msg
                      )
                    );
                  }
                  currentMessage += data.delta;
                  break;
                  
                case 'RUN_FINISHED':
                  console.log('Run finished:', data);
                  setIsLoading(false);
                  break;
                  
                case 'RUN_ERROR':
                  console.error('Run error:', data);
                  setMessages((prev) => [...prev, { 
                    text: `Error: ${data.message}`, 
                    sender: 'system' 
                  }]);
                  setIsLoading(false);
                  break;
                  
                default:
                  console.log('Unhandled AG-UI event:', data);
              }
            } catch (e) {
              console.error('Error parsing event data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [...prev, { 
        text: `Connection error: ${error.message}`, 
        sender: 'system' 
      }]);
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'row',
      maxWidth: '1400px', 
      margin: '0 auto', 
      padding: '20px',
      boxSizing: 'border-box',
      gap: '20px',
      backgroundColor: '#f7f8fa',
      fontFamily: '"Inter UI", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
      color: '#343741'
    }}>
      {/* Maximized Graph Modal */}
      {maximizedGraph && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.8)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '20px'
          }}
          onClick={() => setMaximizedGraph(null)}
        >
          <div 
            style={{
              backgroundColor: 'white',
              borderRadius: '12px',
              padding: '20px',
              width: '90vw',
              height: '80vh',
              position: 'relative'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setMaximizedGraph(null)}
              style={{
                position: 'absolute',
                top: '10px',
                right: '15px',
                background: 'none',
                border: 'none',
                fontSize: '24px',
                cursor: 'pointer',
                color: '#666'
              }}
            >
              ×
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                window.open(maximizedGraph.prometheusUrl, '_blank');
              }}
              style={{
                position: 'absolute',
                top: '15px',
                right: '50px',
                background: '#e6522c',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                padding: '10px 16px',
                fontSize: '14px',
                cursor: 'pointer',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.target.style.background = '#d63031';
                e.target.style.transform = 'scale(1.05)';
              }}
              onMouseLeave={(e) => {
                e.target.style.background = '#e6522c';
                e.target.style.transform = 'scale(1)';
              }}
              title="Open query in Prometheus UI (new window)"
            >
              🔥 Open in Prometheus
            </button>
            <h3 style={{ margin: '0 0 20px 0' }}>{maximizedGraph.title}</h3>
            <div style={{ height: 'calc(100% - 60px)' }}>
              <Line 
                data={maximizedGraph.data} 
                options={{
                  ...maximizedGraph.options,
                  maintainAspectRatio: false,
                  responsive: true
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Task Tracker Panel */}
      {tasks.length > 0 && (
        <div style={{
          width: '300px',
          flexShrink: 0,
          display: 'flex',
          flexDirection: 'column'
        }}>
          <h3 style={{ margin: '0 0 16px 0', color: '#343741', fontSize: '16px', fontWeight: '600' }}>Investigation Tasks</h3>
          <div style={{
            border: '1px solid #d3dae6',
            borderRadius: '6px',
            padding: '16px',
            backgroundColor: '#ffffff',
            boxShadow: '0 2px 2px -1px rgba(152, 162, 179, 0.3), 0 1px 5px -2px rgba(152, 162, 179, 0.3)',
            flex: 1,
            overflowY: 'auto'
          }}>
            {tasks.map((task, index) => (
              <div key={task.id || index} style={{
                display: 'flex',
                alignItems: 'flex-start',
                marginBottom: '8px',
                padding: '8px',
                backgroundColor: 'white',
                borderRadius: '4px',
                border: '1px solid #e0e0e0'
              }}>
                <div style={{
                  marginRight: '8px',
                  fontSize: '16px',
                  minWidth: '20px'
                }}>
                  {task.status === 'completed' ? '✅' : 
                   task.status === 'in_progress' ? '🔄' : '⏳'}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ 
                    fontSize: '12px', 
                    color: '#666',
                    marginBottom: '2px'
                  }}>
                    Task {task.id}
                  </div>
                  <div style={{ 
                    fontSize: '14px',
                    lineHeight: '1.3'
                  }}>
                    {task.content}
                  </div>
                  <div style={{
                    fontSize: '11px',
                    color: task.status === 'completed' ? '#28a745' :
                           task.status === 'in_progress' ? '#ffc107' : '#6c757d',
                    marginTop: '4px',
                    fontWeight: 'bold'
                  }}>
                    {task.status.replace('_', ' ').toUpperCase()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div style={{ 
        flex: 1,
        display: 'flex', 
        flexDirection: 'column',
        minWidth: 0,
        backgroundColor: '#ffffff',
        borderRadius: '6px',
        border: '1px solid #d3dae6',
        boxShadow: '0 2px 2px -1px rgba(152, 162, 179, 0.3), 0 1px 5px -2px rgba(152, 162, 179, 0.3)',
        overflow: 'hidden'
      }}>
        <h1 style={{ 
          margin: '0', 
          padding: '20px 20px 16px 20px',
          flexShrink: 0, 
          color: '#343741', 
          fontSize: '24px', 
          fontWeight: '600',
          borderBottom: '1px solid #d3dae6'
        }}>Holmes GPT Chat (AG-UI compatible)</h1>
        
        <div style={{ 
          flex: 1,
          overflowY: 'auto', 
          padding: '16px 20px',
          backgroundColor: '#f7f8fa',
          minHeight: 0
        }}>
          {messages.map((msg, index) => (
            <div key={index} style={{ 
              textAlign: msg.sender === 'user' ? 'right' : 'left',
              marginBottom: '10px'
            }}>
              <div style={{
                display: 'inline-block',
                padding: '12px 16px',
                borderRadius: '6px',
                background: msg.sender === 'user' ? '#e3eff8' : 
                           msg.sender === 'system' ? '#bd271e' : 'linear-gradient(135deg, #ecf4ff 0%, #f1e7fe 100%)',
                border: msg.sender === 'user' ? '1px solid #0268bc' : 
                        msg.sender === 'system' ? 'none' : '1px solid #9435b5',
                color: msg.sender === 'system' ? '#ffffff' : '#343741',
                maxWidth: msg.text.includes('📊 **GRAPH_DATA:**') ? '95%' : '70%',
                wordWrap: 'break-word',
                boxShadow: '0 2px 2px -1px rgba(152, 162, 179, 0.3), 0 1px 5px -2px rgba(152, 162, 179, 0.3)'
              }}>
                <strong>{msg.sender}:</strong>{' '}
                {msg.sender === 'user' ? (
                  msg.text
                ) : (
                  <div>
                    {parseAndRenderContent(msg.text).map((part, partIndex) => (
                      <div key={partIndex}>
                        {part.type === 'text' ? (
                          <ReactMarkdown 
                            components={{
                              p: ({children}) => <span>{children}</span>,
                              strong: ({children}) => <strong style={{fontWeight: 'bold'}}>{children}</strong>,
                              em: ({children}) => <em style={{fontStyle: 'italic'}}>{children}</em>,
                              code: ({children}) => <code style={{backgroundColor: 'rgba(0,0,0,0.1)', color: '#0268bc', padding: '2px 4px', borderRadius: '3px', fontWeight: '500'}}>{children}</code>,
                              pre: ({children}) => <pre style={{backgroundColor: 'rgba(0,0,0,0.1)', color: '#0268bc', padding: '8px', borderRadius: '4px', overflow: 'auto', fontWeight: '500'}}>{children}</pre>
                            }}
                          >
                            {part.content}
                          </ReactMarkdown>
                        ) : part.type === 'graph' ? (
                          renderPrometheusGraph(part.data)
                        ) : null}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div style={{ textAlign: 'left', color: '#666' }}>
              <em>Holmes is thinking...</em>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div style={{ 
          display: 'flex', 
          gap: '8px', 
          flexShrink: 0,
          padding: '16px 20px',
          borderTop: '1px solid #d3dae6',
          backgroundColor: '#ffffff'
        }}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask Holmes a question..."
            disabled={isLoading}
            style={{ 
              flex: 1, 
              padding: '12px 16px', 
              border: '1px solid #d3dae6',
              borderRadius: '6px',
              minWidth: 0,
              fontSize: '14px',
              fontFamily: 'inherit',
              backgroundColor: '#ffffff',
              color: '#343741',
              outline: 'none'
            }}
          />
          <button 
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            style={{
              padding: '12px 24px',
              backgroundColor: isLoading || !input.trim() ? '#98a2b3' : '#006bb4',
              color: '#ffffff',
              border: 'none',
              borderRadius: '6px',
              cursor: isLoading || !input.trim() ? 'not-allowed' : 'pointer',
              flexShrink: 0,
              fontSize: '14px',
              fontWeight: '500',
              fontFamily: 'inherit'
            }}
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
