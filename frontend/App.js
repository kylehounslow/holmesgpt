import React, { useState, useEffect, useRef } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const threadIdRef = useRef(null);

  // Generate thread_id on first load
  useEffect(() => {
    threadIdRef.current = `thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
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
        messages: [
          ...messages.map(msg => ({
            role: msg.sender === 'user' ? 'user' : 'assistant',
            content: msg.text
          })),
          {
            role: 'user',
            content: userMessage
          }
        ]
      };

      // Send request to your AG-UI endpoint
      const response = await fetch('http://localhost:8000/api/agui/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(agUIRequest),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
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
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
      <h1>Holmes GPT Chat (AG-UI)</h1>
      
      <div style={{ 
        height: '400px', 
        overflowY: 'scroll', 
        border: '1px solid #ccc', 
        padding: '10px',
        marginBottom: '10px',
        backgroundColor: '#f9f9f9'
      }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ 
            textAlign: msg.sender === 'user' ? 'right' : 'left',
            marginBottom: '10px'
          }}>
            <div style={{
              display: 'inline-block',
              padding: '8px 12px',
              borderRadius: '8px',
              backgroundColor: msg.sender === 'user' ? '#007bff' : 
                             msg.sender === 'system' ? '#dc3545' : '#28a745',
              color: 'white',
              maxWidth: '70%'
            }}>
              <strong>{msg.sender}:</strong> {msg.text}
            </div>
          </div>
        ))}
        {isLoading && (
          <div style={{ textAlign: 'left', color: '#666' }}>
            <em>Holmes is thinking...</em>
          </div>
        )}
      </div>
      
      <div style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask Holmes a question..."
          disabled={isLoading}
          style={{ 
            flex: 1, 
            padding: '10px', 
            border: '1px solid #ccc',
            borderRadius: '4px'
          }}
        />
        <button 
          onClick={sendMessage}
          disabled={isLoading || !input.trim()}
          style={{
            padding: '10px 20px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer'
          }}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default App;
