import React, { useState, useRef, useEffect } from 'react';
import { Send, Zap, Code, Brain, Settings2, Plus, Mic } from 'lucide-react';
import { marked } from 'marked';
import { motion, AnimatePresence } from 'framer-motion';
import ChatBubble from './ChatBubble';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [thinkMode, setThinkMode] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  
  const chatContainerRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  // Health check ping
  useEffect(() => {
    const pingBackend = async () => {
      try {
        const res = await fetch('/api/health');
        if (res.ok) setIsConnected(true);
        else setIsConnected(false);
      } catch (err) {
        setIsConnected(false);
      }
    };
    pingBackend();
    const interval = setInterval(pingBackend, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleInput = (e) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  const handleSend = async (textOverride) => {
    const text = textOverride || input.trim();
    if (!text || isStreaming || !isConnected) return;

    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    
    // Add user message and empty axiom message
    setMessages(prev => [
      ...prev, 
      { role: 'user', content: text },
      { role: 'axiom', content: '' }
    ]);
    setIsStreaming(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, temperature: temperature, max_tokens: maxTokens })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').trim();
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.token) {
                setMessages(prev => {
                  const newMsgs = [...prev];
                  const lastMsg = { ...newMsgs[newMsgs.length - 1] };
                  lastMsg.content += data.token;
                  
                  // Clean up the stop token if it bleeds into the stream
                  if (lastMsg.content.includes('<|endoftext|>')) {
                      lastMsg.content = lastMsg.content.replace('<|endoftext|>', '');
                  }
                  
                  newMsgs[newMsgs.length - 1] = lastMsg;
                  return newMsgs;
                });
              }
            } catch (err) {}
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].content += '\n\n*Error connecting to backend.*';
        return newMsgs;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo">
          <Brain size={28} color={isConnected ? '#7b87ff' : '#f87171'} />
          <h1>Axiom</h1>
        </div>
        <p className={`status ${isConnected ? '' : 'disconnected'}`}>
          {isConnected ? '● Brain Connected' : '○ Brain Disconnected'}
        </p>
      </header>

      <main className="chat-container" ref={chatContainerRef}>
        <AnimatePresence mode="wait">
          {messages.length === 0 ? (
            <motion.div 
              key="empty-state"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9, y: -20 }}
              transition={{ duration: 0.4, type: 'spring' }}
              className="empty-state"
            >
              <div className="empty-logo-orb">
                <Brain size={48} color={isConnected ? '#7b87ff' : '#6a7180'} />
              </div>
              <h2>How can I assist you today?</h2>
              
              <div className="suggestions-grid">
                <button className="suggestion-card" onClick={() => handleSend("Tell me about your neural architecture. How many parameters do you have?")}>
                  <Brain size={20} className="icon text-blue" />
                  <p>Model Architecture</p>
                  <span>Ask about my 476M parameters & structure.</span>
                </button>
                <button className="suggestion-card" onClick={() => handleSend("Explain your training pipeline. How were you aligned using DPO?")}>
                  <Zap size={20} className="icon text-yellow" />
                  <p>Training Pipeline</p>
                  <span>Learn about Pretraining, SFT, and DPO.</span>
                </button>
                <button className="suggestion-card" onClick={() => handleSend("Demonstrate your reasoning capabilities by solving a complex logic puzzle.")}>
                  <Code size={20} className="icon text-green" />
                  <p>Test Reasoning</p>
                  <span>Challenge my logic and coding abilities.</span>
                </button>
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="chat-box"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="chat-box"
            >
              {messages.map((msg, idx) => (
                <ChatBubble 
                  key={idx} 
                  role={msg.role} 
                  content={msg.content} 
                  isStreaming={isStreaming && idx === messages.length - 1 && msg.role === 'axiom'} 
                />
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="input-container">
        <AnimatePresence>
          {showSettings && (
            <motion.div 
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
              className="settings-panel"
            >
              <div className="setting-item">
                <div className="setting-header">
                  <label>Temperature</label>
                  <span>{temperature}</span>
                </div>
                <input 
                  type="range" 
                  min="0" max="1" step="0.1" 
                  value={temperature} 
                  onChange={(e) => setTemperature(parseFloat(e.target.value))} 
                />
              </div>
              <div className="setting-item">
                <div className="setting-header">
                  <label>Max Tokens</label>
                  <span>{maxTokens}</span>
                </div>
                <input 
                  type="range" 
                  min="32" max="2048" step="32" 
                  value={maxTokens} 
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))} 
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        
        <div className="input-box">
          <textarea 
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={isConnected ? "Message Axiom..." : "Brain is disconnected..."}
            rows="1" 
            autoFocus
          />

          <div className="input-right-actions">
            <button 
              className={`think-btn ${showSettings ? 'active' : ''}`}
              onClick={() => setShowSettings(!showSettings)}
              title="Adjust generation parameters"
            >
              <Settings2 size={16} />
              <span>Settings</span>
            </button>
            <button 
              className="send-btn" 
              onClick={handleSend}
              disabled={!input.trim() || isStreaming || !isConnected}
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
