import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import EmptyState from './components/EmptyState';
import SettingsModal from './components/SettingsModal';
import { Zap } from 'lucide-react';

const DEFAULT_SETTINGS = {
  temperature: 0.2,
  maxTokens: 500,
  systemPrompt: 'You are a highly intelligent, logical, and helpful AI assistant named Axiom.',
  apiBaseUrl: 'http://localhost:8000'
};

export default function App() {
  // Load conversations from localStorage
  const [conversations, setConversations] = useState(() => {
    try {
      const saved = localStorage.getItem('axiom_conversations');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [activeId, setActiveId] = useState(() => {
    try {
      const saved = localStorage.getItem('axiom_active_chat');
      return saved || null;
    } catch {
      return null;
    }
  });

  // Settings
  const [settings, setSettings] = useState(() => {
    try {
      const saved = localStorage.getItem('axiom_settings');
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  const chatContainerRef = useRef(null);

  // Sync to localStorage
  useEffect(() => {
    localStorage.setItem('axiom_conversations', JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    if (activeId) {
      localStorage.setItem('axiom_active_chat', activeId);
    }
  }, [activeId]);

  useEffect(() => {
    localStorage.setItem('axiom_settings', JSON.stringify(settings));
  }, [settings]);

  // Current conversation
  const activeConversation = conversations.find((c) => c.id === activeId) || null;
  const messages = activeConversation ? activeConversation.messages : [];

  // Scroll to bottom
  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTo({
        top: chatContainerRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  // Keyboard Shortcuts (Cmd+K for new chat, Cmd+, for settings)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        handleNewConversation();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === ',') {
        e.preventDefault();
        setIsSettingsOpen(true);
      }
      if (e.key === 'Escape' && isSettingsOpen) {
        setIsSettingsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isSettingsOpen]);

  // Handlers
  const handleNewConversation = () => {
    const newId = 'chat_' + Date.now();
    const newChat = {
      id: newId,
      title: 'New Conversation',
      messages: [],
      createdAt: new Date().toISOString()
    };
    setConversations((prev) => [newChat, ...prev]);
    setActiveId(newId);
    setInput('');
  };

  const handleDeleteConversation = (id) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeId === id) {
      const remaining = conversations.filter((c) => c.id !== id);
      setActiveId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  const handleClearCurrentChat = () => {
    if (!activeId) return;
    setConversations((prev) =>
      prev.map((c) => (c.id === activeId ? { ...c, messages: [] } : c))
    );
  };

  const handleExportChat = () => {
    if (!messages || messages.length === 0) return;

    let markdownContent = `# Axiom Chat Export\n*Generated on ${new Date().toLocaleString()}*\n\n---\n\n`;
    messages.forEach((msg) => {
      const author = msg.role === 'user' ? '### 👤 User' : '### 🤖 Axiom';
      markdownContent += `${author}\n\n${msg.content}\n\n---\n\n`;
    });

    const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `axiom_chat_${Date.now()}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleSendPrompt = async (customPrompt) => {
    const promptToSend = (customPrompt || input).trim();
    if (!promptToSend || isGenerating) return;

    let currentChatId = activeId;

    // If no active chat, create one
    if (!currentChatId) {
      currentChatId = 'chat_' + Date.now();
      const newChat = {
        id: currentChatId,
        title: promptToSend.slice(0, 36) + (promptToSend.length > 36 ? '...' : ''),
        messages: [],
        createdAt: new Date().toISOString()
      };
      setConversations((prev) => [newChat, ...prev]);
      setActiveId(currentChatId);
    }

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const userMessage = { role: 'user', content: promptToSend, timestamp };

    // Append user message & update title if first message
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id === currentChatId) {
          const isFirstMessage = c.messages.length === 0;
          return {
            ...c,
            title: isFirstMessage
              ? promptToSend.slice(0, 36) + (promptToSend.length > 36 ? '...' : '')
              : c.title,
            messages: [...c.messages, userMessage]
          };
        }
        return c;
      })
    );

    setInput('');
    setIsGenerating(true);

    try {
      const endpoint = `${settings.apiBaseUrl.replace(/\/$/, '')}/chat`;
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: promptToSend,
          max_tokens: settings.maxTokens,
          temperature: settings.temperature
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP status ${res.status}`);
      }

      const data = await res.json();
      const assistantMessage = {
        role: 'axiom',
        content: data.response,
        speed: data.speed,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentChatId
            ? { ...c, messages: [...c.messages, assistantMessage] }
            : c
        )
      );
    } catch (err) {
      const errorMessage = {
        role: 'axiom',
        content: `⚠️ **Connection Notice:** Unable to reach the backend at \`${settings.apiBaseUrl}\`.\n\nPlease verify that the Python server is running (\`python3 app.py\`) or adjust the API Base URL in **Model Parameters** (\`Cmd+,\`).`,
        speed: 0,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentChatId
            ? { ...c, messages: [...c.messages, errorMessage] }
            : c
        )
      );
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-dark-950 text-slate-100 overflow-hidden select-text font-sans">
      
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelectConversation={(id) => setActiveId(id)}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        temperature={settings.temperature}
      />

      {/* Main App Canvas */}
      <div className="flex-1 flex flex-col h-full min-w-0 bg-dark-950 relative">
        
        {/* Header Bar */}
        <Header
          onToggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          onOpenSettings={() => setIsSettingsOpen(true)}
          onClearChat={handleClearCurrentChat}
          onExportChat={handleExportChat}
          hasMessages={messages.length > 0}
          temperature={settings.temperature}
        />

        {/* Chat Scroll Container */}
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col"
        >
          {messages.length === 0 ? (
            <EmptyState onSelectPrompt={(p) => handleSendPrompt(p)} />
          ) : (
            <div className="flex-1 pb-6 pt-4 space-y-1">
              {messages.map((msg, idx) => (
                <ChatMessage key={idx} message={msg} />
              ))}

              {/* Generating Loader */}
              {isGenerating && (
                <div className="w-full max-w-4xl mx-auto px-4 py-5 flex gap-4 sm:gap-5 items-start">
                  <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-lg shadow-brand-500/15 border border-white/10 mt-1 animate-pulse">
                    <Zap className="w-4 h-4 text-white fill-white/20" />
                  </div>
                  <div className="flex flex-col gap-2 mt-2">
                    <div className="text-xs text-slate-400 font-medium">Axiom</div>
                    <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-dark-850 border border-dark-750 text-xs text-slate-400">
                      <div className="typing-loader">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                      <span className="font-mono text-[11px]">Generating next tokens...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Floating Input Bar */}
        <ChatInput
          input={input}
          setInput={setInput}
          onSend={() => handleSendPrompt()}
          isGenerating={isGenerating}
          temperature={settings.temperature}
        />
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSaveSettings={(newSettings) => setSettings(newSettings)}
        onResetSettings={() => setSettings(DEFAULT_SETTINGS)}
      />
    </div>
  );
}
