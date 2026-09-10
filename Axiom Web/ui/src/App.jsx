import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import EmptyState from './components/EmptyState';
import SettingsModal from './components/SettingsModal';

const DEFAULT_SETTINGS = {
  temperature: 0.2,
  maxTokens: 512,
  systemPrompt: 'You are a highly intelligent, logical, and helpful AI assistant named Axiom.',
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
};

export default function App() {
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

  const [settings, setSettings] = useState(() => {
    try {
      const saved = localStorage.getItem('axiom_settings');
      return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
    } catch {
      return DEFAULT_SETTINGS;
    }
  });

  const [input, setInput] = useState('');
  const [isWebSearch, setIsWebSearch] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // RAG Document state
  const [ragDocuments, setRagDocuments] = useState([]);
  const [isUploadingDoc, setIsUploadingDoc] = useState(false);

  const chatContainerRef = useRef(null);

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

  // Fetch RAG documents on load
  const fetchRagDocuments = async () => {
    try {
      const endpoint = `${settings.apiBaseUrl.replace(/\/$/, '')}/rag/documents`;
      const res = await fetch(endpoint);
      if (res.ok) {
        const data = await res.json();
        setRagDocuments(data.documents || []);
      }
    } catch {
      // Backend may not be reachable initially
    }
  };

  useEffect(() => {
    fetchRagDocuments();
  }, [settings.apiBaseUrl]);

  const activeConversation = conversations.find((c) => c.id === activeId) || null;
  const messages = activeConversation ? activeConversation.messages : [];

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

  // --- Document RAG Actions ---
  const handleUploadFile = async (file) => {
    if (!file) return;
    setIsUploadingDoc(true);
    try {
      const endpoint = `${settings.apiBaseUrl.replace(/\/$/, '')}/rag/upload`;
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        alert(`Document upload failed: ${err.detail || 'Unknown error'}`);
      } else {
        await fetchRagDocuments();
      }
    } catch (e) {
      alert(`Network error uploading document: ${e.message}`);
    } finally {
      setIsUploadingDoc(false);
    }
  };

  const handleDeleteRagDocument = async (filename) => {
    try {
      const endpoint = `${settings.apiBaseUrl.replace(/\/$/, '')}/rag/documents?filename=${encodeURIComponent(filename)}`;
      const res = await fetch(endpoint, { method: 'DELETE' });
      if (res.ok) {
        await fetchRagDocuments();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleClearRagDocuments = async () => {
    if (!window.confirm('Clear all indexed documents from the Knowledge Base?')) return;
    try {
      const endpoint = `${settings.apiBaseUrl.replace(/\/$/, '')}/rag/documents`;
      const res = await fetch(endpoint, { method: 'DELETE' });
      if (res.ok) {
        setRagDocuments([]);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleSendPrompt = async (customPrompt) => {
    const promptToSend = (customPrompt || input).trim();
    if (!promptToSend || isGenerating) return;

    let currentChatId = activeId;
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
      const abortController = new AbortController();
      window.currentAbortController = abortController;

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortController.signal,
        body: JSON.stringify({
          prompt: promptToSend,
          max_tokens: settings.maxTokens,
          temperature: settings.temperature,
          system_prompt: settings.systemPrompt,
          web_search: isWebSearch,
          rag_search: true
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let streamedResponse = '';
      let finalSpeed = '0.00';
      let sources = [];

      const timestampStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentChatId
            ? { ...c, messages: [...c.messages, { role: 'axiom', content: '', speed: null, sources: [], timestamp: timestampStr }] }
            : c
        )
      );

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            
            if (data.startsWith('__AXIOM_SPEED_')) {
              finalSpeed = data.replace('__AXIOM_SPEED_', '').replace('__', '');
              continue;
            }

            if (data.startsWith('__AXIOM_SOURCES__')) {
              try {
                const rawJson = data.replace('__AXIOM_SOURCES__', '').replace(/__$/, '');
                sources = JSON.parse(rawJson);
              } catch (e) {
                console.error('Failed to parse sources:', e);
              }
              continue;
            }
            
            streamedResponse += data;
            
            setConversations((prev) =>
              prev.map((c) => {
                if (c.id === currentChatId) {
                  const updatedMsgs = [...c.messages];
                  updatedMsgs[updatedMsgs.length - 1] = {
                    ...updatedMsgs[updatedMsgs.length - 1],
                    content: streamedResponse.replace(/<\|endoftext\|>/g, '')
                  };
                  return { ...c, messages: updatedMsgs };
                }
                return c;
              })
            );
          }
        }
      }
      
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === currentChatId) {
            const updatedMsgs = [...c.messages];
            updatedMsgs[updatedMsgs.length - 1].speed = finalSpeed;
            updatedMsgs[updatedMsgs.length - 1].sources = sources;
            return { ...c, messages: updatedMsgs };
          }
          return c;
        })
      );

    } catch (err) {
      if (err.name === 'AbortError') {
         console.log("Generation stopped by user.");
      } else {
        const errorMessage = {
          role: 'axiom',
          content: `⚠️ **Connection Notice:** Unable to reach the backend at \`${settings.apiBaseUrl}\`.\n\nPlease verify that the Python server is running (\`python3 app.py\`) or adjust the API Base URL in **Model Parameters** (\`Cmd+,\`).`,
          speed: 0,
          sources: [],
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        setConversations((prev) =>
          prev.map((c) =>
            c.id === currentChatId
              ? { ...c, messages: [...c.messages, errorMessage] }
              : c
          )
        );
      }
    } finally {
      setIsGenerating(false);
      window.currentAbortController = null;
    }
  };

  return (
    <div className="flex h-screen w-screen bg-dark-950 text-slate-100 overflow-hidden select-text font-sans">
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
        maxTokens={settings.maxTokens}
        ragDocuments={ragDocuments}
        onClearRagDocuments={handleClearRagDocuments}
        onDeleteRagDocument={handleDeleteRagDocument}
      />

      <div className="flex-1 flex flex-col h-full min-w-0 bg-dark-950 relative">
        <Header
          onToggleSidebar={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          onOpenSettings={() => setIsSettingsOpen(true)}
          onClearChat={handleClearCurrentChat}
          onExportChat={handleExportChat}
          hasMessages={messages.length > 0}
          temperature={settings.temperature}
        />

        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col"
        >
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="flex-1 pb-6 pt-4 space-y-1">
              {messages.map((msg, idx) => (
                <ChatMessage key={idx} message={msg} />
              ))}
            </div>
          )}
        </div>

        <ChatInput
          input={input}
          setInput={setInput}
          onSend={() => handleSendPrompt()}
          isGenerating={isGenerating}
          temperature={settings.temperature}
          maxTokens={settings.maxTokens}
          onOpenSettings={() => setIsSettingsOpen(true)}
          isWebSearch={isWebSearch}
          setIsWebSearch={setIsWebSearch}
          uploadedDocs={ragDocuments}
          isUploadingDoc={isUploadingDoc}
          onUploadFile={handleUploadFile}
          onRemoveDoc={handleDeleteRagDocument}
        />
      </div>

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
