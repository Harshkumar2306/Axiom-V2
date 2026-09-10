import React, { useRef, useEffect } from 'react';
import { ArrowUp, X, StopCircle, Globe, Paperclip, FileText, Loader2, Database } from 'lucide-react';

export default function ChatInput({
  input,
  setInput,
  onSend,
  isGenerating,
  temperature,
  maxTokens,
  onOpenSettings,
  isWebSearch,
  setIsWebSearch,
  uploadedDocs = [],
  isUploadingDoc = false,
  onUploadFile,
  onRemoveDoc
}) {
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && input.trim()) {
        onSend();
      }
    }
  };

  const handleStop = () => {
    if (window.currentAbortController) {
      window.currentAbortController.abort();
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && onUploadFile) {
      onUploadFile(file);
      e.target.value = '';
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-4 sm:pb-6">
      <div className="relative rounded-2xl bg-dark-900 border border-dark-750/90 shadow-2xl focus-within:border-brand-500/50 focus-within:ring-2 focus-within:ring-brand-500/15 transition-all duration-200">
        
        {/* Active Uploaded RAG Documents Tray */}
        {uploadedDocs.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 px-4 pt-3 pb-1 border-b border-dark-800/80">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-amber-400/90 flex items-center gap-1 mr-1">
              <Database className="w-3 h-3 text-amber-400" />
              RAG Docs:
            </span>
            {uploadedDocs.map((doc, idx) => (
              <div
                key={idx}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs shadow-sm"
              >
                <FileText className="w-3 h-3 text-amber-400 shrink-0" />
                <span className="truncate max-w-[140px]">{doc.filename}</span>
                {doc.chunks && (
                  <span className="text-[10px] text-amber-400/70 font-mono">({doc.chunks}c)</span>
                )}
                {onRemoveDoc && (
                  <button
                    onClick={() => onRemoveDoc(doc.filename)}
                    title="Remove from knowledge base"
                    className="p-0.5 hover:bg-amber-500/20 rounded text-amber-400 hover:text-amber-200 transition-colors ml-0.5"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Input Text Area */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            uploadedDocs.length > 0
              ? "Ask a question about your uploaded documents or anything else..."
              : isWebSearch
              ? "Ask anything... Axiom will search and scrape the live web..."
              : "Message Axiom... (Enter to send, Shift+Enter for new line)"
          }
          rows={1}
          className="w-full bg-transparent border-none focus:outline-none focus:ring-0 text-slate-100 placeholder-slate-500 resize-none px-4 pt-3.5 pb-12 text-sm sm:text-base leading-relaxed max-h-48"
        />

        {/* Hidden File Input for Document RAG */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md,.py,.json,.csv,.docx"
          onChange={handleFileChange}
          className="hidden"
        />

        {/* Bottom Bar Tools */}
        <div className="absolute left-3 right-3 bottom-2.5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            
            {/* Clickable Temp & Token Pills */}
            <button
              type="button"
              onClick={onOpenSettings}
              title="Click to adjust Temperature & Max Tokens in Model Parameters (Cmd+,)"
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-dark-800/80 hover:bg-dark-750 border border-dark-700/60 text-[11px] font-mono text-slate-300 hover:text-white transition-colors cursor-pointer group"
            >
              <span>Temp: <strong className="text-brand-400 font-medium">{temperature.toFixed(2)}</strong></span>
              <span className="text-slate-600">•</span>
              <span>Max: <strong className="text-brand-400 font-medium">{maxTokens || 512}</strong> tok</span>
            </button>

            {/* Live Web Search Toggle */}
            <button
              onClick={() => setIsWebSearch(!isWebSearch)}
              title={isWebSearch ? "Live Web Search Enabled" : "Enable Live Web Search & Scraping"}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors border ${
                isWebSearch 
                  ? 'bg-cyber-cyan/15 text-cyber-cyan border-cyber-cyan/40 shadow-sm shadow-cyber-cyan/20' 
                  : 'bg-dark-800/80 text-slate-400 border-dark-700/50 hover:bg-dark-750 hover:text-slate-300'
              }`}
            >
              <Globe className={`w-3 h-3 ${isWebSearch ? 'animate-pulse' : ''}`} />
              <span>Web Search</span>
            </button>

            {/* Attach Document for RAG */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploadingDoc}
              title="Attach Document to RAG Knowledge Base (PDF, TXT, MD, Code)"
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors border ${
                uploadedDocs.length > 0
                  ? 'bg-amber-500/15 text-amber-300 border-amber-500/40'
                  : 'bg-dark-800/80 text-slate-400 border-dark-700/50 hover:bg-dark-750 hover:text-slate-300'
              }`}
            >
              {isUploadingDoc ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin text-amber-400" />
                  <span>Indexing...</span>
                </>
              ) : (
                <>
                  <Paperclip className="w-3 h-3" />
                  <span>{uploadedDocs.length > 0 ? 'Add Document' : 'Attach File (RAG)'}</span>
                </>
              )}
            </button>
          </div>

          {/* Action buttons (Clear / Stop / Send) */}
          <div className="flex items-center gap-1.5">
            {input.trim() && !isGenerating && (
              <button
                onClick={() => setInput('')}
                title="Clear input"
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-dark-800 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}

            {isGenerating ? (
              <button
                onClick={handleStop}
                title="Stop generating"
                className="flex items-center justify-center w-8 h-8 rounded-xl bg-rose-500/20 text-rose-500 hover:bg-rose-500 hover:text-white transition-colors"
              >
                <StopCircle className="w-5 h-5" />
              </button>
            ) : (
              <button
                onClick={onSend}
                disabled={!input.trim()}
                className={`flex items-center justify-center w-8 h-8 rounded-xl transition-all ${
                  input.trim()
                    ? 'bg-white text-black hover:bg-slate-200 shadow-md scale-100'
                    : 'bg-dark-800 text-slate-600 cursor-not-allowed scale-95'
                }`}
              >
                <ArrowUp className="w-4 h-4 stroke-[2.5]" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
