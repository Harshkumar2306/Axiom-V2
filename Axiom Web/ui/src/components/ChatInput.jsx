import React, { useRef, useEffect } from 'react';
import { ArrowUp, X, StopCircle } from 'lucide-react';

export default function ChatInput({
  input,
  setInput,
  onSend,
  isGenerating,
  temperature
}) {
  const textareaRef = useRef(null);

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

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-4 sm:pb-6">
      <div className="relative rounded-2xl bg-dark-900 border border-dark-750/90 shadow-2xl focus-within:border-brand-500/50 focus-within:ring-2 focus-within:ring-brand-500/15 transition-all duration-200">
        
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message Axiom... (Enter to send, Shift+Enter for new line)"
          rows={1}
          className="w-full bg-transparent border-none focus:outline-none focus:ring-0 text-slate-100 placeholder-slate-500 resize-none px-4 pt-3.5 pb-12 text-sm sm:text-base leading-relaxed max-h-48"
        />

        <div className="absolute left-3 right-3 bottom-2.5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-slate-400 bg-dark-800/80 px-2 py-0.5 rounded-md border border-dark-700/50">
              Temp: {temperature.toFixed(2)}
            </span>
          </div>

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
