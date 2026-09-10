import React, { useState } from 'react';
import { Zap, Copy, Check, User, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';
import MarkdownRenderer from './MarkdownRenderer';

export default function ChatMessage({ message }) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className={`w-full max-w-4xl mx-auto px-4 py-5 flex gap-4 sm:gap-5 ${
        isUser ? 'justify-end' : 'justify-start'
      }`}
    >
      {/* Assistant Avatar */}
      {!isUser && (
        <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 flex items-center justify-center shrink-0 shadow-lg shadow-brand-500/15 border border-white/10 mt-1">
          <Zap className="w-4 h-4 sm:w-4.5 sm:h-4.5 text-white fill-white/20" />
        </div>
      )}

      {/* Message Content Area */}
      <div className={`flex flex-col min-w-0 max-w-[88%] sm:max-w-[82%] ${isUser ? 'items-end' : 'items-start flex-1'}`}>
        
        {/* Name / Role */}
        <div className="flex items-center gap-2 mb-1.5 text-xs text-slate-400 font-medium">
          <span>{isUser ? 'You' : 'Axiom'}</span>
          {message.timestamp && (
            <span className="text-[10px] text-slate-400">{message.timestamp}</span>
          )}
        </div>

        {/* Message Bubble or Markdown */}
        {isUser ? (
          <div className="group relative bg-dark-850 hover:bg-dark-800 text-slate-100 px-4 sm:px-5 py-3 rounded-2xl rounded-tr-sm border border-dark-750/80 shadow-md text-sm leading-relaxed whitespace-pre-wrap transition-colors">
            {message.content}
            <button
              onClick={handleCopy}
              title="Copy message"
              className="absolute -left-8 top-2.5 opacity-0 group-hover:opacity-100 p-1 rounded-md text-slate-400 hover:text-slate-200 transition-opacity"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-cyber-emerald" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
        ) : (
          <div className="w-full">
            <div className="text-slate-200 leading-relaxed">
              <MarkdownRenderer content={message.content} />
            </div>

            {/* Assistant Footer Utility Bar */}
            <div className="flex items-center gap-3 mt-3 pt-2 text-xs text-slate-400">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md hover:bg-dark-800 hover:text-slate-200 text-slate-400 transition-colors border border-transparent hover:border-dark-750"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-cyber-emerald" />
                    <span className="text-cyber-emerald text-[11px]">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span className="text-[11px]">Copy response</span>
                  </>
                )}
              </button>

              {message.speed && parseFloat(message.speed) > 0 && (
                <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-dark-850 border border-dark-750 text-[11px] font-mono text-slate-400">
                  <Zap className="w-3 h-3 text-cyber-emerald" />
                  <span>{message.speed} tok/s</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-dark-800 border border-dark-700 flex items-center justify-center shrink-0 mt-1 text-slate-300 shadow-md">
          <User className="w-4 h-4" />
        </div>
      )}
    </motion.div>
  );
}
