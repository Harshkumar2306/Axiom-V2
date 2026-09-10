import React from 'react';
import { Download, Trash2, Sliders, Menu, Sparkles } from 'lucide-react';

export default function Header({
  onToggleSidebar,
  onOpenSettings,
  onClearChat,
  onExportChat,
  hasMessages,
  temperature
}) {
  return (
    <header className="h-16 border-b border-dark-800 bg-dark-900/60 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between shrink-0 z-20">
      <div className="flex items-center gap-3">
        {/* Mobile menu trigger */}
        <button
          onClick={onToggleSidebar}
          className="md:hidden p-2 rounded-lg text-slate-400 hover:text-white hover:bg-dark-800 transition-colors"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Model info pill */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-dark-850 border border-dark-750 text-xs">
            <span className="w-2 h-2 rounded-full bg-cyber-emerald"></span>
            <span className="font-medium text-slate-200">Axiom 476M SFT</span>
            <span className="text-slate-400 font-mono text-[10px]">T={temperature.toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-1.5 sm:gap-2">
        {hasMessages && (
          <>
            <button
              onClick={onExportChat}
              title="Export Conversation to Markdown"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-300 hover:text-white bg-dark-850 hover:bg-dark-800 border border-dark-750 transition-all"
            >
              <Download className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Export</span>
            </button>

            <button
              onClick={onClearChat}
              title="Clear current chat"
              className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-dark-850 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </>
        )}

        <button
          onClick={onOpenSettings}
          title="Model Parameters"
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-dark-850 transition-colors"
        >
          <Sliders className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
