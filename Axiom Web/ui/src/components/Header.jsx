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


      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-1.5 sm:gap-2">


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
