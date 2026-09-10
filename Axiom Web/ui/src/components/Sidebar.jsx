import React from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  Cpu,
  Sliders,
  ChevronLeft,
  ChevronRight,
  Zap,
  Check
} from 'lucide-react';

export default function Sidebar({
  conversations,
  activeId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  isCollapsed,
  onToggleCollapse,
  onOpenSettings,
  temperature
}) {
  return (
    <aside
      className={`relative flex flex-col h-full bg-dark-900 border-r border-dark-800 transition-all duration-300 z-30 shrink-0 select-none ${
        isCollapsed ? 'w-[72px]' : 'w-72'
      }`}
    >
      {/* Top Brand & Collapse */}
      <div className="flex items-center justify-between p-4 border-b border-dark-800 h-16">
        {!isCollapsed && (
          <div className="flex items-center gap-3">
            <div className="relative flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 shadow-md shadow-brand-500/20">
              <Zap className="w-5 h-5 text-white fill-white/20" />
            </div>
            <div>
              <h1 className="font-display font-semibold text-base tracking-tight text-white flex items-center gap-1.5">
                Axiom
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-brand-500/10 text-brand-400 border border-brand-500/20">
                  v2
                </span>
              </h1>
            </div>
          </div>
        )}

        {isCollapsed && (
          <div className="w-full flex justify-center">
            <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 shadow-md">
              <Zap className="w-5 h-5 text-white fill-white/20" />
            </div>
          </div>
        )}

        <button
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="hidden md:flex p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-dark-800 transition-colors"
        >
          {isCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onNewConversation}
          title="New Chat (Cmd+K)"
          className={`w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-dark-800 hover:bg-dark-750 text-slate-100 border border-dark-700/80 hover:border-brand-500/40 transition-all text-sm font-medium shadow-sm group ${
            isCollapsed ? 'px-0' : 'px-4'
          }`}
        >
          <Plus className="w-4 h-4 text-brand-400 group-hover:rotate-90 transition-transform duration-200" />
          {!isCollapsed && <span>New Chat</span>}
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
        {!isCollapsed && conversations.length > 0 && (
          <div className="px-3 py-1.5 text-[11px] font-medium uppercase tracking-wider text-slate-400">
            Recent Conversations
          </div>
        )}

        {conversations.map((conv) => {
          const isActive = conv.id === activeId;
          return (
            <div
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className={`group relative flex items-center gap-3 px-3 py-2.5 rounded-xl cursor-pointer text-xs transition-all ${
                isActive
                  ? 'bg-dark-800 text-white font-medium border border-dark-700 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-dark-850'
              } ${isCollapsed ? 'justify-center px-0' : ''}`}
            >
              <MessageSquare
                className={`w-4 h-4 shrink-0 ${
                  isActive ? 'text-brand-400' : 'text-slate-400 group-hover:text-slate-300'
                }`}
              />

              {!isCollapsed && (
                <>
                  <span className="truncate flex-1">{conv.title}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(conv.id);
                    }}
                    title="Delete chat"
                    className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-dark-700 text-slate-400 hover:text-rose-400 transition-all shrink-0"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </>
              )}
            </div>
          );
        })}

        {!isCollapsed && conversations.length === 0 && (
          <div className="p-4 text-center text-xs text-slate-400">
            No previous chats
          </div>
        )}
      </div>

      {/* Hardware & Settings Card */}
      <div className="p-3 border-t border-dark-800 bg-dark-900/50 space-y-2">
        {!isCollapsed ? (
          <>
            {/* Status card */}
            <div className="p-2.5 rounded-xl bg-dark-850 border border-dark-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </div>
                <div className="text-[11px] leading-tight">
                  <div className="text-slate-200 font-medium">MPS Unified Memory</div>
                  <div className="text-slate-400 text-[10px]">Engine Active</div>
                </div>
              </div>
              <Cpu className="w-4 h-4 text-slate-400" />
            </div>

            {/* Settings button */}
            <button
              onClick={onOpenSettings}
              className="w-full flex items-center gap-2 px-3 py-2 rounded-xl text-xs text-slate-300 hover:text-white hover:bg-dark-800 transition-colors font-medium border border-transparent hover:border-dark-750"
            >
              <Sliders className="w-4 h-4 text-slate-400" />
              <span>Model Parameters</span>
            </button>
          </>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <button
              onClick={onOpenSettings}
              title="Settings"
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-dark-800 transition-colors"
            >
              <Sliders className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}
