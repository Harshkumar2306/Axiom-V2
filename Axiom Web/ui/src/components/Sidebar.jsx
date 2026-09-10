import React from 'react';
import { 
  Plus, 
  MessageSquare, 
  Trash2, 
  Sliders, 
  ChevronLeft, 
  ChevronRight, 
  Cpu, 
  Database,
  FileText,
  Trash
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
  temperature,
  ragDocuments = [],
  onClearRagDocuments,
  onDeleteRagDocument
}) {
  return (
    <aside
      className={`relative flex flex-col border-r border-dark-800 bg-dark-900/90 backdrop-blur-xl transition-all duration-300 ease-in-out z-30 ${
        isCollapsed ? 'w-16' : 'w-72'
      }`}
    >
      {/* Sidebar Header */}
      <div className="p-3 border-b border-dark-800 flex items-center justify-between">
        {!isCollapsed && (
          <div className="flex items-center gap-2 pl-2">
            <span className="font-semibold text-sm tracking-wide text-white">Axiom</span>
            <span className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-brand-500/10 text-brand-400 border border-brand-500/20">
              476M SFT
            </span>
          </div>
        )}

        <button
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-dark-800 transition-colors mx-auto"
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={onNewConversation}
          title="New Conversation (Cmd+K)"
          className={`w-full flex items-center gap-2 px-3 py-2.5 rounded-xl text-xs font-medium bg-dark-800 hover:bg-dark-750 text-slate-200 hover:text-white border border-dark-700/80 shadow-sm transition-all group ${
            isCollapsed ? 'justify-center px-0' : 'justify-start'
          }`}
        >
          <Plus className="w-4 h-4 text-brand-400 group-hover:rotate-90 transition-transform duration-200" />
          {!isCollapsed && (
            <div className="flex-1 flex items-center justify-between">
              <span>New chat</span>
              <kbd className="text-[10px] text-slate-400 font-mono px-1.5 py-0.5 bg-dark-850 rounded border border-dark-700">
                ⌘K
              </kbd>
            </div>
          )}
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-2 space-y-1">
        {!isCollapsed && (
          <div className="px-3 pt-2 pb-1 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Recent Chats
          </div>
        )}

        {conversations.map((conv) => {
          const isActive = conv.id === activeId;
          return (
            <div
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className={`group relative flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-medium cursor-pointer transition-all ${
                isActive
                  ? 'bg-dark-800 text-white shadow-sm border border-dark-700/80'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-dark-850 border border-transparent'
              } ${isCollapsed ? 'justify-center px-0' : ''}`}
            >
              <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-brand-400' : 'text-slate-500'}`} />

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

      {/* RAG Knowledge Base Drawer in Sidebar */}
      {!isCollapsed && ragDocuments.length > 0 && (
        <div className="p-3 border-t border-dark-800/80 bg-dark-950/40">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5 text-[11px] font-semibold text-amber-400/90 uppercase tracking-wider">
              <Database className="w-3.5 h-3.5" />
              <span>Knowledge Base ({ragDocuments.length})</span>
            </div>
            {onClearRagDocuments && (
              <button
                onClick={onClearRagDocuments}
                title="Clear Knowledge Base"
                className="text-[10px] text-slate-500 hover:text-rose-400 transition-colors flex items-center gap-1"
              >
                <Trash className="w-3 h-3" />
                Clear
              </button>
            )}
          </div>

          <div className="space-y-1 max-h-32 overflow-y-auto pr-1">
            {ragDocuments.map((doc, idx) => (
              <div
                key={idx}
                className="group flex items-center justify-between px-2 py-1.5 rounded-lg bg-dark-850/70 border border-dark-800 text-xs text-slate-300"
              >
                <div className="flex items-center gap-1.5 min-w-0 flex-1">
                  <FileText className="w-3 h-3 text-amber-400 shrink-0" />
                  <span className="truncate text-[11px]">{doc.filename}</span>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <span className="text-[10px] font-mono text-slate-500">{doc.chunks}c</span>
                  {onDeleteRagDocument && (
                    <button
                      onClick={() => onDeleteRagDocument(doc.filename)}
                      className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400 p-0.5 rounded"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

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
                  <div className="text-slate-400 text-[10px]">Temp: {temperature.toFixed(2)}</div>
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
              title="Model Parameters"
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-dark-800 transition-colors"
            >
              <Sliders className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}
