import React from 'react';
import { X, Sliders, RotateCcw, Check, Globe } from 'lucide-react';
import { motion } from 'framer-motion';

export default function SettingsModal({
  isOpen,
  onClose,
  settings,
  onSaveSettings,
  onResetSettings
}) {
  if (!isOpen) return null;

  const getTempDescription = (temp) => {
    if (temp <= 0.2) return 'Deterministic & Logical (Optimal for Code & Math)';
    if (temp <= 0.6) return 'Balanced & Analytical';
    return 'Creative & Divergent';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="w-full max-w-lg rounded-2xl bg-dark-900 border border-dark-700/80 shadow-2xl overflow-hidden flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-dark-800 bg-dark-850">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 rounded-lg bg-brand-500/10 text-brand-500 border border-brand-500/20">
              <Sliders className="w-4 h-4" />
            </div>
            <h3 className="font-semibold text-slate-100 text-base">Inference Parameters</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-dark-750 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6 max-h-[75vh] overflow-y-auto">
          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-200">
                Temperature
              </label>
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-dark-800 text-brand-500 border border-dark-700">
                {settings.temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={settings.temperature}
              onChange={(e) =>
                onSaveSettings({ ...settings, temperature: parseFloat(e.target.value) })
              }
              className="w-full h-1.5 bg-dark-750 rounded-lg appearance-none cursor-pointer accent-brand-500"
            />
            <p className="text-xs text-slate-400">
              {getTempDescription(settings.temperature)}
            </p>
          </div>

          {/* Max Tokens */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-200">
                Max Output Tokens
              </label>
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-dark-800 text-brand-500 border border-dark-700">
                {settings.maxTokens}
              </span>
            </div>
            <input
              type="range"
              min="64"
              max="1024"
              step="32"
              value={settings.maxTokens}
              onChange={(e) =>
                onSaveSettings({ ...settings, maxTokens: parseInt(e.target.value, 10) })
              }
              className="w-full h-1.5 bg-dark-750 rounded-lg appearance-none cursor-pointer accent-brand-500"
            />
            <p className="text-xs text-slate-400">
              Controls maximum length of generated responses (64 - 1024 tokens).
            </p>
          </div>

          {/* Backend API Endpoint */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-200 flex items-center gap-1.5">
              <Globe className="w-3.5 h-3.5 text-cyber-cyan" />
              Backend API Base URL
            </label>
            <input
              type="text"
              value={settings.apiBaseUrl}
              onChange={(e) =>
                onSaveSettings({ ...settings, apiBaseUrl: e.target.value })
              }
              placeholder="http://localhost:8000"
              className="w-full px-3.5 py-2.5 rounded-xl bg-dark-800 border border-dark-700 text-slate-200 text-sm font-mono focus:border-brand-500 focus:outline-none transition-colors"
            />
            <p className="text-xs text-slate-400">
              Point to a deployed cloud server or your local FastAPI instance.
            </p>
          </div>

          {/* System Prompt */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-200">
              System Instruction
            </label>
            <textarea
              rows={3}
              value={settings.systemPrompt}
              onChange={(e) =>
                onSaveSettings({ ...settings, systemPrompt: e.target.value })
              }
              className="w-full p-3 rounded-xl bg-dark-800 border border-dark-700 text-slate-200 text-xs leading-relaxed focus:border-brand-500 focus:outline-none transition-colors resize-none"
            />
            <p className="text-xs text-slate-400">
              Preconditioning prompt injected into the model's ChatML template.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 bg-dark-850 border-t border-dark-800">
          <button
            onClick={onResetSettings}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-dark-800 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Defaults
          </button>
          <button
            onClick={onClose}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-brand-500 hover:bg-brand-600 text-white text-xs font-medium transition-colors shadow-lg shadow-brand-500/20"
          >
            <Check className="w-3.5 h-3.5" />
            Apply Changes
          </button>
        </div>
      </motion.div>
    </div>
  );
}
