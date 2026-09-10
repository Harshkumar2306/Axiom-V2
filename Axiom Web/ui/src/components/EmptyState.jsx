import React from 'react';
import { Zap, Code, Brain, BookOpen, Sparkles, Terminal } from 'lucide-react';
import { motion } from 'framer-motion';

const SUGGESTIONS = [
  {
    icon: Code,
    title: 'Python Algorithm',
    desc: 'Write a Python function to check if a number is prime with optimized O(sqrt(n)) time.',
    prompt: 'Write a python function to check if a number is prime with step-by-step logic.',
    color: 'from-blue-500/20 to-indigo-500/20 text-blue-400 border-blue-500/30'
  },
  {
    icon: Brain,
    title: 'Multi-Step Logic',
    desc: 'Solve a sequential arithmetic riddle step-by-step.',
    prompt: 'I have 3 apples. I eat 1, and give 1 to a friend. How many apples do I have left? Show the step-by-step math.',
    color: 'from-emerald-500/20 to-teal-500/20 text-emerald-400 border-emerald-500/30'
  },
  {
    icon: BookOpen,
    title: 'Transformer Architecture',
    desc: 'Explain how Rotary Positional Embeddings (RoPE) and GQA work.',
    prompt: 'Explain Grouped-Query Attention (GQA) and Rotary Positional Embeddings (RoPE) in LLMs.',
    color: 'from-purple-500/20 to-violet-500/20 text-purple-400 border-purple-500/30'
  },
  {
    icon: Terminal,
    title: 'Solar System Facts',
    desc: 'Recall planetary science and astronomy foundations.',
    prompt: 'What is the solar system and what is at its center?',
    color: 'from-amber-500/20 to-orange-500/20 text-amber-400 border-amber-500/30'
  }
];

export default function EmptyState({ onSelectPrompt }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] max-w-3xl mx-auto px-4 text-center">
      {/* Emblem with glow */}
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="relative mb-6"
      >
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-brand-600 to-cyber-cyan opacity-40 blur-xl animate-pulse-subtle"></div>
        <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-dark-800 to-dark-900 border border-brand-500/40 flex items-center justify-center shadow-2xl">
          <Zap className="w-8 h-8 text-brand-500 fill-brand-500/30" />
        </div>
      </motion.div>

      {/* Main Title */}
      <motion.div
        initial={{ y: 15, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.15, duration: 0.5 }}
      >
        <h2 className="text-3xl sm:text-4xl font-display font-semibold tracking-tight text-white mb-2">
          Axiom Intelligence
        </h2>
        <p className="text-slate-400 text-sm sm:text-base max-w-md mx-auto mb-10 leading-relaxed">
          476M parameter neural model trained from scratch with GQA, RoPE, and Supervised Fine-Tuning.
        </p>
      </motion.div>

      {/* Suggestion Cards */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.25, duration: 0.5 }}
        className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 w-full text-left"
      >
        {SUGGESTIONS.map((item, idx) => {
          const Icon = item.icon;
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.prompt)}
              className="group p-4 rounded-xl bg-dark-900/60 hover:bg-dark-850/90 border border-dark-750 hover:border-brand-500/40 transition-all duration-200 shadow-sm hover:shadow-brand-500/5 hover:-translate-y-0.5 flex flex-col justify-between"
            >
              <div className="flex items-center gap-3 mb-2">
                <div className={`p-2 rounded-lg bg-gradient-to-br border ${item.color}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <span className="text-sm font-medium text-slate-200 group-hover:text-brand-500 transition-colors">
                  {item.title}
                </span>
              </div>
              <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                {item.desc}
              </p>
            </button>
          );
        })}
      </motion.div>
    </div>
  );
}
