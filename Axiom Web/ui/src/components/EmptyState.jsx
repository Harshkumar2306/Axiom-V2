import React from 'react';
import { Zap } from 'lucide-react';
import { motion } from 'framer-motion';

export default function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] max-w-2xl mx-auto px-4 text-center select-none">
      {/* Emblem with glow */}
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="relative mb-6"
      >
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-brand-600 to-cyber-cyan opacity-35 blur-2xl animate-pulse-subtle"></div>
        <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-br from-dark-800 to-dark-900 border border-brand-500/40 flex items-center justify-center shadow-2xl">
          <Zap className="w-8 h-8 text-brand-500 fill-brand-500/30" />
        </div>
      </motion.div>

      {/* Title */}
      <motion.div
        initial={{ y: 15, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.15, duration: 0.5 }}
      >
        <h2 className="text-3xl sm:text-4xl font-display font-semibold tracking-tight text-white mb-2">
          Axiom
        </h2>
        <p className="text-slate-400 text-sm sm:text-base max-w-md mx-auto leading-relaxed">
          How can I help you today?
        </p>
      </motion.div>
    </div>
  );
}
