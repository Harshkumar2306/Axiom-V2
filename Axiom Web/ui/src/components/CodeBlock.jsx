import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import hljs from 'highlight.js';
import 'highlight.js/styles/atom-one-dark.min.css';

export default function CodeBlock({ code, language }) {
  const [copied, setCopied] = useState(false);

  const lang = language ? language.toLowerCase() : 'plaintext';
  const validLang = hljs.getLanguage(lang) ? lang : 'plaintext';
  let highlightedCode = code;
  try {
    highlightedCode = hljs.highlight(code, { language: validLang }).value;
  } catch {
    highlightedCode = code;
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-4 rounded-xl overflow-hidden border border-dark-700/80 bg-[#161620] shadow-xl text-sm font-mono">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#1b1b26] border-b border-dark-700/70 text-xs text-slate-400">
        <span className="font-medium tracking-wide uppercase text-[11px] text-slate-300">
          {language || 'text'}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-dark-800/80 hover:bg-dark-700 hover:text-slate-100 text-slate-400 transition-all border border-dark-700/50"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-cyber-emerald" />
              <span className="text-cyber-emerald font-sans">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span className="font-sans">Copy code</span>
            </>
          )}
        </button>
      </div>

      {/* Code content */}
      <div className="p-4 overflow-x-auto text-[13px] leading-relaxed">
        <pre className="!m-0 !p-0 !bg-transparent">
          <code
            className={`hljs language-${validLang}`}
            dangerouslySetInnerHTML={{ __html: highlightedCode }}
          />
        </pre>
      </div>
    </div>
  );
}
