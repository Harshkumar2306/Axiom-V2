import React, { useMemo } from 'react';
import { User, Brain } from 'lucide-react';
import { marked } from 'marked';
import hljs from 'highlight.js';
import { motion } from 'framer-motion';
import 'highlight.js/styles/tokyo-night-dark.min.css';

marked.setOptions({
  highlight: function (code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
});

export default function ChatBubble({ role, content, isStreaming }) {
  const isUser = role === 'user';

  const renderedContent = useMemo(() => {
    if (isUser) return content;
    return marked.parse(content);
  }, [content, isUser]);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: 'spring', stiffness: 260, damping: 20 }}
      className={`message ${isUser ? 'user-msg' : 'axiom-msg'}`}
    >
      <div className="avatar">
        {isUser ? <User size={18} color="#cbd5e1" /> : <Brain size={18} color="#7b87ff" />}
      </div>
      <div className="content">
        {isUser ? (
          <div>{content}</div>
        ) : (
          <div>
            <span dangerouslySetInnerHTML={{ __html: renderedContent }} />
            {isStreaming && <span className="cursor-blink"></span>}
          </div>
        )}
      </div>
    </motion.div>
  );
}
