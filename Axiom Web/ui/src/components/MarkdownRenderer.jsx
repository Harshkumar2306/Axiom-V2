import React, { useMemo } from 'react';
import { marked } from 'marked';
import CodeBlock from './CodeBlock';

// Configure marked options
marked.setOptions({
  gfm: true,
  breaks: true,
});

export default function MarkdownRenderer({ content }) {
  const renderedContent = useMemo(() => {
    if (!content) return null;

    try {
      const tokens = marked.lexer(content);
      const elements = [];
      let textTokens = [];

      const flushTextTokens = (key) => {
        if (textTokens.length > 0) {
          const html = marked.parser(textTokens);
          elements.push(
            <div
              key={`text-${key}`}
              className="markdown-body"
              dangerouslySetInnerHTML={{ __html: html }}
            />
          );
          textTokens = [];
        }
      };

      tokens.forEach((token, index) => {
        if (token.type === 'code') {
          flushTextTokens(index);
          elements.push(
            <CodeBlock
              key={`code-${index}`}
              code={token.text}
              language={token.lang}
            />
          );
        } else {
          textTokens.push(token);
        }
      });

      flushTextTokens('final');
      return elements;
    } catch (err) {
      // Fallback in case of lexer error
      return (
        <div
          className="markdown-body"
          dangerouslySetInnerHTML={{ __html: marked.parse(content || '') }}
        />
      );
    }
  }, [content]);

  return <div className="space-y-2">{renderedContent}</div>;
}
