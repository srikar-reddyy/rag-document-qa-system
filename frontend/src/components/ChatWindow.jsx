/**
 * ChatWindow Component
 * ChatGPT-style chat interface with message history
 */

import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, History, Trash2 } from 'lucide-react';
import MessageBubble from './MessageBubble';
import { sendChatMessageStream, clearChatHistory } from '../services/api';

const looksLikeIncompleteListTail = (text) => {
  if (!text) return false;
  const tail = text.split('\n').pop()?.trim() || '';
  if (!tail) return false;
  if (!/^(?:[-*]|\d+\.)\s+/.test(tail)) return false;
  return !/[.!?]\s*$/.test(tail);
};

const getFlushBoundary = (buffer) => {
  if (!buffer) return -1;

  const lastNewline = buffer.lastIndexOf('\n');
  let sentenceEnd = -1;
  const regex = /[.!?](?=\s|$)/g;
  let match = regex.exec(buffer);
  while (match) {
    sentenceEnd = match.index + 1;
    match = regex.exec(buffer);
  }

  let boundary = Math.max(lastNewline >= 0 ? lastNewline + 1 : -1, sentenceEnd);
  if (boundary <= 0) return -1;

  const prefix = buffer.slice(0, boundary);
  if (looksLikeIncompleteListTail(prefix)) {
    const prevNewline = prefix.slice(0, -1).lastIndexOf('\n');
    boundary = prevNewline >= 0 ? prevNewline + 1 : -1;
  }

  return boundary;
};

const ChatWindow = ({ initialMessages = [], isLoadingState = false, selectedDocumentIds = [], onHighlightClick }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const streamBufferRef = useRef('');

  // Load initial messages on mount
  useEffect(() => {
    if (initialMessages && initialMessages.length > 0 && messages.length === 0) {
      setMessages(initialMessages);
    }
  }, [initialMessages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    // Validate document selection
    if (!selectedDocumentIds || selectedDocumentIds.length === 0) {
      setMessages(prev => [
        ...prev,
        { role: 'error', content: 'Please select at least one document to query.' }
      ]);
      return;
    }

    const userMessage = input.trim();
    setInput('');
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    // Add placeholder assistant message for progressive streaming
    setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [], highlights: [], streaming: true }]);
    streamBufferRef.current = '';
    setLoading(true);

    try {
      // Call streaming endpoint and update message progressively
      const finalText = await sendChatMessageStream(userMessage, selectedDocumentIds, (chunk) => {
        streamBufferRef.current += chunk;
        let boundary = getFlushBoundary(streamBufferRef.current);
        if (boundary <= 0) return;

        const flushText = streamBufferRef.current.slice(0, boundary);
        streamBufferRef.current = streamBufferRef.current.slice(boundary);

        setMessages(prev => {
          const next = [...prev];
          for (let i = next.length - 1; i >= 0; i -= 1) {
            if (next[i]?.role === 'assistant' && next[i]?.streaming) {
              next[i] = {
                ...next[i],
                content: `${next[i].content || ''}${flushText}`,
              };
              break;
            }
          }
          return next;
        });
      });

      // Mark stream complete
      setMessages(prev => {
        const next = [...prev];
        for (let i = next.length - 1; i >= 0; i -= 1) {
          if (next[i]?.role === 'assistant' && next[i]?.streaming) {
            const remaining = streamBufferRef.current;
            streamBufferRef.current = '';
            next[i] = {
              ...next[i],
              content: finalText || `${next[i].content || ''}${remaining}` || 'No answer generated.',
              streaming: false,
            };
            break;
          }
        }
        return next;
      });
    } catch (error) {
      setMessages(prev => {
        const next = [...prev];
        let replaced = false;
        for (let i = next.length - 1; i >= 0; i -= 1) {
          if (next[i]?.role === 'assistant' && next[i]?.streaming) {
            next[i] = { role: 'error', content: `Error: ${error.message}` };
            replaced = true;
            break;
          }
        }
        if (!replaced) {
          next.push({ role: 'error', content: `Error: ${error.message}` });
        }
        return next;
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = async () => {
    if (window.confirm('Are you sure you want to clear chat history?')) {
      try {
        await clearChatHistory();
        setMessages([]);
      } catch (error) {
        console.error('Error clearing chat history:', error);
      }
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 bg-white border-b border-gray-200 shadow-sm">
        <div>
          <h2 className="text-lg font-semibold text-gray-800">AI Assistant</h2>
          <p className="text-xs text-gray-500">
            {selectedDocumentIds.length > 0 
              ? `Querying ${selectedDocumentIds.length} selected document(s)`
              : 'Select documents to query'}
          </p>
        </div>
        <div className="flex items-center space-x-3">
          {messages.length > 0 && (
            <button
              onClick={handleClearChat}
              className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
              title="Clear chat history"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${loading ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'}`}></div>
            <span className="text-xs text-gray-500">{loading ? 'Thinking...' : 'Ready'}</span>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="p-6 bg-white rounded-2xl shadow-sm max-w-md">
              <div className="w-16 h-16 bg-gradient-to-br from-primary-400 to-purple-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Send className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">
                Start a Conversation
              </h3>
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, index) => (
              <MessageBubble key={index} message={msg} onHighlightClick={onHighlightClick} />
            ))}
            {loading && (
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <Loader2 className="w-5 h-5 text-white animate-spin" />
                </div>
                <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white px-6 py-4">
        <div className="flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Shift+Enter for new line)"
              disabled={loading}
              rows={1}
              className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400 transition-all max-h-32 overflow-y-auto"
            />
          </div>
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="flex-shrink-0 p-3 bg-primary-500 text-white rounded-xl hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-sm hover:shadow-md disabled:hover:shadow-sm"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2">
          Phase 2: Responses will include document citations and sources
        </p>
      </div>
    </div>
  );
};

export default ChatWindow;
