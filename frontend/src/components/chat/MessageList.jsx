import React, { useEffect, useRef } from 'react';
import { Brain, Heart, BookOpen, Activity, Loader2 } from 'lucide-react';

const agentIcons = {
  knowledge: <BookOpen className="w-4 h-4" />,
  empathy: <Heart className="w-4 h-4" />,
  cognitive: <Brain className="w-4 h-4" />,
  system: <Activity className="w-4 h-4" />,
};

const agentColors = {
  knowledge: 'bg-blue-100 text-blue-800',
  empathy: 'bg-pink-100 text-pink-800',
  cognitive: 'bg-purple-100 text-purple-800',
  system: 'bg-gray-100 text-gray-800',
};

const MessageList = ({ messages, isLoading }) => {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-4">
      {messages.map((message, index) => {
        if (message.role === 'assistant' && message.isVoiceOnly) {
          return null;
        }

        return (
          <div
            key={index}
            className={`flex ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            } message-enter`}
          >
            <div
              className={`message-bubble ${
                message.role === 'user'
                  ? 'message-user'
                  : 'message-assistant'
              }`}
            >
              <p className="whitespace-pre-wrap text-large">{message.content}</p>
              {message.agent && (
                <div className="flex items-center gap-2 mt-3 pt-3 border-t border-gray-200">
                  <span
                    className={`agent-badge ${
                      agentColors[message.agent]
                    }`}
                  >
                    {agentIcons[message.agent]}
                    <span className="capitalize font-semibold">{message.agent}</span>
                  </span>
                </div>
              )}
            </div>
          </div>
        );
      })}

      {isLoading && (
        <div className="flex justify-start">
          <div className="message-bubble message-assistant">
            <div className="flex items-center gap-3">
              <div className="loading-dots">
                <div></div>
                <div></div>
                <div></div>
              </div>
              <span className="text-gray-600 font-medium">Thinking...</span>
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;
