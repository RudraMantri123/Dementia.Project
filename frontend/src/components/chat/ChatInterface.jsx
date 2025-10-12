import React, { useState } from 'react';
import { Brain, MessageSquarePlus } from 'lucide-react';
import { useVoice } from '../../hooks/useVoice';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import WelcomeScreen from './WelcomeScreen';

const ChatInterface = ({ messages, onSendMessage, isLoading, isInitialized, onReset }) => {
  const [input, setInput] = useState('');
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  const voiceHook = useVoice();
  const { speak, voiceEnabled, clearTranscript } = voiceHook;

  React.useEffect(() => {
    if (messages.length > 0 && voiceEnabled) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === 'assistant' && !isLoading) {
        speak(lastMessage.content);
      }
    }
  }, [messages, voiceEnabled, isLoading]);

  const handleResetConfirm = () => {
    onReset();
    setShowResetConfirm(false);
    clearTranscript();
  };

  const handleSelectTopic = (prompt) => {
    setInput(prompt);
  };

  if (!isInitialized) {
    return <WelcomeScreen onSelectTopic={handleSelectTopic} hasMessages={false} />;
  }

  return (
    <div className="flex flex-col h-full">
      {messages.length > 0 && (
        <div className="border-b border-gray-200 bg-white px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary-600" />
            <h2 className="font-semibold text-gray-900">Conversation</h2>
            <span className="text-xs text-gray-500">
              {messages.filter(msg => msg.role === 'user').length} message{messages.filter(msg => msg.role === 'user').length !== 1 ? 's' : ''}
            </span>
          </div>
          <button
            onClick={() => setShowResetConfirm(true)}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-700 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors duration-200"
          >
            <MessageSquarePlus className="w-4 h-4" />
            New Conversation
          </button>
        </div>
      )}

      {showResetConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md mx-4 shadow-xl">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Start New Conversation?</h3>
            <p className="text-sm text-gray-600 mb-6">
              This will clear your current conversation history.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowResetConfirm(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg transition-colors duration-200"
              >
                Cancel
              </button>
              <button
                onClick={handleResetConfirm}
                className="px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors duration-200"
              >
                Start New
              </button>
            </div>
          </div>
        </div>
      )}

      {messages.length === 0 ? (
        <WelcomeScreen onSelectTopic={handleSelectTopic} hasMessages={true} />
      ) : (
        <MessageList messages={messages} isLoading={isLoading} />
      )}

      <MessageInput
        onSendMessage={onSendMessage}
        isLoading={isLoading}
        voiceHook={voiceHook}
      />
    </div>
  );
};

export default ChatInterface;
