import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Brain, Heart, BookOpen, Activity, MessageSquarePlus } from 'lucide-react';
import { useVoice } from '../hooks/useVoice';
import VoiceControls from './VoiceControls';

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

const ChatInterface = ({ messages, onSendMessage, isLoading, isInitialized, onReset }) => {
  const [input, setInput] = useState('');
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const messagesEndRef = useRef(null);

  const {
    isListening,
    isSpeaking,
    transcript,
    voiceEnabled,
    error: voiceError,
    startListening,
    stopListening,
    speak,
    stopSpeaking,
    toggleVoice,
  } = useVoice();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Update input when voice transcript changes
  useEffect(() => {
    if (transcript) {
      setInput(transcript);
    }
  }, [transcript]);

  // Auto-submit when voice recognition ends
  useEffect(() => {
    if (!isListening && transcript && voiceEnabled) {
      handleSubmit(null);
    }
  }, [isListening, transcript, voiceEnabled]);

  // Speak the latest assistant message
  useEffect(() => {
    if (messages.length > 0 && voiceEnabled) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === 'assistant' && !isLoading) {
        speak(lastMessage.content);
      }
    }
  }, [messages, voiceEnabled, isLoading]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleResetConfirm = () => {
    onReset();
    setShowResetConfirm(false);
  };

  if (!isInitialized) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8">
        <Brain className="w-16 h-16 text-primary-500 mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Welcome to Dementia Support Chatbot
        </h2>
        <p className="text-gray-600 mb-8 max-w-md">
          Please configure your API key in the sidebar to get started
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl w-full">
          <div className="card">
            <BookOpen className="w-8 h-8 text-blue-500 mb-3" />
            <h3 className="font-semibold mb-2">Knowledge Agent</h3>
            <p className="text-sm text-gray-600">
              Ask factual questions about dementia, symptoms, and caregiving
            </p>
          </div>

          <div className="card">
            <Heart className="w-8 h-8 text-pink-500 mb-3" />
            <h3 className="font-semibold mb-2">Empathy Agent</h3>
            <p className="text-sm text-gray-600">
              Share your feelings and receive emotional support
            </p>
          </div>

          <div className="card">
            <Brain className="w-8 h-8 text-purple-500 mb-3" />
            <h3 className="font-semibold mb-2">Cognitive Agent</h3>
            <p className="text-sm text-gray-600">
              Practice memory exercises and brain training activities
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with New Conversation Button */}
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

      {/* Reset Confirmation Modal */}
      {showResetConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md mx-4 shadow-xl">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Start New Conversation?</h3>
            <p className="text-sm text-gray-600 mb-6">
              This will clear your current conversation history. Analytics will no longer be available for this conversation.
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

      {/* Voice Error Display */}
      {voiceError && (
        <div className="bg-yellow-50 border-b border-yellow-200 p-3">
          <p className="text-sm text-yellow-800">{voiceError}</p>
        </div>
      )}

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Brain className="w-12 h-12 text-primary-500 mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              Welcome to Dementia Support
            </h3>
            <p className="text-gray-600 mb-6">
              Ask a question or explore suggested topics below
            </p>

            <div className="space-y-3 text-left max-w-lg w-full">
              <div
                onClick={() => setInput("What are the early signs and symptoms of dementia?")}
                className="p-4 bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl cursor-pointer hover:shadow-md hover:from-blue-100 hover:to-blue-200 transition-all duration-200 border border-blue-200"
              >
                <div className="flex items-start gap-3">
                  <BookOpen className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-blue-900 mb-1">
                      Learn About Dementia
                    </p>
                    <p className="text-xs text-blue-700">
                      Understand early signs and symptoms
                    </p>
                  </div>
                </div>
              </div>
              <div
                onClick={() => setInput("I need support with caregiving challenges")}
                className="p-4 bg-gradient-to-r from-pink-50 to-pink-100 rounded-xl cursor-pointer hover:shadow-md hover:from-pink-100 hover:to-pink-200 transition-all duration-200 border border-pink-200"
              >
                <div className="flex items-start gap-3">
                  <Heart className="w-5 h-5 text-pink-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-pink-900 mb-1">
                      Caregiver Support
                    </p>
                    <p className="text-xs text-pink-700">
                      Get emotional support and guidance
                    </p>
                  </div>
                </div>
              </div>
              <div
                onClick={() => setInput("Please provide a cognitive exercise")}
                className="p-4 bg-gradient-to-r from-purple-50 to-purple-100 rounded-xl cursor-pointer hover:shadow-md hover:from-purple-100 hover:to-purple-200 transition-all duration-200 border border-purple-200"
              >
                <div className="flex items-start gap-3">
                  <Brain className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-purple-900 mb-1">
                      Cognitive Exercises
                    </p>
                    <p className="text-xs text-purple-700">
                      Practice memory and brain training activities
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${
                  message.role === 'user' ? 'justify-end' : 'justify-start'
                } message-enter`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white border border-gray-200 text-gray-900'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  {message.agent && (
                    <div className="flex items-center gap-2 mt-2 pt-2 border-t border-gray-200">
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                          agentColors[message.agent]
                        }`}
                      >
                        {agentIcons[message.agent]}
                        <span className="capitalize">{message.agent}</span>
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3">
                  <Loader2 className="w-5 h-5 animate-spin text-primary-600" />
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form with Voice Controls */}
      <div className="border-t border-gray-200 bg-white p-4">
        {/* Analytics Progress Indicator */}
        {messages.length > 0 && messages.filter(msg => msg.role === 'user').length < 5 && (
          <div className="mb-3 p-2 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-xs text-blue-700 flex items-center justify-between">
              <span>
                📊 Analytics available after {5 - messages.filter(msg => msg.role === 'user').length} more message(s)
              </span>
              <span className="text-blue-600 font-medium">
                {messages.filter(msg => msg.role === 'user').length}/5
              </span>
            </p>
          </div>
        )}

        {/* Voice Status */}
        {isListening && (
          <div className="mb-2 p-2 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-800 flex items-center gap-2">
              <span className="animate-pulse">🎤</span>
              Listening... {transcript && `"${transcript}"`}
            </p>
          </div>
        )}

        {isSpeaking && (
          <div className="mb-2 p-2 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800 flex items-center gap-2">
              <span className="animate-pulse">🔊</span>
              Speaking...
            </p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex gap-3 items-center">
          {/* Voice Controls */}
          <VoiceControls
            isListening={isListening}
            isSpeaking={isSpeaking}
            voiceEnabled={voiceEnabled}
            onToggleVoice={toggleVoice}
            onStartListening={startListening}
            onStopListening={stopListening}
            onStopSpeaking={stopSpeaking}
          />

          {/* Text Input */}
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isListening ? "Listening..." : "Type or speak your message..."}
            disabled={isLoading || isListening}
            className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          />

          {/* Send Button */}
          <button
            type="submit"
            disabled={isLoading || !input.trim() || isListening}
            className="btn-primary px-6 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
