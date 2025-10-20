import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Brain, Heart, BookOpen, Activity, MessageSquarePlus, UserRound } from 'lucide-react';
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
    clearTranscript,
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
      onSendMessage(input, voiceEnabled && isListening); // Pass voice mode flag
      setInput('');
      clearTranscript(); // Clear transcript after sending message
    }
  };

  const handleResetConfirm = () => {
    console.log('Reset confirmed, calling onReset...');
    onReset();
    setShowResetConfirm(false);
    clearTranscript(); // Clear transcript when starting new conversation
  };

  if (!isInitialized) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8 bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <div className="mb-8">
          <div className="relative">
            <Brain className="w-20 h-20 text-blue-600 mb-6 mx-auto animate-pulse" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4 text-xl-large">
            Welcome to Dementia Support
          </h1>
          <p className="text-gray-600 mb-8 max-w-lg text-large">
            Your compassionate AI companion for dementia care and support
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full">
          <div className="card-interactive group">
            <div className="flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                <BookOpen className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Knowledge Agent</h3>
              <p className="text-gray-600 text-large">
                Ask factual questions about dementia, symptoms, and caregiving
              </p>
            </div>
          </div>

          <div className="card-interactive group">
            <div className="flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-pink-500 to-pink-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                <Heart className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Empathy Agent</h3>
              <p className="text-gray-600 text-large">
                Share your feelings and receive emotional support
              </p>
            </div>
          </div>

          <div className="card-interactive group">
            <div className="flex flex-col items-center text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Cognitive Agent</h3>
              <p className="text-gray-600 text-large">
                Practice memory exercises and brain training activities
              </p>
            </div>
          </div>
        </div>

        <div className="mt-12 p-6 bg-gradient-to-r from-blue-100 to-purple-100 rounded-2xl border-2 border-blue-200">
          <p className="text-gray-700 text-large font-medium">
            [Tip] <strong>Getting Started:</strong> Configure your API key in the sidebar to begin your journey
          </p>
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
            onClick={() => {
              console.log('New conversation button clicked');
              setShowResetConfirm(true);
            }}
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
          <div className="flex flex-col items-center justify-center h-full text-center p-6">
            <div className="mb-8">
              <div className="relative mb-6">
                <Brain className="w-16 h-16 text-blue-600 mx-auto animate-pulse" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-3 text-xl-large">
                Welcome to Dementia Support
              </h3>
              <p className="text-gray-600 mb-8 text-large">
                Ask a question or explore suggested topics below
              </p>
            </div>

            <div className="space-y-4 text-left max-w-2xl w-full">
              <div
                onClick={() => setInput("What are the early signs and symptoms of dementia?")}
                className="suggested-topic bg-gradient-to-r from-blue-50 to-blue-100 hover:from-blue-100 hover:to-blue-200 border-blue-200 group"
              >
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300 flex-shrink-0">
                    <BookOpen className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h4 className="text-lg font-bold text-blue-900 mb-2">
                      Learn About Dementia
                    </h4>
                    <p className="text-blue-700 text-large">
                      Understand early signs, symptoms, and care strategies
                    </p>
                  </div>
                  <div className="text-blue-400 group-hover:text-blue-600 transition-colors duration-300">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </div>

              <div
                onClick={() => setInput("I need support with caregiving challenges")}
                className="suggested-topic bg-gradient-to-r from-pink-50 to-pink-100 hover:from-pink-100 hover:to-pink-200 border-pink-200 group"
              >
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-pink-500 to-pink-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300 flex-shrink-0">
                    <Heart className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h4 className="text-lg font-bold text-pink-900 mb-2">
                      Caregiver Support
                    </h4>
                    <p className="text-pink-700 text-large">
                      Get emotional support and practical guidance
                    </p>
                  </div>
                  <div className="text-pink-400 group-hover:text-pink-600 transition-colors duration-300">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </div>

              <div
                onClick={() => setInput("Please provide a cognitive exercise")}
                className="suggested-topic bg-gradient-to-r from-purple-50 to-purple-100 hover:from-purple-100 hover:to-purple-200 border-purple-200 group"
              >
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300 flex-shrink-0">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div className="flex-1">
                    <h4 className="text-lg font-bold text-purple-900 mb-2">
                      Cognitive Exercises
                    </h4>
                    <p className="text-purple-700 text-large">
                      Practice memory and brain training activities
                    </p>
                  </div>
                  <div className="text-purple-400 group-hover:text-purple-600 transition-colors duration-300">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message, index) => {
              // Hide assistant messages if they are voice-only responses
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
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form with Voice Controls */}
      <div className="border-t border-gray-200 bg-gradient-to-r from-white to-gray-50 p-6">
        {/* Analytics Progress Indicator */}
        {messages.length > 0 && messages.filter(msg => msg.role === 'user').length < 5 && (
          <div className="mb-4 p-4 bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-200 rounded-xl">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                  <span className="text-white font-bold text-sm">[Analytics]</span>
                </div>
                <div>
                  <p className="text-blue-800 font-medium text-large">
                    Analytics available after {5 - messages.filter(msg => msg.role === 'user').length} more message(s)
                  </p>
                </div>
              </div>
              <div className="bg-blue-200 rounded-full px-3 py-1">
                <span className="text-blue-800 font-bold text-lg">
                  {messages.filter(msg => msg.role === 'user').length}/5
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Therapist Avatar - Shows when voice is enabled */}
        {voiceEnabled && (
          <div className="mb-4 flex items-center justify-center">
            <div className="bg-gradient-to-br from-pink-100 to-purple-100 border-2 border-pink-300 rounded-2xl p-4 shadow-lg">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <div className="w-16 h-16 bg-gradient-to-br from-pink-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg">
                    <UserRound className="w-10 h-10 text-white" />
                  </div>
                  {(isListening || isSpeaking) && (
                    <div className="absolute -bottom-1 -right-1 w-5 h-5 bg-green-500 rounded-full border-2 border-white"></div>
                  )}
                </div>
                <div className="text-left">
                  <h4 className="text-lg font-bold text-gray-900">Therapeutic Support</h4>
                  <p className="text-sm text-gray-600">Voice mode active</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Voice Status */}
        {isListening && (
          <div className="mb-4 status-indicator bg-gradient-to-r from-red-50 to-red-100 border-2 border-red-200 text-red-800">
            <div className="w-6 h-6 bg-red-500 rounded-full animate-pulse"></div>
            <span className="font-semibold">Listening...</span>
            {transcript && (
              <span className="text-red-600 italic">"{transcript}"</span>
            )}
          </div>
        )}

        {isSpeaking && (
          <div className="mb-4 status-indicator bg-gradient-to-r from-blue-50 to-blue-100 border-2 border-blue-200 text-blue-800">
            <div className="w-6 h-6 bg-blue-500 rounded-full animate-pulse"></div>
            <span className="font-semibold">Speaking...</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex gap-4 items-end">
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
          <div className="flex-1 relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={isListening ? "Listening..." : "Type or speak your message..."}
              disabled={isLoading || isListening}
              className="input-field"
            />
          </div>

          {/* Send Button */}
          <button
            type="submit"
            disabled={isLoading || !input.trim() || isListening}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 min-w-[120px]"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Sending</span>
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                <span>Send</span>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
