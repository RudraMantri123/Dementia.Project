import React, { useState, useEffect } from 'react';
import { Send, Loader2, UserRound } from 'lucide-react';
import VoiceControls from './VoiceControls';

const MessageInput = ({
  onSendMessage,
  isLoading,
  voiceHook,
}) => {
  const [input, setInput] = useState('');

  const {
    isListening,
    isSpeaking,
    transcript,
    voiceEnabled,
    error: voiceError,
    startListening,
    stopListening,
    stopSpeaking,
    toggleVoice,
    clearTranscript,
  } = voiceHook;

  useEffect(() => {
    if (transcript) {
      setInput(transcript);
    }
  }, [transcript]);

  useEffect(() => {
    if (!isListening && transcript && voiceEnabled) {
      handleSubmit(null);
    }
  }, [isListening, transcript, voiceEnabled]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input, voiceEnabled && isListening);
      setInput('');
      clearTranscript();
    }
  };

  return (
    <div className="border-t border-gray-200 bg-gradient-to-r from-white to-gray-50 p-6">
      {voiceError && (
        <div className="mb-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <p className="text-sm text-yellow-800">{voiceError}</p>
        </div>
      )}

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
        <VoiceControls
          isListening={isListening}
          isSpeaking={isSpeaking}
          voiceEnabled={voiceEnabled}
          onToggleVoice={toggleVoice}
          onStartListening={startListening}
          onStopListening={stopListening}
          onStopSpeaking={stopSpeaking}
        />

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
  );
};

export default MessageInput;
