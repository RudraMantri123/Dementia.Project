import React from 'react';
import { Mic, MicOff, Volume2, VolumeX, Radio } from 'lucide-react';

const VoiceControls = ({
  isListening,
  isSpeaking,
  voiceEnabled,
  onToggleVoice,
  onStartListening,
  onStopListening,
  onStopSpeaking
}) => {
  return (
    <div className="flex items-center gap-2">
      {/* Voice Mode Toggle */}
      <button
        onClick={onToggleVoice}
        className={`p-2 rounded-lg transition-colors ${
          voiceEnabled
            ? 'bg-green-100 text-green-700 hover:bg-green-200'
            : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
        }`}
        title={voiceEnabled ? 'Voice Mode: ON' : 'Voice Mode: OFF'}
      >
        {voiceEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
      </button>

      {/* Microphone Button */}
      {voiceEnabled && (
        <button
          onClick={isListening ? onStopListening : onStartListening}
          disabled={isSpeaking}
          className={`p-2 rounded-lg transition-all ${
            isListening
              ? 'bg-red-500 text-white animate-pulse'
              : isSpeaking
              ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
              : 'bg-primary-600 text-white hover:bg-primary-700'
          }`}
          title={isListening ? 'Stop listening' : 'Start voice input'}
        >
          {isListening ? (
            <Radio className="w-5 h-5" />
          ) : (
            <Mic className="w-5 h-5" />
          )}
        </button>
      )}

      {/* Stop Speaking Button */}
      {isSpeaking && (
        <button
          onClick={onStopSpeaking}
          className="p-2 rounded-lg bg-orange-100 text-orange-700 hover:bg-orange-200 transition-colors"
          title="Stop speaking"
        >
          <MicOff className="w-5 h-5" />
        </button>
      )}
    </div>
  );
};

export default VoiceControls;
