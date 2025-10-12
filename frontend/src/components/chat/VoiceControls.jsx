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
    <div className="flex items-center gap-3">
      <button
        onClick={onToggleVoice}
        className={`btn-voice transition-all duration-300 ${
          voiceEnabled
            ? 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 shadow-lg'
            : 'bg-gradient-to-r from-gray-400 to-gray-500 hover:from-gray-500 hover:to-gray-600'
        }`}
        title={voiceEnabled ? 'Voice Mode: ON' : 'Voice Mode: OFF'}
      >
        {voiceEnabled ? <Volume2 className="w-6 h-6" /> : <VolumeX className="w-6 h-6" />}
      </button>

      {voiceEnabled && (
        <button
          onClick={isListening ? onStopListening : onStartListening}
          disabled={isSpeaking}
          className={`btn-voice transition-all duration-300 ${
            isListening
              ? 'bg-gradient-to-r from-red-500 to-red-600 animate-pulse shadow-xl scale-105'
              : isSpeaking
              ? 'bg-gradient-to-r from-gray-400 to-gray-500 cursor-not-allowed'
              : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 shadow-lg hover:shadow-xl'
          }`}
          title={isListening ? 'Stop listening' : 'Start voice input'}
        >
          {isListening ? (
            <Radio className="w-6 h-6" />
          ) : (
            <Mic className="w-6 h-6" />
          )}
        </button>
      )}

      {isSpeaking && (
        <button
          onClick={onStopSpeaking}
          className="btn-voice bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 shadow-lg"
          title="Stop speaking"
        >
          <MicOff className="w-6 h-6" />
        </button>
      )}
    </div>
  );
};

export default VoiceControls;
