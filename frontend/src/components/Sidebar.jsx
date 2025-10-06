import React, { useState } from 'react';
import { Settings, Trash2, Brain, Loader2, AlertCircle } from 'lucide-react';

const Sidebar = ({ onInitialize, onReset, isInitialized, isLoading }) => {
  const [modelType, setModelType] = useState('ollama');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('llama3:latest');  // Use exact model name with tag
  const [error, setError] = useState('');

  const handleInitialize = () => {
    if (modelType === 'openai' && !apiKey.trim()) {
      setError('Please enter an API key for OpenAI');
      return;
    }

    setError('');
    onInitialize(modelType, apiKey, model).catch(err => {
      setError(err.message || 'Failed to initialize');
    });
  };

  const handleModelTypeChange = (e) => {
    const newType = e.target.value;
    setModelType(newType);
    setModel(newType === 'ollama' ? 'llama3:latest' : 'gpt-3.5-turbo');
  };

  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col h-full">
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center gap-3 mb-2">
          <Brain className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-xl font-bold text-gray-900">Dementia Support</h1>
            <p className="text-xs text-gray-500">Multi-Agent AI System</p>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-4 flex-1 overflow-y-auto">
        <div>
          <h2 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Configuration
          </h2>

          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Type
              </label>
              <select
                value={modelType}
                onChange={handleModelTypeChange}
                disabled={isInitialized}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:bg-gray-100"
              >
                <option value="ollama">🆓 Free (Ollama - Local)</option>
                <option value="openai">💳 Paid (OpenAI - Cloud)</option>
              </select>
            </div>

            {modelType === 'openai' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  OpenAI API Key
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  disabled={isInitialized}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:bg-gray-100"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model
              </label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={isInitialized}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:bg-gray-100"
              >
                {modelType === 'ollama' ? (
                  <>
                    <option value="llama3:latest">Llama 3 (Recommended) ✓</option>
                    <option value="llava:latest">Llava ✓</option>
                    <option value="gemma3:270m">Gemma 3 (Fast) ✓</option>
                  </>
                ) : (
                  <>
                    <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    <option value="gpt-4">GPT-4</option>
                    <option value="gpt-4-turbo-preview">GPT-4 Turbo</option>
                  </>
                )}
              </select>
            </div>

            {error && (
              <div className="flex items-start gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="w-4 h-4 text-red-600 mt-0.5" />
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

            {!isInitialized ? (
              <button
                onClick={handleInitialize}
                disabled={isLoading}
                className="w-full bg-primary-600 hover:bg-primary-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Initializing...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4" />
                    Initialize System
                  </>
                )}
              </button>
            ) : (
              <div className="space-y-2">
                <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-sm text-green-800 font-medium">✓ System Initialized</p>
                  <p className="text-xs text-green-700 mt-1">
                    {modelType === 'ollama' ? '🆓 Free' : '💳 Paid'} - {model}
                  </p>
                </div>
                <button
                  onClick={onReset}
                  className="w-full bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  New Conversation
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="pt-4 border-t border-gray-200">
          <h3 className="text-sm font-semibold text-gray-900 mb-3">Specialized Agents</h3>
          <div className="space-y-3">
            <div className="p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full" />
                <p className="font-medium text-sm text-blue-900">Knowledge Agent</p>
              </div>
              <p className="text-xs text-blue-700">Answers questions using research-backed information</p>
            </div>

            <div className="p-3 bg-pink-50 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-pink-500 rounded-full" />
                <p className="font-medium text-sm text-pink-900">Empathy Agent</p>
              </div>
              <p className="text-xs text-pink-700">Provides emotional support and understanding</p>
            </div>

            <div className="p-3 bg-purple-50 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-purple-500 rounded-full" />
                <p className="font-medium text-sm text-purple-900">Cognitive Agent</p>
              </div>
              <p className="text-xs text-purple-700">Offers memory exercises and brain training</p>
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-gray-200">
        <p className="text-xs text-gray-500 text-center">Powered by Multi-Agent AI</p>
      </div>
    </div>
  );
};

export default Sidebar;
