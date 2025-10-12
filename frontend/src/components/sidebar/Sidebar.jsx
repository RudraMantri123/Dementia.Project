import React, { useState } from 'react';
import { Settings, Trash2, Brain, Loader2, AlertCircle } from 'lucide-react';
import ThemeToggle from '../common/ThemeToggle';

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
    <div className="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col h-full">
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-primary-600 dark:text-primary-400" />
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">Dementia Support</h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">Multi-Agent AI System</p>
            </div>
          </div>
          <ThemeToggle />
        </div>
      </div>

      <div className="p-6 space-y-4 flex-1 overflow-y-auto">
        <div>
          <h2 className="text-sm font-semibold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Configuration
          </h2>

          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Model Type
              </label>
              <select
                value={modelType}
                onChange={handleModelTypeChange}
                disabled={isInitialized}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="ollama">[Free] Ollama - Local</option>
                <option value="openai">[Paid] OpenAI - Cloud</option>
              </select>
            </div>

            {modelType === 'openai' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  OpenAI API Key
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  disabled={isInitialized}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Model
              </label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={isInitialized}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:bg-gray-100 dark:disabled:bg-gray-700 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                {modelType === 'ollama' ? (
                  <>
                    <option value="llama3:latest">Llama 3 (Recommended)</option>
                    <option value="llava:latest">Llava</option>
                    <option value="gemma3:270m">Gemma 3 (Fast)</option>
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
              <div className="flex items-start gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <AlertCircle className="w-4 h-4 text-red-600 dark:text-red-400 mt-0.5" />
                <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
              </div>
            )}

            {!isInitialized ? (
              <button
                onClick={handleInitialize}
                disabled={isLoading}
                className="w-full bg-primary-600 hover:bg-primary-700 dark:bg-primary-500 dark:hover:bg-primary-600 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
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
                <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
                  <p className="text-sm text-green-800 dark:text-green-300 font-medium">[Success] System Initialized</p>
                  <p className="text-xs text-green-700 dark:text-green-400 mt-1">
                    {modelType === 'ollama' ? '[Free]' : '[Paid]'} - {model}
                  </p>
                </div>
                <button
                  onClick={onReset}
                  className="w-full bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  New Conversation
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Specialized Agents</h3>
          <div className="space-y-3">
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-blue-500 dark:bg-blue-400 rounded-full" />
                <p className="font-medium text-sm text-blue-900 dark:text-blue-300">Knowledge Agent</p>
              </div>
              <p className="text-xs text-blue-700 dark:text-blue-400">Answers questions using research-backed information</p>
            </div>

            <div className="p-3 bg-pink-50 dark:bg-pink-900/20 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-pink-500 dark:bg-pink-400 rounded-full" />
                <p className="font-medium text-sm text-pink-900 dark:text-pink-300">Empathy Agent</p>
              </div>
              <p className="text-xs text-pink-700 dark:text-pink-400">Provides emotional support and understanding</p>
            </div>

            <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-purple-500 dark:bg-purple-400 rounded-full" />
                <p className="font-medium text-sm text-purple-900 dark:text-purple-300">Cognitive Agent</p>
              </div>
              <p className="text-xs text-purple-700 dark:text-purple-400">Offers memory exercises and brain training</p>
            </div>
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-xs text-gray-500 dark:text-gray-400 text-center">Powered by Multi-Agent AI</p>
      </div>
    </div>
  );
};

export default Sidebar;
