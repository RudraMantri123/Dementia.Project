import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import { chatAPI } from './services/api';

function App() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [stats, setStats] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);

  // Count user messages
  const userMessageCount = messages.filter(msg => msg.role === 'user').length;
  const canShowAnalytics = userMessageCount >= 5;

  // Fetch analytics when explicitly shown
  useEffect(() => {
    if (!isInitialized || !showAnalytics) return;

    const fetchData = async () => {
      setAnalyticsLoading(true);
      try {
        const [statsData, analyticsData] = await Promise.all([
          chatAPI.getStats(),
          chatAPI.getAnalytics(),
        ]);
        setStats(statsData);
        setAnalytics(analyticsData);
      } catch (error) {
        console.error('Error fetching analytics:', error);
      } finally {
        setAnalyticsLoading(false);
      }
    };

    fetchData();
  }, [isInitialized, showAnalytics]);

  const handleInitialize = async (modelType, apiKey, model) => {
    setIsLoading(true);
    try {
      await chatAPI.initialize(modelType, apiKey, model);
      setIsInitialized(true);
      setMessages([]);
    } catch (error) {
      console.error('Initialization error:', error);
      throw new Error(
        error.response?.data?.detail || 'Failed to initialize chatbot'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (message, isVoiceInput = false) => {
    // Add user message
    setMessages((prev) => [...prev, { role: 'user', content: message }]);
    setIsLoading(true);

    try {
      const response = await chatAPI.sendMessage(message);

      // Add assistant message (hide from display if voice input)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.response,
          agent: response.agent,
          intent: response.intent,
          isVoiceOnly: isVoiceInput, // Mark as voice-only response
        },
      ]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
          agent: 'system',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      console.log('Starting reset process...');
      await chatAPI.resetConversation();
      console.log('Reset API call successful');
      setMessages([]);
      setStats(null);
      setAnalytics(null);
      setShowAnalytics(false);
      console.log('Reset completed successfully');
    } catch (error) {
      console.error('Reset error:', error);
    }
  };

  const handleShowAnalytics = () => {
    setShowAnalytics(true);
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Sidebar */}
      <Sidebar
        onInitialize={handleInitialize}
        onReset={handleReset}
        isInitialized={isInitialized}
        isLoading={isLoading}
        canShowAnalytics={canShowAnalytics}
        onShowAnalytics={handleShowAnalytics}
      />

      {/* Main Chat Area or Full Analytics Page */}
      {showAnalytics ? (
        <div className="flex-1 flex flex-col min-w-0 bg-white overflow-y-auto">
          {/* Analytics Header */}
          <div className="border-b border-gray-200 bg-white px-6 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
            <div className="flex items-center gap-3">
              <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <h2 className="text-2xl font-bold text-gray-900">Conversation Analytics</h2>
            </div>
            <button
              onClick={() => setShowAnalytics(false)}
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium py-2 px-6 rounded-lg transition-colors duration-200 flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Close Analytics
            </button>
          </div>

          {/* Full Analytics Dashboard */}
          <div className="p-8">
            <AnalyticsDashboard stats={stats} analytics={analytics} isLoading={analyticsLoading} />
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col min-w-0">
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            isInitialized={isInitialized}
            onReset={handleReset}
          />
        </div>
      )}
    </div>
  );
}

export default App;
