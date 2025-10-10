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

  // Count user messages
  const userMessageCount = messages.filter(msg => msg.role === 'user').length;
  const canShowAnalytics = userMessageCount >= 5;

  // Fetch analytics when explicitly shown
  useEffect(() => {
    if (!isInitialized || !showAnalytics) return;

    const fetchData = async () => {
      try {
        const [statsData, analyticsData] = await Promise.all([
          chatAPI.getStats(),
          chatAPI.getAnalytics(),
        ]);
        setStats(statsData);
        setAnalytics(analyticsData);
      } catch (error) {
        console.error('Error fetching analytics:', error);
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

  const handleSendMessage = async (message) => {
    // Add user message
    setMessages((prev) => [...prev, { role: 'user', content: message }]);
    setIsLoading(true);

    try {
      const response = await chatAPI.sendMessage(message);

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.response,
          agent: response.agent,
          intent: response.intent,
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
      await chatAPI.resetConversation();
      setMessages([]);
      setStats(null);
      setAnalytics(null);
      setShowAnalytics(false);
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
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <ChatInterface
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          isInitialized={isInitialized}
          onReset={handleReset}
        />
      </div>

      {/* Analytics Panel */}
      {isInitialized && showAnalytics && (
        <div className="w-80 bg-white border-l-2 border-gray-200 shadow-xl">
          <AnalyticsDashboard stats={stats} analytics={analytics} />
        </div>
      )}

      {/* End Conversation & View Analytics Button */}
      {isInitialized && canShowAnalytics && !showAnalytics && (
        <button
          onClick={handleShowAnalytics}
          className="fixed bottom-8 right-8 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-4 rounded-2xl shadow-2xl transition-all duration-300 flex items-center gap-3 z-10 font-bold text-lg hover:scale-105 transform"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          View Analytics
        </button>
      )}

      {/* Hide Analytics Button */}
      {isInitialized && showAnalytics && (
        <button
          onClick={() => setShowAnalytics(false)}
          className="fixed bottom-8 right-8 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white px-8 py-4 rounded-2xl shadow-2xl transition-all duration-300 flex items-center gap-3 z-10 font-bold text-lg hover:scale-105 transform"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
          Hide Analytics
        </button>
      )}
    </div>
  );
}

export default App;
