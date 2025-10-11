/**
 * App - Main application component (simplified without analytics)
 */

import React, { useState } from 'react';
import Sidebar from './components/sidebar/Sidebar';
import ChatInterface from './components/chat/ChatInterface';
import { chatService } from './services/chatService';

function App() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([]);

  const handleInitialize = async (modelType, apiKey, model) => {
    setIsLoading(true);
    try {
      await chatService.initialize(modelType, apiKey, model);
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
      const response = await chatService.sendMessage(message);

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.response,
          agent: response.agent,
          intent: response.intent,
          isVoiceOnly: isVoiceInput,
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
      await chatService.resetConversation();
      setMessages([]);
    } catch (error) {
      console.error('Reset error:', error);
    }
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
    </div>
  );
}

export default App;
