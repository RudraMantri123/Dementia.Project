import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 second timeout
});

// Request interceptor for logging and token handling
api.interceptors.request.use(
  (config) => {
    // Log requests in development
    if (import.meta.env.DEV) {
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle different error scenarios
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const message = error.response.data?.message || error.response.data?.detail || 'An error occurred';

      console.error(`API Error ${status}:`, message);

      // Handle specific status codes
      switch (status) {
        case 400:
          throw new Error(`Bad Request: ${message}`);
        case 401:
          throw new Error('Unauthorized. Please check your credentials.');
        case 403:
          throw new Error('Access denied.');
        case 404:
          throw new Error(`Not Found: ${message}`);
        case 422:
          throw new Error(`Validation Error: ${message}`);
        case 429:
          throw new Error('Too many requests. Please try again later.');
        case 500:
          throw new Error('Server error. Please try again later.');
        case 503:
          throw new Error('Service unavailable. Please try again later.');
        default:
          throw new Error(message);
      }
    } else if (error.request) {
      // Request made but no response received
      console.error('No response received from server:', error.request);
      throw new Error('Cannot connect to server. Please check your internet connection.');
    } else {
      // Something else happened
      console.error('Request error:', error.message);
      throw new Error('An unexpected error occurred.');
    }
  }
);

// Retry logic for failed requests
const retryRequest = async (requestFn, maxRetries = 3, delay = 1000) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await requestFn();
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      // Don't retry on client errors (4xx)
      if (error.response && error.response.status >= 400 && error.response.status < 500) {
        throw error;
      }

      console.log(`Retry attempt ${i + 1}/${maxRetries} after ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay * (i + 1)));
    }
  }
};

export const chatAPI = {
  // Initialize chatbot with retry logic
  initialize: async (modelType = 'ollama', apiKey = null, modelName = 'llama3:latest') => {
    try {
      const response = await retryRequest(
        () => api.post('/initialize', {
          model_type: modelType,
          api_key: apiKey,
          model_name: modelName,
        }),
        2 // Retry once for initialization
      );
      return response.data;
    } catch (error) {
      console.error('Initialization failed:', error);
      throw new Error(error.message || 'Failed to initialize chatbot');
    }
  },

  // Send chat message with retry
  sendMessage: async (message, sessionId = 'default') => {
    if (!message || message.trim().length === 0) {
      throw new Error('Message cannot be empty');
    }

    if (message.length > 10000) {
      throw new Error('Message is too long. Please keep it under 10,000 characters.');
    }

    try {
      const response = await retryRequest(
        () => api.post('/chat', {
          message: message.trim(),
          session_id: sessionId,
        }),
        3 // Retry up to 3 times for chat messages
      );
      return response.data;
    } catch (error) {
      console.error('Chat message failed:', error);
      throw new Error(error.message || 'Failed to send message');
    }
  },

  // Get conversation stats
  getStats: async (sessionId = 'default') => {
    try {
      const response = await api.get('/stats', {
        params: { session_id: sessionId },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get stats:', error);
      // Return default stats on error
      return {
        total_messages: 0,
        agent_distribution: {},
        intent_distribution: {},
      };
    }
  },

  // Get analytics
  getAnalytics: async (sessionId = 'default') => {
    try {
      const response = await api.post('/analytics', {
        message: '',
        session_id: sessionId,
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get analytics:', error);
      // Return default analytics on error
      return {
        overall_sentiment: 'neutral',
        sentiment_distribution: {},
        needs_support: { level: 'low' },
        insights: [],
      };
    }
  },

  // Reset conversation
  resetConversation: async (sessionId = 'default') => {
    try {
      const response = await api.post('/reset', null, {
        params: { session_id: sessionId },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to reset conversation:', error);
      throw new Error(error.message || 'Failed to reset conversation');
    }
  },

  // Health check with timeout
  healthCheck: async () => {
    try {
      const response = await api.get('/health', { timeout: 5000 });
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      return { status: 'unhealthy', error: error.message };
    }
  },
};

// Export configured axios instance
export default api;
