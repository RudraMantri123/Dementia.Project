import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

export const chatService = {
  initialize: async (modelType = 'ollama', apiKey = null, modelName = 'llama3.2') => {
    const response = await api.post('/initialize', {
      model_type: modelType,
      api_key: apiKey,
      model_name: modelName,
    });
    return response.data;
  },

  sendMessage: async (message, sessionId = 'default') => {
    const response = await api.post('/chat', { message, session_id: sessionId });
    return response.data;
  },

  resetConversation: async (sessionId = 'default') => {
    const response = await api.post('/reset', null, { params: { session_id: sessionId } });
    return response.data;
  },
};

export default chatService;
