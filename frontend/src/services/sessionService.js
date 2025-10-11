/**
 * Session service - handles health checks and session status
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sessionService = {
  /**
   * Health check endpoint
   */
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  /**
   * Root endpoint
   */
  getStatus: async () => {
    const response = await api.get('/');
    return response.data;
  },
};

export default sessionService;
