import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

const api = {
  // Get all available analysts
  getAnalysts: async () => {
    try {
      const response = await axios.get(`${API_URL}/analysts`);
      return response.data;
    } catch (error) {
      console.error('Error fetching analysts:', error);
      throw error;
    }
  },

  // Get all available LLM models
  getModels: async () => {
    try {
      const response = await axios.get(`${API_URL}/models`);
      return response.data;
    } catch (error) {
      console.error('Error fetching models:', error);
      throw error;
    }
  },

  // Run a hedge fund simulation
  runSimulation: async (params) => {
    try {
      const response = await axios.post(`${API_URL}/simulate`, params);
      return response.data;
    } catch (error) {
      console.error('Error running simulation:', error);
      throw error;
    }
  }
};

export default api;
