import axios from 'axios';
import { Analyst, PortfolioSummary, TradeRecommendation, StockData, SimulationParameters, SimulationStatus } from '../types';

// Configure API endpoint based on environment
const getApiBaseUrl = () => {
  // For local development
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return '/api';
  }
  // For production - replace with your actual backend URL when deployed
  const apiUrl = import.meta.env.VITE_API_URL as string;
  return apiUrl || 'https://your-deployed-backend-url.com/api';
};

const API_BASE_URL = getApiBaseUrl();

const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Get all available analysts
export const getAnalysts = async (): Promise<Analyst[]> => {
  try {
    const response = await api.get('/analysts');
    return response.data;
  } catch (error) {
    console.error('Error fetching analysts:', error);
    return [];
  }
};

// Start a new hedge fund simulation
export const startHedgeFund = async (params: SimulationParameters): Promise<{ simulationId: string }> => {
  try {
    const response = await api.post('/simulate', params);
    return response.data;
  } catch (error) {
    console.error('Error starting hedge fund:', error);
    throw error;
  }
};

// Get the status of a simulation
export const getSimulationStatus = async (simulationId: string): Promise<SimulationStatus> => {
  try {
    const response = await api.get(`/simulation/${simulationId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching simulation status for ${simulationId}:`, error);
    throw error;
  }
};

// Get portfolio summary for a specific simulation
export const getPortfolioSummary = async (simulationId: string): Promise<PortfolioSummary> => {
  try {
    const response = await api.get(`/portfolio-summary/${simulationId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching portfolio summary:', error);
    throw error;
  }
};

// Get trade recommendations for a specific simulation
export const getTradeRecommendations = async (simulationId: string): Promise<TradeRecommendation[]> => {
  try {
    const response = await api.get(`/trade-recommendations/${simulationId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching trade recommendations:', error);
    return [];
  }
};

// Get stock data for a specific ticker
export const getStockData = async (ticker: string): Promise<StockData[]> => {
  try {
    const response = await api.get(`/stock-data/${ticker}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching stock data for ${ticker}:`, error);
    return [];
  }
};

// Start a real-time trading session
export const startRealtimeTrading = async (params: SimulationParameters): Promise<{ sessionId: string }> => {
  try {
    const response = await api.post('/start-realtime', params);
    return response.data;
  } catch (error) {
    console.error('Error starting real-time trading:', error);
    throw error;
  }
};

// Stop a real-time trading session
export const stopRealtimeTrading = async (sessionId: string): Promise<{ status: string }> => {
  try {
    const response = await api.post(`/stop-realtime/${sessionId}`);
    return response.data;
  } catch (error) {
    console.error(`Error stopping real-time trading session ${sessionId}:`, error);
    throw error;
  }
};

export default api;
