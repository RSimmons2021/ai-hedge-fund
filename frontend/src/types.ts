// Analyst information
export interface Analyst {
  id: string;
  name: string;
  displayName: string;
  description: string;
  investmentStrategy: string;
  avatarSrc: string;
  color: string;
  strengths: string[];
  weaknesses: string[];
  keyMetrics: string[];
}

// Portfolio position
export interface Position {
  ticker: string;
  long: number;
  short: number;
  longCostBasis: number;
  shortCostBasis: number;
}

// Performance history entry
export interface PerformanceEntry {
  date: string;
  value: number;
}

// Portfolio summary
export interface PortfolioSummary {
  initialValue: number;
  currentValue: number;
  totalGainLoss: number;
  percentageGainLoss: number;
  positions: Position[];
  performanceHistory: PerformanceEntry[];
}

// Trade recommendation
export interface TradeRecommendation {
  ticker: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  analyst: string;
}

// Stock data
export interface StockData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Simulation parameters
export interface SimulationParameters {
  tickers: string[];
  startDate: string;
  endDate: string;
  initialCash: number;
  marginRequirement: number;
  showReasoning: boolean;
  sequential: boolean;
  selectedAnalysts: string[];
}

// Simulation status
export interface SimulationStatus {
  simulationId: string;
  status: 'starting' | 'running' | 'completed' | 'failed';
  parameters: SimulationParameters;
  results?: any;
  error?: string;
  output?: string;
}
