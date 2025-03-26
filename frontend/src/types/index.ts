export interface Analyst {
  id: string;
  name: string;
  displayName: string;
  avatarSrc: string;
  description: string;
  investmentStrategy: string;
  strengths: string[];
  weaknesses: string[];
  keyMetrics: string[];
  color: string;
}

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

export interface SimulationStatus {
  status: 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
}

export interface Position {
  ticker: string;
  long: number;
  short: number;
  longCostBasis: number;
  shortCostBasis: number;
}

export interface PerformancePoint {
  date: string;
  value: number;
}

export interface PortfolioSummary {
  initialValue: number;
  currentValue: number;
  totalGainLoss: number;
  percentageGainLoss: number;
  positions: Position[];
  performanceHistory: PerformancePoint[];
}

export interface TradeRecommendation {
  ticker: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  shares?: number;
  confidence?: number;
  reasoning: string;
  analyst: string;
  showReasoning?: boolean;
}

export interface StockData {
  ticker: string;
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockMetrics {
  ticker: string;
  pe: number;
  marketCap: number;
  dividend: number;
  beta: number;
  dayRange: [number, number];
  yearRange: [number, number];
}
