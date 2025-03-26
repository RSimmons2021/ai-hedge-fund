import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import {
  Box, Typography, Paper, Grid, Button, Chip, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TableFooter,
  styled, Divider, Card, CardContent, Avatar, Tooltip
} from '@mui/material';
import { motion } from 'framer-motion';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from 'recharts';
import { getPortfolioSummary, getTradeRecommendations, getSimulationStatus, stopRealtimeTrading } from '../api/api';
import { PortfolioSummary, TradeRecommendation, SimulationStatus } from '../types';

const DashboardContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  maxWidth: 1200,
  margin: '0 auto',
}));

const GlowPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  backgroundColor: 'rgba(18, 18, 30, 0.8)',
  backdropFilter: 'blur(10px)',
  borderRadius: theme.shape.borderRadius,
  border: '1px solid rgba(255, 255, 255, 0.1)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '2px',
    background: 'linear-gradient(90deg, transparent, #00f5ff, transparent)',
    animation: 'glowScan 4s infinite',
  },
  '@keyframes glowScan': {
    '0%': { transform: 'translateX(-100%)' },
    '100%': { transform: 'translateX(100%)' },
  },
}));

const SectionTitle = styled(Typography)(({ theme }) => ({
  fontFamily: "'Press Start 2P', monospace",
  fontSize: '1rem',
  marginBottom: theme.spacing(2),
  color: theme.palette.primary.main,
  textShadow: `0 0 10px ${theme.palette.primary.main}`,
}));

const ActionChip = styled(Chip)<{ actiontype: 'BUY' | 'SELL' | 'HOLD' }>(({ theme, actiontype }) => ({
  fontWeight: 'bold',
  backgroundColor: 
    actiontype === 'BUY' ? 'rgba(0, 200, 83, 0.2)' : 
    actiontype === 'SELL' ? 'rgba(255, 82, 82, 0.2)' : 
    'rgba(255, 193, 7, 0.2)',
  color: 
    actiontype === 'BUY' ? '#00c853' : 
    actiontype === 'SELL' ? '#ff5252' : 
    '#ffc107',
  border: `1px solid ${actiontype === 'BUY' ? '#00c853' : actiontype === 'SELL' ? '#ff5252' : '#ffc107'}`,
}));

const AnalystCard = styled(Card)(({ theme }) => ({
  backgroundColor: 'rgba(18, 18, 30, 0.5)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: theme.shape.borderRadius,
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: '0 10px 20px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(0, 245, 255, 0.3)',
  },
}));

const AnalystAvatar = styled(Avatar)(({ theme }) => ({
  width: 60,
  height: 60,
  border: '2px solid rgba(0, 245, 255, 0.5)',
  boxShadow: '0 0 15px rgba(0, 245, 255, 0.3)',
}));

const StopButton = styled(Button)(({ theme }) => ({
  background: 'linear-gradient(45deg, #ff5252 30%, #ff1744 90%)',
  border: 0,
  borderRadius: 3,
  boxShadow: '0 3px 5px 2px rgba(255, 82, 82, .3)',
  color: 'white',
  height: 48,
  padding: '0 30px',
  fontFamily: "'Press Start 2P', monospace",
  fontSize: '0.8rem',
}));

const RefreshButton = styled(Button)(({ theme }) => ({
  background: 'linear-gradient(45deg, #00f5ff 30%, #b967ff 90%)',
  border: 0,
  borderRadius: 3,
  boxShadow: '0 3px 5px 2px rgba(0, 245, 255, .3)',
  color: 'white',
  height: 48,
  padding: '0 30px',
  fontFamily: "'Press Start 2P', monospace",
  fontSize: '0.8rem',
  marginRight: theme.spacing(2),
}));

const Dashboard: React.FC = () => {
  const location = useLocation();
  const { simulationId, tradingMode, selectedAnalysts } = location.state || {};
  
  // Log the received data for debugging
  console.log('Dashboard received state:', { simulationId, tradingMode, selectedAnalysts });
  
  const [portfolioSummary, setPortfolioSummary] = useState<PortfolioSummary | null>(null);
  const [tradeRecommendations, setTradeRecommendations] = useState<TradeRecommendation[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [simulationStatus, setSimulationStatus] = useState<SimulationStatus | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);
  const [isRealtime, setIsRealtime] = useState<boolean>(tradingMode === 'realtime');
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    if (!simulationId) {
      setError('No simulation ID provided. Please start a new simulation.');
      setLoading(false);
      return;
    }
    
    try {
      // Get simulation status first
      if (tradingMode === 'backtest') {
        const status = await getSimulationStatus(simulationId);
        setSimulationStatus(status);
        console.log('Simulation status:', status);
        
        // If simulation is still running, don't fetch other data yet
        if (status.status === 'running') {
          return;
        }
        
        // If simulation failed, show error
        if (status.status === 'failed') {
          setError(`Simulation failed: ${status.error || 'Unknown error'}`);
          setLoading(false);
          return;
        }
      }
      
      // Fetch portfolio summary and trade recommendations
      try {
        const [summaryData, recommendationsData] = await Promise.all([
          getPortfolioSummary(simulationId),
          getTradeRecommendations(simulationId)
        ]);
        
        console.log('Portfolio summary:', summaryData);
        console.log('Trade recommendations:', recommendationsData);
        
        setPortfolioSummary(summaryData);
        setTradeRecommendations(recommendationsData);
      } catch (dataError) {
        console.error('Error fetching dashboard data:', dataError);
        setError('Error loading portfolio data. The simulation may still be processing.');
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setError('Error connecting to the server. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleStopTrading = async () => {
    if (!simulationId || tradingMode !== 'realtime') return;
    
    try {
      await stopRealtimeTrading(simulationId);
      clearInterval(refreshInterval as NodeJS.Timeout);
      setRefreshInterval(null);
      setIsRealtime(false);
    } catch (error) {
      console.error('Error stopping real-time trading:', error);
    }
  };

  const handleRefresh = () => {
    fetchDashboardData();
  };

  useEffect(() => {
    fetchDashboardData();
    
    // Set up auto-refresh for real-time trading
    if (tradingMode === 'realtime') {
      const interval = setInterval(fetchDashboardData, 10000); // Refresh every 10 seconds
      setRefreshInterval(interval);
      
      return () => clearInterval(interval);
    }
    
    // For backtest mode, poll status until complete
    if (tradingMode === 'backtest') {
      const checkStatus = async () => {
        if (!simulationId) return;
        
        try {
          const status = await getSimulationStatus(simulationId);
          setSimulationStatus(status);
          
          if (status.status === 'completed' || status.status === 'failed') {
            clearInterval(statusInterval);
            fetchDashboardData();
          }
        } catch (error) {
          console.error('Error checking simulation status:', error);
        }
      };
      
      const statusInterval = setInterval(checkStatus, 2000); // Check every 2 seconds
      
      return () => clearInterval(statusInterval);
    }
  }, [simulationId, tradingMode]);

  if (loading || !portfolioSummary) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh', flexDirection: 'column' }}>
        <CircularProgress size={60} sx={{ color: '#00f5ff' }} />
        <Typography sx={{ mt: 2, fontFamily: "'Press Start 2P', monospace", color: '#00f5ff' }}>
          {simulationStatus?.status === 'running' 
            ? `SIMULATION IN PROGRESS: ${Math.round(simulationStatus.progress * 100)}%` 
            : 'LOADING DASHBOARD...'}
        </Typography>
        {simulationStatus?.message && (
          <Typography sx={{ mt: 2, color: '#b0b0b0', maxWidth: '600px', textAlign: 'center' }}>
            {simulationStatus.message}
          </Typography>
        )}
        {error && (
          <Typography sx={{ mt: 2, color: '#ff5252', maxWidth: '600px', textAlign: 'center' }}>
            {error}
          </Typography>
        )}
      </Box>
    );
  }

  return (
    <DashboardContainer>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography 
              variant="h3" 
              component="h1" 
              sx={{ 
                fontFamily: "'Press Start 2P', monospace", 
                color: '#ff00ff', 
                textShadow: '0 0 10px #ff00ff, 0 0 20px #ff00ff',
                mb: 1
              }}
            >
              AI HEDGE FUND
            </Typography>
            <Typography variant="body1" sx={{ color: '#b0b0b0' }}>
              {isRealtime 
                ? 'Real-time trading mode - Live market analysis and recommendations' 
                : 'Backtest results - Historical performance analysis'}
            </Typography>
          </Box>
          
          <Box>
            {isRealtime && (
              <StopButton onClick={handleStopTrading}>
                STOP TRADING
              </StopButton>
            )}
            {!isRealtime && (
              <RefreshButton onClick={handleRefresh}>
                REFRESH DATA
              </RefreshButton>
            )}
          </Box>
        </Box>

        {/* Analyst Reasoning Section - NEW */}
        {tradeRecommendations.length > 0 && tradeRecommendations.some(rec => rec.reasoning) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <GlowPaper sx={{ mb: 3, p: 3 }}>
              <Typography 
                variant="h6" 
                sx={{ 
                  fontFamily: "'Press Start 2P', monospace", 
                  color: '#00f5ff',
                  mb: 2,
                  fontSize: '0.9rem'
                }}
              >
                LATEST ANALYST REASONING
              </Typography>
              {tradeRecommendations
                .filter(rec => rec.reasoning && rec.action !== 'HOLD')
                .slice(0, 1)
                .map((rec, index) => (
                  <Box key={index} sx={{ backgroundColor: 'rgba(0,0,0,0.3)', p: 3, borderRadius: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <ActionChip 
                        label={rec.action} 
                        actiontype={rec.action as 'BUY' | 'SELL' | 'HOLD'}
                        size="medium"
                      />
                      <Typography 
                        variant="h5" 
                        sx={{ 
                          ml: 2, 
                          fontWeight: 'bold',
                          color: '#fff'
                        }}
                      >
                        {rec.ticker} - {rec.shares || ''} shares
                      </Typography>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          ml: 'auto',
                          color: '#b0b0b0',
                          fontStyle: 'italic'
                        }}
                      >
                        Analysis by: {rec.analyst}
                      </Typography>
                    </Box>
                    
                    <Typography 
                      variant="body1" 
                      sx={{ 
                        backgroundColor: 'rgba(0, 245, 255, 0.05)', 
                        p: 2, 
                        borderRadius: 1,
                        border: '1px solid rgba(0, 245, 255, 0.2)',
                        fontFamily: 'monospace',
                        whiteSpace: 'pre-wrap',
                        color: '#e0e0e0'
                      }}
                    >
                      {rec.reasoning}
                    </Typography>
                  </Box>
              ))}
            </GlowPaper>
          </motion.div>
        )}
        
        <Grid container spacing={3}>
          {/* Portfolio Summary */}
          <Grid item xs={12}>
            <GlowPaper>
              <SectionTitle>PORTFOLIO PERFORMANCE</SectionTitle>
              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={portfolioSummary?.performanceHistory || []}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis 
                          dataKey="date" 
                          stroke="#b0b0b0"
                          tickFormatter={(dateStr) => {
                            // Enhanced date handling
                            try {
                              if (!dateStr) return 'Date';
                              
                              // Handle numeric indexes as dates
                              if (!isNaN(Number(dateStr))) {
                                // If it's just a number, use it as the day
                                const today = new Date();
                                return `Day ${dateStr}`;
                              }
                              
                              const date = new Date(dateStr);
                              // Check if date is valid
                              if (isNaN(date.getTime())) {
                                return dateStr.substring(0, 10); // Just show the first 10 chars
                              }
                              return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                            } catch (e) {
                              console.error('Error formatting date:', dateStr, e);
                              return String(dateStr).substring(0, 10);
                            }
                          }}
                        />
                        <YAxis 
                          stroke="#b0b0b0"
                          tickFormatter={(value) => `$${value.toLocaleString()}`}
                          domain={['dataMin - 1000', 'dataMax + 1000']}
                        />
                        <RechartsTooltip
                          formatter={(value) => [`$${Number(value).toLocaleString()}`, 'Portfolio Value']}
                          labelFormatter={(label) => {
                            try {
                              if (!isNaN(Number(label))) {
                                return `Day ${label}`;
                              }
                              
                              const date = new Date(label);
                              if (isNaN(date.getTime())) {
                                return String(label);
                              }
                              return date.toLocaleDateString('en-US', { 
                                year: 'numeric',
                                month: 'long', 
                                day: 'numeric' 
                              });
                            } catch (e) {
                              return String(label);
                            }
                          }}
                          contentStyle={{ 
                            backgroundColor: '#0a0a2a', 
                            border: '1px solid #33ccff',
                            color: '#fff' 
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey="value" 
                          stroke="#33ccff" 
                          strokeWidth={2}
                          dot={{
                            fill: '#33ccff',
                            r: 4,
                            strokeWidth: 0
                          }}
                          activeDot={{
                            fill: '#ffffff',
                            stroke: '#33ccff',
                            r: 6,
                            strokeWidth: 2
                          }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="h6" sx={{ color: '#b0b0b0', mb: 1 }}>Current Value</Typography>
                      <Typography variant="h3" sx={{ 
                        color: '#00f5ff', 
                        fontWeight: 'bold',
                        textShadow: '0 0 10px rgba(0, 245, 255, 0.5)'
                      }}>
                        ${portfolioSummary.currentValue.toLocaleString()}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="h6" sx={{ color: '#b0b0b0', mb: 1 }}>Total Gain/Loss</Typography>
                      <Typography variant="h4" sx={{ 
                        color: portfolioSummary.totalGainLoss >= 0 ? '#00c853' : '#ff5252', 
                        fontWeight: 'bold',
                        textShadow: `0 0 10px ${portfolioSummary.totalGainLoss >= 0 ? 'rgba(0, 200, 83, 0.5)' : 'rgba(255, 82, 82, 0.5)'}`
                      }}>
                        {portfolioSummary.totalGainLoss >= 0 ? '+' : ''}
                        ${portfolioSummary.totalGainLoss.toLocaleString()} 
                        ({portfolioSummary.percentageGainLoss >= 0 ? '+' : ''}
                        {portfolioSummary.percentageGainLoss.toFixed(2)}%)
                      </Typography>
                    </Box>
                    
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="h6" sx={{ color: '#b0b0b0', mb: 1 }}>Initial Investment</Typography>
                      <Typography variant="h5" sx={{ color: '#b0b0b0' }}>
                        ${portfolioSummary.initialValue.toLocaleString()}
                      </Typography>
                    </Box>
                  </Box>
                </Grid>
              </Grid>
            </GlowPaper>
          </Grid>

          {/* Current Positions */}
          <Grid item xs={12} md={6}>
            <GlowPaper>
              <SectionTitle>CURRENT POSITIONS</SectionTitle>
              <TableContainer component={Box}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Ticker</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Long</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Short</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Avg. Cost</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolioSummary.positions.map((position) => (
                      <TableRow key={position.ticker}>
                        <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                          {position.ticker}
                        </TableCell>
                        <TableCell align="right" sx={{ color: position.long > 0 ? '#00c853' : '#b0b0b0' }}>
                          {position.long > 0 ? position.long : '-'}
                        </TableCell>
                        <TableCell align="right" sx={{ color: position.short > 0 ? '#ff5252' : '#b0b0b0' }}>
                          {position.short > 0 ? position.short : '-'}
                        </TableCell>
                        <TableCell align="right">
                          ${position.long > 0 
                            ? position.longCostBasis.toFixed(2) 
                            : position.short > 0 
                              ? position.shortCostBasis.toFixed(2) 
                              : '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                    {portfolioSummary.positions.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={4} align="center" sx={{ color: '#b0b0b0' }}>
                          No positions currently held
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </GlowPaper>
          </Grid>

          {/* Trade Recommendations */}
          <Grid item xs={12} md={6}>
            <GlowPaper>
              <SectionTitle>TRADE RECOMMENDATIONS</SectionTitle>
              <TableContainer component={Box}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Ticker</TableCell>
                      <TableCell sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Action</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Quantity</TableCell>
                      <TableCell sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Analyst</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {tradeRecommendations.map((recommendation, index) => (
                      <TableRow 
                        key={`${recommendation.ticker}-${index}`}
                        hover
                        sx={{ cursor: 'pointer' }}
                        onClick={() => {
                          // Toggle showing reasoning in expanded section
                          const updated = [...tradeRecommendations];
                          updated[index] = {...updated[index], showReasoning: !updated[index].showReasoning};
                          setTradeRecommendations(updated);
                        }}
                      >
                        <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                          {recommendation.ticker}
                        </TableCell>
                        <TableCell>
                          <ActionChip 
                            label={recommendation.action} 
                            actiontype={recommendation.action as 'BUY' | 'SELL' | 'HOLD'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right">
                          {recommendation.shares || 0}
                        </TableCell>
                        <TableCell>{recommendation.analyst}</TableCell>
                      </TableRow>
                    ))}
                    {tradeRecommendations.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={4} align="center" sx={{ color: '#b0b0b0' }}>
                          No trade recommendations available
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
                
                {/* Reasoning Display */}
                {tradeRecommendations.filter(rec => rec.showReasoning).map((rec, index) => (
                  <Box 
                    key={`reasoning-${index}`} 
                    sx={{ 
                      mt: 2, 
                      p: 2, 
                      backgroundColor: 'rgba(0, 0, 0, 0.3)', 
                      borderRadius: 1,
                      border: '1px solid rgba(0, 245, 255, 0.3)'
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ color: '#00f5ff', fontWeight: 'bold', mb: 1 }}>
                      Reasoning for {rec.ticker} {rec.action}:
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#ffffff', whiteSpace: 'pre-wrap' }}>
                      {rec.reasoning || "No detailed reasoning provided."}
                    </Typography>
                  </Box>
                ))}
              </TableContainer>
            </GlowPaper>
          </Grid>

          {/* Profit Summary */}
          <Grid item xs={12}>
            <GlowPaper>
              <SectionTitle>PROFIT SUMMARY</SectionTitle>
              <TableContainer component={Box}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Ticker</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Shares</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Price</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Value</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Profit/Loss</TableCell>
                      <TableCell align="right" sx={{ color: '#00f5ff', fontWeight: 'bold' }}>Return %</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolioSummary.positions.map((position) => {
                      // Calculate total shares (long - short)
                      const shares = position.long - position.short;
                      // Using average cost as a simple estimate
                      const price = position.long > 0 ? position.longCostBasis : position.shortCostBasis;
                      // Estimate value based on shares and price
                      const value = Math.abs(shares) * price;
                      // For demo purposes - in real app you'd have actual P&L data
                      const profitLoss = position.long > 0 ? value * 0.05 * (Math.random() > 0.5 ? 1 : -1) : 0;
                      const returnPct = profitLoss !== 0 ? (profitLoss / value) * 100 : 0;
                      
                      return (
                        <TableRow key={`profit-${position.ticker}`}>
                          <TableCell component="th" scope="row" sx={{ fontWeight: 'bold' }}>
                            {position.ticker}
                          </TableCell>
                          <TableCell align="right">{shares !== 0 ? shares : '-'}</TableCell>
                          <TableCell align="right">${price.toFixed(2)}</TableCell>
                          <TableCell align="right">${value.toFixed(2)}</TableCell>
                          <TableCell align="right" sx={{ color: profitLoss >= 0 ? '#00c853' : '#ff5252' }}>
                            ${Math.abs(profitLoss).toFixed(2)}
                          </TableCell>
                          <TableCell align="right" sx={{ color: returnPct >= 0 ? '#00c853' : '#ff5252' }}>
                            {returnPct.toFixed(2)}%
                          </TableCell>
                        </TableRow>
                      );
                    })}
                    {portfolioSummary.positions.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={6} align="center" sx={{ color: '#b0b0b0' }}>
                          No positions to display profit summary
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                  <TableFooter>
                    <TableRow sx={{ backgroundColor: 'rgba(0, 0, 0, 0.3)' }}>
                      <TableCell colSpan={3} sx={{ fontWeight: 'bold' }}>Portfolio Total</TableCell>
                      <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                        ${portfolioSummary.currentValue.toLocaleString()}
                      </TableCell>
                      <TableCell align="right" sx={{ 
                        fontWeight: 'bold',
                        color: portfolioSummary.totalGainLoss >= 0 ? '#00c853' : '#ff5252' 
                      }}>
                        ${Math.abs(portfolioSummary.totalGainLoss).toLocaleString()}
                      </TableCell>
                      <TableCell align="right" sx={{ 
                        fontWeight: 'bold',
                        color: portfolioSummary.percentageGainLoss >= 0 ? '#00c853' : '#ff5252' 
                      }}>
                        {portfolioSummary.percentageGainLoss.toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  </TableFooter>
                </Table>
              </TableContainer>
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                <Typography variant="body2" sx={{ color: '#b0b0b0', fontStyle: 'italic' }}>
                  Cash Balance: ${portfolioSummary.initialValue - portfolioSummary.positions.reduce((sum, pos) => {
                    const value = pos.long * pos.longCostBasis + pos.short * pos.shortCostBasis;
                    return sum + value;
                  }, 0).toFixed(2)}
                </Typography>
              </Box>
            </GlowPaper>
          </Grid>

          {/* Analyst Team */}
          <Grid item xs={12}>
            <GlowPaper>
              <SectionTitle>ANALYST TEAM</SectionTitle>
              <Grid container spacing={2}>
                {selectedAnalysts.map((analyst: any) => (
                  <Grid item xs={12} sm={6} md={4} lg={3} key={analyst.id}>
                    <AnalystCard>
                      <CardContent sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
                        <AnalystAvatar 
                          src={analyst.avatarSrc} 
                          alt={analyst.displayName}
                          sx={{ bgcolor: analyst.color }}
                        />
                        <Typography variant="h6" sx={{ mt: 2, color: '#fff', fontWeight: 'bold' }}>
                          {analyst.displayName}
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#b0b0b0', fontStyle: 'italic', mb: 1 }}>
                          {analyst.description}
                        </Typography>
                        <Divider sx={{ width: '100%', my: 1, borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                        <Typography variant="body2" sx={{ color: '#00f5ff', fontWeight: 'bold', mb: 0.5 }}>
                          Key Metrics:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 0.5 }}>
                          {analyst.keyMetrics.slice(0, 3).map((metric: string, index: number) => (
                            <Chip 
                              key={index} 
                              label={metric} 
                              size="small" 
                              sx={{ 
                                backgroundColor: 'rgba(0, 245, 255, 0.1)',
                                color: '#00f5ff',
                                border: '1px solid rgba(0, 245, 255, 0.3)',
                                fontSize: '0.7rem'
                              }} 
                            />
                          ))}
                        </Box>
                      </CardContent>
                    </AnalystCard>
                  </Grid>
                ))}
              </Grid>
            </GlowPaper>
          </Grid>
        </Grid>
      </motion.div>
    </DashboardContainer>
  );
};

export default Dashboard;
