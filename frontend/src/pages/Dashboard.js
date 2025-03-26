import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Typography, Grid, Paper, Button, CircularProgress,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Divider, Chip, Accordion, AccordionSummary, AccordionDetails, Alert
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { styled } from '@mui/material/styles';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import api from '../utils/api';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Styled components
const DashboardContainer = styled(Box)(({ theme }) => ({
  minHeight: 'calc(100vh - 64px)',
  padding: theme.spacing(3),
  backgroundImage: `linear-gradient(rgba(18, 18, 18, 0.8), rgba(18, 18, 18, 0.9)),
    linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px),
    linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px)`,
  backgroundSize: '100% 100%, 20px 20px, 20px 20px',
  backgroundPosition: '0 0, 0 0, 0 0',
}));

const DashboardPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  backgroundColor: 'rgba(30, 30, 30, 0.8)',
  border: '1px solid rgba(0, 255, 255, 0.3)',
  borderRadius: 0,
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
}));

const ChartTitle = styled(Typography)(({ theme }) => ({
  fontFamily: '"Orbitron", sans-serif',
  fontWeight: 600,
  marginBottom: theme.spacing(2),
  color: '#00ffff',
  textShadow: '0 0 5px rgba(0, 255, 255, 0.3)',
}));

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  borderBottom: '1px solid rgba(0, 255, 255, 0.2)',
  padding: theme.spacing(1.5),
}));

const StyledTableHeadCell = styled(StyledTableCell)(({ theme }) => ({
  backgroundColor: 'rgba(0, 255, 255, 0.1)',
  fontFamily: '"Orbitron", sans-serif',
  fontWeight: 600,
  color: '#00ffff',
}));

const Dashboard = () => {
  const [simulationParams, setSimulationParams] = useState(null);
  const [simulationResult, setSimulationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysts, setAnalysts] = useState({});
  const navigate = useNavigate();

  // Chart styles and options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#ffffff',
          font: {
            family: 'Roboto',
          },
        },
      },
      tooltip: {
        backgroundColor: 'rgba(30, 30, 30, 0.8)',
        titleFont: {
          family: 'Orbitron',
        },
        bodyFont: {
          family: 'Roboto',
        },
        borderColor: 'rgba(0, 255, 255, 0.3)',
        borderWidth: 1,
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: '#ffffff',
        },
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        ticks: {
          color: '#ffffff',
        },
      },
    },
  };

  // Load simulation parameters and analysts data
  useEffect(() => {
    const fetchData = async () => {
      try {
        // Get simulation parameters from session storage
        const storedParams = sessionStorage.getItem('simulationParams');
        if (!storedParams) {
          // If no parameters, redirect to setup page
          navigate('/setup');
          return;
        }
        
        const params = JSON.parse(storedParams);
        setSimulationParams(params);

        // Fetch analysts data for displaying names
        const analystsData = await api.getAnalysts();
        setAnalysts(analystsData);

      } catch (err) {
        console.error('Error loading data:', err);
        setError('Failed to load simulation parameters. Please try again.');
      }
    };

    fetchData();
  }, [navigate]);

  // Run simulation when parameters are loaded
  useEffect(() => {
    const runSimulation = async () => {
      if (!simulationParams) return;

      try {
        setLoading(true);
        setError(null);

        // Call API to run simulation
        const result = await api.runSimulation(simulationParams);
        setSimulationResult(result);
        setLoading(false);
      } catch (err) {
        console.error('Simulation error:', err);
        setError('Failed to run simulation. Please check your parameters and try again.');
        setLoading(false);
      }
    };

    runSimulation();
  }, [simulationParams]);

  // Format currency values
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  // Format percentage values
  const formatPercentage = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value / 100);
  };

  // Generate portfolio allocation chart data
  const generatePortfolioData = () => {
    if (!simulationResult || !simulationResult.result || !simulationResult.result.decisions) return null;

    const decisions = simulationResult.result.decisions;
    const portfolio = decisions.portfolio || {};
    const positions = portfolio.positions || {};
    
    // Calculate total portfolio value including cash
    const cash = portfolio.cash || 0;
    let totalValue = cash;
    
    const positionValues = {};
    Object.entries(positions).forEach(([ticker, position]) => {
      const longValue = position.current_price * position.long || 0;
      const shortValue = position.current_price * position.short || 0;
      positionValues[ticker] = longValue - shortValue;
      totalValue += longValue - shortValue;
    });

    // Prepare data for doughnut chart
    const labels = ['Cash', ...Object.keys(positionValues)];
    const data = [cash, ...Object.values(positionValues)];
    const backgroundColors = [
      'rgba(0, 255, 255, 0.7)',
      'rgba(255, 99, 132, 0.7)',
      'rgba(54, 162, 235, 0.7)',
      'rgba(255, 206, 86, 0.7)',
      'rgba(75, 192, 192, 0.7)',
      'rgba(153, 102, 255, 0.7)',
      'rgba(255, 159, 64, 0.7)',
      'rgba(255, 99, 255, 0.7)',
      'rgba(199, 255, 132, 0.7)',
    ];

    return {
      labels,
      datasets: [
        {
          data,
          backgroundColor: backgroundColors,
          borderColor: backgroundColors.map(color => color.replace('0.7', '1')),
          borderWidth: 1,
        },
      ],
    };
  };

  // Generate analyst signals chart data
  const generateAnalystSignalsData = () => {
    if (!simulationResult || !simulationResult.result || !simulationResult.result.analyst_signals) return null;

    const analystSignals = simulationResult.result.analyst_signals;
    const tickers = simulationParams?.tickers || [];

    // Create datasets for each ticker
    const datasets = tickers.map((ticker, index) => {
      const signals = [];
      const confidences = [];
      const analysts = [];

      // Collect signals for this ticker from each analyst
      Object.entries(analystSignals).forEach(([analystId, tickerSignals]) => {
        if (tickerSignals[ticker]) {
          const signal = tickerSignals[ticker].signal;
          const confidence = tickerSignals[ticker].confidence;
          
          // Convert signal to numeric value for chart
          let signalValue = 0;
          if (signal === 'bullish') signalValue = 1;
          else if (signal === 'bearish') signalValue = -1;
          
          signals.push(signalValue * confidence);
          confidences.push(confidence);
          analysts.push(analystId);
        }
      });

      // Colors for each ticker
      const colors = [
        'rgba(0, 255, 255, 0.7)',
        'rgba(255, 99, 132, 0.7)',
        'rgba(54, 162, 235, 0.7)',
        'rgba(255, 206, 86, 0.7)',
        'rgba(75, 192, 192, 0.7)',
      ];

      return {
        label: ticker,
        data: signals,
        backgroundColor: colors[index % colors.length],
        borderColor: colors[index % colors.length].replace('0.7', '1'),
        borderWidth: 1,
      };
    });

    // Get analyst names for labels
    const analystLabels = Object.keys(analystSignals).map(analystId => {
      return analysts[analystId]?.name || analystId;
    });

    return {
      labels: analystLabels,
      datasets,
    };
  };

  // Render loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 64px)' }}>
        <Box sx={{ textAlign: 'center' }}>
          <CircularProgress size={60} sx={{ color: '#00ffff', mb: 3 }} />
          <Typography variant="h6" sx={{ fontFamily: '"Orbitron", sans-serif' }}>
            Running Simulation...
          </Typography>
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mt: 1 }}>
            This may take a few minutes depending on the complexity of your analysis
          </Typography>
        </Box>
      </Box>
    );
  }

  // Render error state
  if (error) {
    return (
      <DashboardContainer>
        <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4 }}>
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
          <Button
            variant="contained"
            onClick={() => navigate('/setup')}
            sx={{ 
              fontFamily: '"Orbitron", sans-serif',
              borderRadius: 0
            }}
          >
            Back to Setup
          </Button>
        </Box>
      </DashboardContainer>
    );
  }

  // If no simulation result yet
  if (!simulationResult) {
    return (
      <DashboardContainer>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 100px)' }}>
          <CircularProgress size={40} sx={{ color: '#00ffff', mr: 2 }} />
          <Typography>Preparing dashboard...</Typography>
        </Box>
      </DashboardContainer>
    );
  }

  // Prepare portfolio data for display
  const decisions = simulationResult.result?.decisions || {};
  const portfolio = decisions.portfolio || {};
  const positions = portfolio.positions || {};
  const portfolioChartData = generatePortfolioData();
  const analystSignalsChartData = generateAnalystSignalsData();

  return (
    <DashboardContainer>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography 
          variant="h4" 
          sx={{ 
            fontFamily: '"Orbitron", sans-serif',
            fontWeight: 700,
            color: '#00ffff',
            textShadow: '0 0 10px rgba(0, 255, 255, 0.5)'
          }}
        >
          Trading Dashboard
        </Typography>
        <Button
          variant="outlined"
          onClick={() => navigate('/setup')}
          sx={{ 
            fontFamily: '"Orbitron", sans-serif',
            borderRadius: 0
          }}
        >
          New Simulation
        </Button>
      </Box>

      {/* Simulation Parameters Summary */}
      <Paper 
        sx={{ 
          p: 2, 
          mb: 4, 
          backgroundColor: 'rgba(30, 30, 30, 0.8)',
          border: '1px solid rgba(0, 255, 255, 0.3)',
          borderRadius: 0
        }}
      >
        <Typography variant="h6" sx={{ fontFamily: '"Orbitron", sans-serif', mb: 2, color: '#00ffff' }}>
          Simulation Parameters
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="textSecondary">Tickers</Typography>
            <Typography variant="body1">{simulationParams?.tickers.join(', ')}</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="textSecondary">Date Range</Typography>
            <Typography variant="body1">{simulationParams?.start_date} to {simulationParams?.end_date}</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="textSecondary">Initial Cash</Typography>
            <Typography variant="body1">{formatCurrency(simulationParams?.initial_cash)}</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="textSecondary">Model</Typography>
            <Typography variant="body1">{simulationParams?.model}</Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Portfolio Summary */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <DashboardPaper>
            <ChartTitle variant="h6">Portfolio Performance</ChartTitle>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
              <Box>
                <Typography variant="body2" color="textSecondary">Total Value</Typography>
                <Typography variant="h5" sx={{ fontWeight: 'bold', color: '#00ffff' }}>
                  {formatCurrency(portfolio.total_value || 0)}
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="textSecondary">Cash</Typography>
                <Typography variant="h6">
                  {formatCurrency(portfolio.cash || 0)}
                </Typography>
              </Box>
              <Box>
                <Typography variant="body2" color="textSecondary">Unrealized P/L</Typography>
                <Typography 
                  variant="h6" 
                  sx={{ 
                    color: (portfolio.unrealized_pl || 0) >= 0 ? '#00ff7f' : '#ff5555',
                    fontWeight: 'bold'
                  }}
                >
                  {formatCurrency(portfolio.unrealized_pl || 0)}
                </Typography>
              </Box>
            </Box>
            
            <TableContainer sx={{ mb: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <StyledTableHeadCell>Ticker</StyledTableHeadCell>
                    <StyledTableHeadCell align="right">Price</StyledTableHeadCell>
                    <StyledTableHeadCell align="right">Long</StyledTableHeadCell>
                    <StyledTableHeadCell align="right">Short</StyledTableHeadCell>
                    <StyledTableHeadCell align="right">Value</StyledTableHeadCell>
                    <StyledTableHeadCell align="right">P/L</StyledTableHeadCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(positions).map(([ticker, position]) => {
                    const value = (position.long - position.short) * position.current_price;
                    const pl = position.unrealized_pl || 0;
                    return (
                      <TableRow key={ticker}>
                        <StyledTableCell>{ticker}</StyledTableCell>
                        <StyledTableCell align="right">{formatCurrency(position.current_price)}</StyledTableCell>
                        <StyledTableCell align="right">{position.long}</StyledTableCell>
                        <StyledTableCell align="right">{position.short}</StyledTableCell>
                        <StyledTableCell align="right">{formatCurrency(value)}</StyledTableCell>
                        <StyledTableCell 
                          align="right"
                          sx={{ color: pl >= 0 ? '#00ff7f' : '#ff5555' }}
                        >
                          {formatCurrency(pl)}
                        </StyledTableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          </DashboardPaper>
        </Grid>

        <Grid item xs={12} md={4}>
          <DashboardPaper>
            <ChartTitle variant="h6">Portfolio Allocation</ChartTitle>
            <Box sx={{ height: 300, position: 'relative' }}>
              {portfolioChartData ? (
                <Doughnut data={portfolioChartData} options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    legend: {
                      ...chartOptions.plugins.legend,
                      position: 'bottom'
                    }
                  }
                }} />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                  <Typography variant="body2" color="textSecondary">No allocation data available</Typography>
                </Box>
              )}
            </Box>
          </DashboardPaper>
        </Grid>
      </Grid>

      {/* Analyst Signals */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <DashboardPaper>
            <ChartTitle variant="h6">Analyst Signals</ChartTitle>
            
            {simulationResult.result.analyst_signals && Object.keys(simulationResult.result.analyst_signals).length > 0 ? (
              <>
                <Box sx={{ height: 300, mb: 3 }}>
                  {analystSignalsChartData ? (
                    <Bar data={analystSignalsChartData} options={{
                      ...chartOptions,
                      scales: {
                        ...chartOptions.scales,
                        y: {
                          ...chartOptions.scales.y,
                          min: -1,
                          max: 1,
                          title: {
                            display: true,
                            text: 'Signal Strength',
                            color: '#ffffff'
                          }
                        }
                      }
                    }} />
                  ) : (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                      <Typography variant="body2" color="textSecondary">No signal data available</Typography>
                    </Box>
                  )}
                </Box>

                <Divider sx={{ my: 3, borderColor: 'rgba(0, 255, 255, 0.2)' }} />
                
                {/* Detailed Analyst Signals */}
                <Typography variant="h6" sx={{ mb: 2, fontFamily: '"Orbitron", sans-serif' }}>
                  Detailed Analyst Insights
                </Typography>
                
                {Object.entries(simulationResult.result.analyst_signals).map(([analystId, tickerSignals]) => (
                  <Accordion 
                    key={analystId}
                    sx={{ 
                      backgroundColor: 'rgba(30, 30, 30, 0.5)',
                      mb: 2,
                      '&:before': { display: 'none' },
                      border: '1px solid rgba(0, 255, 255, 0.2)'
                    }}
                  >
                    <AccordionSummary
                      expandIcon={<ExpandMoreIcon sx={{ color: '#00ffff' }} />}
                      sx={{ 
                        borderBottom: '1px solid rgba(0, 255, 255, 0.2)',
                        backgroundColor: 'rgba(0, 255, 255, 0.05)'
                      }}
                    >
                      <Typography sx={{ fontWeight: 'bold', fontFamily: '"Orbitron", sans-serif' }}>
                        {analysts[analystId]?.name || analystId}
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Grid container spacing={2}>
                        {Object.entries(tickerSignals).map(([ticker, signal]) => (
                          <Grid item xs={12} sm={6} md={4} key={ticker}>
                            <Paper 
                              sx={{ 
                                p: 2, 
                                backgroundColor: 'rgba(18, 18, 18, 0.7)',
                                border: '1px solid rgba(0, 255, 255, 0.2)'
                              }}
                            >
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="h6">{ticker}</Typography>
                                <Chip 
                                  label={signal.signal.toUpperCase()}
                                  size="small"
                                  color={
                                    signal.signal === 'bullish' ? 'success' : 
                                    signal.signal === 'bearish' ? 'error' : 'default'
                                  }
                                  sx={{ fontWeight: 'bold' }}
                                />
                              </Box>
                              <Typography variant="body2" color="textSecondary" sx={{ mb: 1 }}>
                                Confidence: {(signal.confidence * 100).toFixed(1)}%
                              </Typography>
                              <Typography variant="body2">
                                {signal.reasoning}
                              </Typography>
                            </Paper>
                          </Grid>
                        ))}
                      </Grid>
                    </AccordionDetails>
                  </Accordion>
                ))}
              </>
            ) : (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="body1" color="textSecondary">
                  No analyst signals available for this simulation
                </Typography>
              </Box>
            )}
          </DashboardPaper>
        </Grid>
      </Grid>

      {/* Trading Decisions */}
      <Grid container spacing={3} sx={{ mt: 3 }}>
        <Grid item xs={12}>
          <DashboardPaper>
            <ChartTitle variant="h6">Trading Decisions</ChartTitle>
            
            {decisions.trades && decisions.trades.length > 0 ? (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <StyledTableHeadCell>Ticker</StyledTableHeadCell>
                      <StyledTableHeadCell>Action</StyledTableHeadCell>
                      <StyledTableHeadCell align="right">Quantity</StyledTableHeadCell>
                      <StyledTableHeadCell align="right">Price</StyledTableHeadCell>
                      <StyledTableHeadCell align="right">Value</StyledTableHeadCell>
                      <StyledTableHeadCell>Reasoning</StyledTableHeadCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {decisions.trades.map((trade, index) => (
                      <TableRow key={index}>
                        <StyledTableCell>{trade.ticker}</StyledTableCell>
                        <StyledTableCell>
                          <Chip 
                            label={trade.action.toUpperCase()}
                            size="small"
                            color={
                              trade.action === 'buy' ? 'success' : 
                              trade.action === 'sell' ? 'error' : 'default'
                            }
                            sx={{ fontWeight: 'bold' }}
                          />
                        </StyledTableCell>
                        <StyledTableCell align="right">{trade.quantity}</StyledTableCell>
                        <StyledTableCell align="right">{formatCurrency(trade.price)}</StyledTableCell>
                        <StyledTableCell align="right">{formatCurrency(trade.quantity * trade.price)}</StyledTableCell>
                        <StyledTableCell>{trade.reasoning}</StyledTableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="body1" color="textSecondary">
                  No trading decisions were made in this simulation
                </Typography>
              </Box>
            )}
          </DashboardPaper>
        </Grid>
      </Grid>
    </DashboardContainer>
  );
};

export default Dashboard;
