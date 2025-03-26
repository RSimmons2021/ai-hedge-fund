import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box, Typography, TextField, Button, Paper, Grid, Slider, Switch,
  FormControlLabel, Chip, styled, Divider, CircularProgress
} from '@mui/material';
import { motion } from 'framer-motion';
import { startHedgeFund, startRealtimeTrading } from '../api/api';
import { SimulationParameters } from '../types';

const SetupContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(4),
  maxWidth: 1000,
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

const TickerChip = styled(Chip)(({ theme }) => ({
  margin: theme.spacing(0.5),
  backgroundColor: 'rgba(0, 245, 255, 0.1)',
  color: theme.palette.primary.main,
  border: '1px solid rgba(0, 245, 255, 0.3)',
  '&:hover': {
    backgroundColor: 'rgba(0, 245, 255, 0.2)',
  },
}));

const GlowButton = styled(Button)(({ theme }) => ({
  background: 'linear-gradient(45deg, #00f5ff 30%, #b967ff 90%)',
  border: 0,
  borderRadius: 3,
  boxShadow: '0 3px 5px 2px rgba(0, 245, 255, .3)',
  color: 'white',
  height: 48,
  padding: '0 30px',
  margin: theme.spacing(2, 0),
  fontFamily: "'Press Start 2P', monospace",
  fontSize: '0.8rem',
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    boxShadow: '0 5px 15px 5px rgba(0, 245, 255, .5)',
    transform: 'translateY(-2px)',
  },
}));

const ModeToggleButton = styled(Button)<{ active: boolean }>(({ theme, active }) => ({
  backgroundColor: active ? 'rgba(0, 245, 255, 0.2)' : 'rgba(18, 18, 30, 0.8)',
  color: active ? theme.palette.primary.main : theme.palette.text.secondary,
  border: active ? '1px solid rgba(0, 245, 255, 0.5)' : '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(1, 2),
  margin: theme.spacing(0, 1, 2, 0),
  fontWeight: active ? 'bold' : 'normal',
  '&:hover': {
    backgroundColor: 'rgba(0, 245, 255, 0.1)',
  },
}));

interface PortfolioSetupProps {}

const PortfolioSetup: React.FC<PortfolioSetupProps> = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const selectedAnalysts = location.state?.selectedAnalysts || [];
  
  // Log the selected analysts to verify they're being received correctly
  console.log('Selected analysts in PortfolioSetup:', selectedAnalysts);
  
  const [tickers, setTickers] = useState<string[]>(['AAPL', 'MSFT', 'NVDA']);
  const [newTicker, setNewTicker] = useState<string>('');
  const [startDate, setStartDate] = useState<string>('2023-01-01');
  const [endDate, setEndDate] = useState<string>('2023-12-31');
  const [initialCash, setInitialCash] = useState<number>(100000);
  const [marginRequirement, setMarginRequirement] = useState<number>(50);
  const [showReasoning, setShowReasoning] = useState<boolean>(true);
  const [sequential, setSequential] = useState<boolean>(true);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [tradingMode, setTradingMode] = useState<'backtest' | 'realtime'>('backtest');

  const handleAddTicker = () => {
    if (newTicker && !tickers.includes(newTicker.toUpperCase())) {
      setTickers([...tickers, newTicker.toUpperCase()]);
      setNewTicker('');
    }
  };

  const handleRemoveTicker = (tickerToRemove: string) => {
    setTickers(tickers.filter(ticker => ticker !== tickerToRemove));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      // Prepare parameters for the simulation
      const params: SimulationParameters = {
        tickers,
        startDate,
        endDate,
        initialCash,
        marginRequirement,
        showReasoning,
        sequential,
        selectedAnalysts: selectedAnalysts.map((analyst: any) => analyst.id),
      };
      
      console.log('Submitting simulation with params:', params);
      
      // Start the simulation or real-time trading based on the selected mode
      if (tradingMode === 'backtest') {
        const response = await startHedgeFund(params);
        navigate('/dashboard', { 
          state: { 
            simulationId: response.simulationId, 
            tradingMode: 'backtest',
            selectedAnalysts
          } 
        });
      } else {
        const response = await startRealtimeTrading(params);
        navigate('/dashboard', { 
          state: { 
            simulationId: response.sessionId, 
            tradingMode: 'realtime',
            selectedAnalysts
          } 
        });
      }
    } catch (error) {
      console.error('Error starting simulation:', error);
      alert('Failed to start simulation. Please try again.');
      setIsSubmitting(false);
    }
  };

  return (
    <SetupContainer>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ mb: 4, textAlign: 'center' }}>
          <Typography 
            variant="h3" 
            component="h1" 
            sx={{ 
              fontFamily: "'Press Start 2P', monospace", 
              color: '#ff00ff', 
              textShadow: '0 0 10px #ff00ff, 0 0 20px #ff00ff',
              mb: 2
            }}
          >
            PORTFOLIO SETUP
          </Typography>
          <Typography variant="body1" sx={{ maxWidth: 800, mx: 'auto', color: '#b0b0b0' }}>
            Configure your AI hedge fund parameters and launch your investment strategy.
          </Typography>
        </Box>

        <GlowPaper>
          <Box sx={{ mb: 4 }}>
            <SectionTitle>TRADING MODE</SectionTitle>
            <Box sx={{ display: 'flex', mt: 2 }}>
              <ModeToggleButton 
                active={tradingMode === 'backtest'}
                onClick={() => setTradingMode('backtest')}
              >
                Backtest Mode
              </ModeToggleButton>
              <ModeToggleButton 
                active={tradingMode === 'realtime'}
                onClick={() => setTradingMode('realtime')}
              >
                Real-Time Trading
              </ModeToggleButton>
            </Box>
            <Typography variant="body2" sx={{ mt: 1, color: '#b0b0b0' }}>
              {tradingMode === 'backtest' 
                ? 'Backtest your strategy using historical data to analyze performance.' 
                : 'Trade in real-time using current market data and live recommendations.'}
            </Typography>
          </Box>

          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <SectionTitle>STOCK SELECTION</SectionTitle>
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  variant="outlined"
                  label="Add Ticker"
                  value={newTicker}
                  onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddTicker()}
                  InputProps={{
                    endAdornment: (
                      <Button 
                        onClick={handleAddTicker}
                        variant="contained"
                        color="primary"
                        disabled={!newTicker}
                      >
                        Add
                      </Button>
                    ),
                  }}
                  sx={{ mb: 2 }}
                />
                <Box sx={{ display: 'flex', flexWrap: 'wrap', mt: 1 }}>
                  {tickers.map((ticker) => (
                    <TickerChip
                      key={ticker}
                      label={ticker}
                      onDelete={() => handleRemoveTicker(ticker)}
                    />
                  ))}
                </Box>
              </Box>

              <SectionTitle>DATE RANGE</SectionTitle>
              <Box sx={{ mb: 3 }}>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      variant="outlined"
                      label="Start Date"
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      InputLabelProps={{ shrink: true }}
                      disabled={tradingMode === 'realtime'}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      variant="outlined"
                      label="End Date"
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      InputLabelProps={{ shrink: true }}
                      disabled={tradingMode === 'realtime'}
                    />
                  </Grid>
                </Grid>
                {tradingMode === 'realtime' && (
                  <Typography variant="body2" sx={{ mt: 1, color: '#b0b0b0' }}>
                    Date range is not applicable in real-time trading mode.
                  </Typography>
                )}
              </Box>
            </Grid>

            <Grid item xs={12} md={6}>
              <SectionTitle>INITIAL CASH</SectionTitle>
              <Box sx={{ mb: 3 }}>
                <TextField
                  fullWidth
                  variant="outlined"
                  label="Initial Cash"
                  type="number"
                  value={initialCash}
                  onChange={(e) => setInitialCash(Number(e.target.value))}
                  InputProps={{
                    startAdornment: <Typography sx={{ mr: 1 }}>$</Typography>,
                  }}
                  sx={{ mb: 1 }}
                />
                <Slider
                  value={initialCash}
                  onChange={(_, value) => setInitialCash(value as number)}
                  min={10000}
                  max={1000000}
                  step={10000}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `$${value.toLocaleString()}`}
                />
              </Box>

              <SectionTitle>MARGIN REQUIREMENT</SectionTitle>
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" gutterBottom>
                  {marginRequirement}%
                </Typography>
                <Slider
                  value={marginRequirement}
                  onChange={(_, value) => setMarginRequirement(value as number)}
                  min={0}
                  max={100}
                  step={5}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value}%`}
                />
              </Box>

              <Divider sx={{ my: 2, borderColor: 'rgba(255, 255, 255, 0.1)' }} />

              <Box>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showReasoning}
                      onChange={(e) => setShowReasoning(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Show Reasoning"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={sequential}
                      onChange={(e) => setSequential(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Sequential Processing"
                />
              </Box>
            </Grid>
          </Grid>

          <Box sx={{ mt: 4, textAlign: 'center' }}>
            <Typography variant="body2" sx={{ mb: 2, color: '#b0b0b0' }}>
              You've selected {selectedAnalysts.length} analysts for your AI hedge fund team.
            </Typography>
            <GlowButton
              onClick={handleSubmit}
              disabled={isSubmitting || tickers.length === 0}
              startIcon={isSubmitting && <CircularProgress size={20} color="inherit" />}
            >
              {isSubmitting 
                ? 'LAUNCHING...' 
                : tradingMode === 'backtest' 
                  ? 'START BACKTEST' 
                  : 'START REAL-TIME TRADING'}
            </GlowButton>
          </Box>
        </GlowPaper>
      </motion.div>
    </SetupContainer>
  );
};

export default PortfolioSetup;
