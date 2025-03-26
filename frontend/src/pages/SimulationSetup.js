import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Typography, TextField, Button, Grid, Paper, Chip,
  FormControl, InputLabel, Select, MenuItem, CircularProgress,
  Divider, Fade, InputAdornment
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useSpring, animated } from 'react-spring';
import api from '../utils/api';

// Styled components
const SetupContainer = styled(Box)(({ theme }) => ({
  minHeight: 'calc(100vh - 64px)',
  padding: theme.spacing(4),
  backgroundImage: `linear-gradient(rgba(18, 18, 18, 0.8), rgba(18, 18, 18, 0.9)),
    linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px),
    linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px)`,
  backgroundSize: '100% 100%, 20px 20px, 20px 20px',
  backgroundPosition: '0 0, 0 0, 0 0',
}));

const FormPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  backgroundColor: 'rgba(30, 30, 30, 0.8)',
  border: '1px solid rgba(0, 255, 255, 0.3)',
  borderRadius: 0,
  boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)',
}));

const AnimatedTitle = animated(Typography);

const SimulationSetup = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingModels, setLoadingModels] = useState(true);
  const [error, setError] = useState(null);
  const [selectedAnalysts, setSelectedAnalysts] = useState([]);
  const [analysts, setAnalysts] = useState({});
  const navigate = useNavigate();

  // Form state
  const [formData, setFormData] = useState({
    tickers: '',
    initialCash: 100000,
    marginRequirement: 0,
    startDate: '',
    endDate: '',
    model: 'gpt-4o',
    showReasoning: false
  });

  // Animation properties
  const titleAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(-50px)' },
    to: { opacity: 1, transform: 'translateY(0)' },
    config: { tension: 100, friction: 10 },
  });

  // Fetch analysts and models data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Get selected analysts from session storage
        const storedAnalysts = sessionStorage.getItem('selectedAnalysts');
        if (storedAnalysts) {
          setSelectedAnalysts(JSON.parse(storedAnalysts));
        } else {
          // If no analysts selected, redirect to selection page
          navigate('/');
          return;
        }

        // Fetch analysts data
        const analystsData = await api.getAnalysts();
        setAnalysts(analystsData);

        // Fetch models
        setLoadingModels(true);
        const modelsData = await api.getModels();
        setModels(modelsData);
        setLoadingModels(false);

        // Set default dates (today and 3 months ago)
        const today = new Date();
        const threeMonthsAgo = new Date();
        threeMonthsAgo.setMonth(today.getMonth() - 3);
        
        setFormData(prev => ({
          ...prev,
          endDate: today.toISOString().split('T')[0],
          startDate: threeMonthsAgo.toISOString().split('T')[0]
        }));

        setLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data. Please try again.');
        setLoading(false);
        setLoadingModels(false);
      }
    };

    fetchData();
  }, [navigate]);

  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate form data
    if (!formData.tickers.trim()) {
      setError('Please enter at least one ticker symbol');
      return;
    }

    // Format tickers into an array
    const tickersArray = formData.tickers.split(',').map(ticker => ticker.trim().toUpperCase());
    
    // Store simulation parameters in session storage
    const simulationParams = {
      tickers: tickersArray,
      initial_cash: parseFloat(formData.initialCash),
      margin_requirement: parseFloat(formData.marginRequirement),
      start_date: formData.startDate,
      end_date: formData.endDate,
      model: formData.model,
      show_reasoning: formData.showReasoning,
      selected_analysts: selectedAnalysts
    };
    
    sessionStorage.setItem('simulationParams', JSON.stringify(simulationParams));
    
    // Navigate to dashboard
    navigate('/dashboard');
  };

  // Render loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 64px)' }}>
        <CircularProgress size={60} sx={{ color: '#00ffff' }} />
      </Box>
    );
  }

  return (
    <SetupContainer>
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <AnimatedTitle 
          variant="h2" 
          style={titleAnimation} 
          className="neon-text"
          sx={{ 
            mb: 2,
            fontWeight: 800,
            letterSpacing: '2px',
            textTransform: 'uppercase'
          }}
        >
          Simulation Setup
        </AnimatedTitle>
        <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
          Configure your trading parameters
        </Typography>
      </Box>

      <Grid container spacing={4} justifyContent="center">
        <Grid item xs={12} md={8} lg={6}>
          <Fade in={true} style={{ transitionDelay: '100ms' }}>
            <FormPaper elevation={3}>
              <form onSubmit={handleSubmit}>
                <Typography variant="h5" sx={{ mb: 3, fontFamily: '"Orbitron", sans-serif', color: '#00ffff' }}>
                  Trading Parameters
                </Typography>
                
                {error && (
                  <Typography color="error" sx={{ mb: 2 }}>
                    {error}
                  </Typography>
                )}

                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Ticker Symbols"
                      name="tickers"
                      value={formData.tickers}
                      onChange={handleInputChange}
                      placeholder="AAPL, MSFT, GOOGL"
                      helperText="Enter comma-separated ticker symbols"
                      required
                      variant="outlined"
                      InputProps={{
                        sx: { borderRadius: 0 }
                      }}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Initial Cash"
                      name="initialCash"
                      type="number"
                      value={formData.initialCash}
                      onChange={handleInputChange}
                      InputProps={{
                        startAdornment: <InputAdornment position="start">$</InputAdornment>,
                        sx: { borderRadius: 0 }
                      }}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Margin Requirement"
                      name="marginRequirement"
                      type="number"
                      value={formData.marginRequirement}
                      onChange={handleInputChange}
                      InputProps={{
                        startAdornment: <InputAdornment position="start">$</InputAdornment>,
                        sx: { borderRadius: 0 }
                      }}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Start Date"
                      name="startDate"
                      type="date"
                      value={formData.startDate}
                      onChange={handleInputChange}
                      InputLabelProps={{ shrink: true }}
                      InputProps={{
                        sx: { borderRadius: 0 }
                      }}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="End Date"
                      name="endDate"
                      type="date"
                      value={formData.endDate}
                      onChange={handleInputChange}
                      InputLabelProps={{ shrink: true }}
                      InputProps={{
                        sx: { borderRadius: 0 }
                      }}
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel id="model-select-label">LLM Model</InputLabel>
                      <Select
                        labelId="model-select-label"
                        id="model-select"
                        name="model"
                        value={formData.model}
                        onChange={handleInputChange}
                        label="LLM Model"
                        sx={{ borderRadius: 0 }}
                      >
                        {loadingModels ? (
                          <MenuItem value="">Loading models...</MenuItem>
                        ) : (
                          models.map((model) => (
                            <MenuItem key={model.id} value={model.id}>
                              {model.name} ({model.provider})
                            </MenuItem>
                          ))
                        )}
                      </Select>
                    </FormControl>
                  </Grid>
                </Grid>

                <Divider sx={{ my: 4, borderColor: 'rgba(0, 255, 255, 0.2)' }} />
                
                <Typography variant="h6" sx={{ mb: 2, fontFamily: '"Orbitron", sans-serif', color: '#00ffff' }}>
                  Selected Analysts
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 4 }}>
                  {selectedAnalysts.map((analystId) => (
                    <Chip
                      key={analystId}
                      label={analysts[analystId]?.name || analystId}
                      color="primary"
                      sx={{ 
                        borderRadius: '4px',
                        fontFamily: '"Orbitron", sans-serif',
                        fontSize: '0.8rem',
                        m: 0.5
                      }}
                    />
                  ))}
                </Box>

                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
                  <Button
                    variant="outlined"
                    onClick={() => navigate('/')}
                    sx={{ 
                      minWidth: 120,
                      fontFamily: '"Orbitron", sans-serif',
                      borderRadius: 0
                    }}
                  >
                    Back
                  </Button>
                  <Button
                    type="submit"
                    variant="contained"
                    sx={{ 
                      minWidth: 120,
                      fontFamily: '"Orbitron", sans-serif',
                      borderRadius: 0,
                      background: 'linear-gradient(45deg, #00ccff 30%, #00ffff 90%)',
                      color: '#000000',
                      fontWeight: 600,
                      '&:hover': {
                        boxShadow: '0 0 10px #00ffff, 0 0 20px rgba(0, 255, 255, 0.5)',
                      },
                    }}
                  >
                    Start Simulation
                  </Button>
                </Box>
              </form>
            </FormPaper>
          </Fade>
        </Grid>
      </Grid>
    </SetupContainer>
  );
};

export default SimulationSetup;
