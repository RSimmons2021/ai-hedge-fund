import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, Typography, Grid, Card, CardContent, CardMedia, 
  Button, CircularProgress, Chip, Fade, Paper
} from '@mui/material';
import { styled } from '@mui/material/styles';
import { useSpring, animated } from 'react-spring';
import api from '../utils/api';

// Styled components for the retro-futuristic UI
const SelectionContainer = styled(Box)(({ theme }) => ({
  minHeight: 'calc(100vh - 64px)',
  padding: theme.spacing(4),
  backgroundImage: `linear-gradient(rgba(18, 18, 18, 0.8), rgba(18, 18, 18, 0.9)),
    linear-gradient(90deg, rgba(0, 255, 255, 0.1) 1px, transparent 1px),
    linear-gradient(rgba(0, 255, 255, 0.1) 1px, transparent 1px)`,
  backgroundSize: '100% 100%, 20px 20px, 20px 20px',
  backgroundPosition: '0 0, 0 0, 0 0',
  overflow: 'hidden',
}));

const AgentCard = styled(Card)(({ theme, selected }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: 'rgba(30, 30, 30, 0.8)',
  border: selected ? '2px solid #00ffff' : '1px solid rgba(0, 255, 255, 0.3)',
  boxShadow: selected ? '0 0 15px #00ffff, 0 0 30px rgba(0, 255, 255, 0.5)' : 'none',
  transition: 'all 0.3s ease',
  transform: selected ? 'translateY(-10px) scale(1.05)' : 'none',
  '&:hover': {
    transform: 'translateY(-5px) scale(1.02)',
    boxShadow: '0 0 10px rgba(0, 255, 255, 0.5)',
    border: '1px solid rgba(0, 255, 255, 0.5)',
  },
}));

const AgentImage = styled(CardMedia)(({ theme }) => ({
  height: 200,
  backgroundSize: 'cover',
  backgroundPosition: 'center',
  borderBottom: '1px solid rgba(0, 255, 255, 0.3)',
  position: 'relative',
  '&::after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.7) 100%)',
  },
}));

const AgentName = styled(Typography)(({ theme }) => ({
  fontFamily: '"Orbitron", sans-serif',
  fontWeight: 700,
  fontSize: '1.5rem',
  marginBottom: theme.spacing(1),
  textShadow: '0 0 5px rgba(0, 255, 255, 0.5)',
  color: '#ffffff',
}));

const AgentStrategy = styled(Typography)(({ theme }) => ({
  fontSize: '0.875rem',
  color: theme.palette.text.secondary,
  marginBottom: theme.spacing(2),
  height: 100,
  overflow: 'auto',
}));

const SelectButton = styled(Button)(({ theme }) => ({
  fontFamily: '"Orbitron", sans-serif',
  marginTop: 'auto',
  background: 'linear-gradient(45deg, #00ccff 30%, #00ffff 90%)',
  color: '#000000',
  fontWeight: 600,
  '&:hover': {
    boxShadow: '0 0 10px #00ffff, 0 0 20px rgba(0, 255, 255, 0.5)',
  },
}));

const AnimatedTitle = animated(Typography);

const AgentSelection = () => {
  const [analysts, setAnalysts] = useState({});
  const [selectedAnalysts, setSelectedAnalysts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Animation properties
  const titleAnimation = useSpring({
    from: { opacity: 0, transform: 'translateY(-50px)' },
    to: { opacity: 1, transform: 'translateY(0)' },
    config: { tension: 100, friction: 10 },
  });

  // Get placeholder images for agents
  const getAgentImage = (agentId) => {
    // In a real app, you'd have actual images for each agent
    // For now, we'll use a placeholder with different colors based on the agent ID
    const colors = [
      'linear-gradient(135deg, #00c6ff 0%, #0072ff 100%)',
      'linear-gradient(135deg, #f5515f 0%, #9f041b 100%)',
      'linear-gradient(135deg, #654ea3 0%, #eaafc8 100%)',
      'linear-gradient(135deg, #ff9966 0%, #ff5e62 100%)',
      'linear-gradient(135deg, #00b09b 0%, #96c93d 100%)',
      'linear-gradient(135deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%)',
      'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      'linear-gradient(135deg, #43cea2 0%, #185a9d 100%)',
      'linear-gradient(135deg, #ba5370 0%, #f4e2d8 100%)',
      'linear-gradient(135deg, #ff00cc 0%, #333399 100%)',
      'linear-gradient(135deg, #40e0d0 0%, #ff8c00 50%, #ff0080 100%)'
    ];
    
    // Use the agent's order as an index, or fallback to a random color
    const index = analysts[agentId]?.order % colors.length || Math.floor(Math.random() * colors.length);
    return colors[index];
  };

  // Fetch analysts data
  useEffect(() => {
    const fetchAnalysts = async () => {
      try {
        setLoading(true);
        const data = await api.getAnalysts();
        setAnalysts(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load analysts. Please try again.');
        setLoading(false);
      }
    };

    fetchAnalysts();
  }, []);

  // Handle analyst selection/deselection
  const toggleAnalyst = (analystId) => {
    if (selectedAnalysts.includes(analystId)) {
      setSelectedAnalysts(selectedAnalysts.filter(id => id !== analystId));
    } else {
      setSelectedAnalysts([...selectedAnalysts, analystId]);
    }
  };

  // Handle continue button click
  const handleContinue = () => {
    if (selectedAnalysts.length > 0) {
      // Store selected analysts in session storage for use in other components
      sessionStorage.setItem('selectedAnalysts', JSON.stringify(selectedAnalysts));
      navigate('/setup');
    }
  };

  // Render loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 64px)' }}>
        <CircularProgress size={60} sx={{ color: '#00ffff' }} />
      </Box>
    );
  }

  // Render error state
  if (error) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 'calc(100vh - 64px)' }}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <SelectionContainer>
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
          Select Your Investment Agents
        </AnimatedTitle>
        <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
          Choose the financial experts who will guide your investment strategy
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {Object.entries(analysts)
          .sort(([, a], [, b]) => a.order - b.order)
          .map(([id, analyst], index) => (
            <Fade in={true} key={id} style={{ transitionDelay: `${index * 100}ms` }}>
              <Grid item xs={12} sm={6} md={4} lg={3}>
                <AgentCard selected={selectedAnalysts.includes(id)}>
                  <AgentImage
                    sx={{ background: getAgentImage(id) }}
                  />
                  <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <AgentName>{analyst.name}</AgentName>
                      <Chip 
                        label={selectedAnalysts.includes(id) ? "Selected" : "Available"}
                        size="small"
                        color={selectedAnalysts.includes(id) ? "primary" : "default"}
                        sx={{ 
                          borderRadius: '4px',
                          fontFamily: '"Orbitron", sans-serif',
                          fontSize: '0.7rem'
                        }}
                      />
                    </Box>
                    <AgentStrategy>
                      {analyst.strategy.split('\n')[0]}
                    </AgentStrategy>
                    <SelectButton 
                      variant={selectedAnalysts.includes(id) ? "contained" : "outlined"}
                      onClick={() => toggleAnalyst(id)}
                    >
                      {selectedAnalysts.includes(id) ? "Deselect" : "Select"}
                    </SelectButton>
                  </CardContent>
                </AgentCard>
              </Grid>
            </Fade>
          ))}
      </Grid>

      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 6 }}>
        <Button
          variant="contained"
          size="large"
          disabled={selectedAnalysts.length === 0}
          onClick={handleContinue}
          sx={{ 
            minWidth: 200, 
            py: 1.5,
            fontFamily: '"Orbitron", sans-serif',
            fontSize: '1.1rem',
            fontWeight: 700,
            letterSpacing: '1px',
            background: selectedAnalysts.length > 0 ? 
              'linear-gradient(45deg, #00ccff 30%, #00ffff 90%)' : 
              'rgba(255, 255, 255, 0.1)',
            boxShadow: selectedAnalysts.length > 0 ?
              '0 0 10px rgba(0, 255, 255, 0.5)' :
              'none',
            '&:hover': {
              background: selectedAnalysts.length > 0 ?
                'linear-gradient(45deg, #00ffff 30%, #00ccff 90%)' :
                'rgba(255, 255, 255, 0.1)',
              boxShadow: selectedAnalysts.length > 0 ?
                '0 0 15px rgba(0, 255, 255, 0.7)' :
                'none',
            }
          }}
        >
          Continue to Simulation
        </Button>
      </Box>

      <Box sx={{ 
        position: 'fixed', 
        bottom: 20, 
        right: 20, 
        p: 2, 
        bgcolor: 'rgba(0, 0, 0, 0.7)', 
        border: '1px solid rgba(0, 255, 255, 0.3)',
        borderRadius: '4px'
      }}>
        <Typography variant="body2" sx={{ color: '#00ffff', fontFamily: '"Orbitron", sans-serif' }}>
          Selected: {selectedAnalysts.length} / {Object.keys(analysts).length}
        </Typography>
      </Box>
    </SelectionContainer>
  );
};

export default AgentSelection;
