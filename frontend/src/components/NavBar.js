import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import ShowChartIcon from '@mui/icons-material/ShowChart';

const NavBar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <AppBar position="static" sx={{ borderBottom: '1px solid rgba(0, 255, 255, 0.3)' }}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }} onClick={() => navigate('/')}>
          <ShowChartIcon sx={{ mr: 1, color: '#00ffff' }} />
          <Typography 
            variant="h5" 
            component="div" 
            sx={{ 
              flexGrow: 1, 
              fontFamily: '"Orbitron", sans-serif',
              fontWeight: 700,
              letterSpacing: '1px',
              background: 'linear-gradient(45deg, #00ffff, #00ccff)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 0 5px rgba(0, 255, 255, 0.5)'
            }}
          >
            AI HEDGE FUND
          </Typography>
        </Box>
        <Box sx={{ flexGrow: 1 }} />
        <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
          <Button 
            color="inherit" 
            onClick={() => navigate('/')} 
            sx={{ 
              mx: 1, 
              borderBottom: location.pathname === '/' ? '2px solid #00ffff' : 'none',
              '&:hover': { backgroundColor: 'rgba(0, 255, 255, 0.1)' }
            }}
          >
            Agents
          </Button>
          <Button 
            color="inherit" 
            onClick={() => navigate('/setup')} 
            sx={{ 
              mx: 1, 
              borderBottom: location.pathname === '/setup' ? '2px solid #00ffff' : 'none',
              '&:hover': { backgroundColor: 'rgba(0, 255, 255, 0.1)' }
            }}
          >
            Simulation
          </Button>
          <Button 
            color="inherit" 
            onClick={() => navigate('/dashboard')} 
            sx={{ 
              mx: 1, 
              borderBottom: location.pathname === '/dashboard' ? '2px solid #00ffff' : 'none',
              '&:hover': { backgroundColor: 'rgba(0, 255, 255, 0.1)' }
            }}
          >
            Dashboard
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default NavBar;
