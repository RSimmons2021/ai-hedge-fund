import React from 'react';
import { Box, AppBar, Toolbar, Typography, Container, styled } from '@mui/material';
import { useLocation } from 'react-router-dom';

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  background: 'linear-gradient(90deg, #121212 0%, #1a1a2e 100%)',
  borderBottom: `1px solid ${theme.palette.primary.main}`,
  boxShadow: '0 4px 30px rgba(0, 0, 0, 0.3)',
}));

const GlowText = styled(Typography)(({ theme }) => ({
  color: theme.palette.primary.main,
  textShadow: `0 0 10px ${theme.palette.primary.main}, 0 0 20px ${theme.palette.primary.main}`,
  fontFamily: "'Press Start 2P', monospace",
  textTransform: 'uppercase',
  letterSpacing: '0.1em',
}));

const MainContent = styled(Container)(({ theme }) => ({
  marginTop: theme.spacing(4),
  marginBottom: theme.spacing(4),
  minHeight: 'calc(100vh - 128px)',
  padding: theme.spacing(3),
}));

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();
  
  // Determine title based on current route
  let title = 'AI Hedge Fund';
  if (location.pathname === '/') {
    title = 'Select Your Analysts';
  } else if (location.pathname === '/setup') {
    title = 'Portfolio Setup';
  } else if (location.pathname === '/dashboard') {
    title = 'Portfolio Dashboard';
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <StyledAppBar position="sticky">
        <Toolbar>
          <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center' }}>
            <GlowText variant="h5" component="div">
              AI Hedge Fund
            </GlowText>
          </Box>
          <Typography 
            variant="h6" 
            sx={{ 
              fontFamily: "'Press Start 2P', monospace",
              fontSize: '0.8rem',
              color: '#80f9ff'
            }}
          >
            {title}
          </Typography>
        </Toolbar>
      </StyledAppBar>
      <MainContent maxWidth="xl">
        {children}
      </MainContent>
      <Box 
        component="footer" 
        sx={{ 
          p: 2, 
          mt: 'auto',
          backgroundColor: '#121212',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          textAlign: 'center'
        }}
      >
        <Typography variant="body2" color="text.secondary">
          © 2025 AI Hedge Fund System - Retro Futuristic Edition
        </Typography>
      </Box>
    </Box>
  );
};

export default Layout;
