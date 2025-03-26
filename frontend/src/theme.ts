import { createTheme } from '@mui/material/styles';

// Create a dark theme with retro-futuristic aesthetic
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00f5ff', // Bright cyan
      light: '#80f9ff',
      dark: '#00c8d4',
      contrastText: '#000000',
    },
    secondary: {
      main: '#ff00ff', // Magenta
      light: '#ff80ff',
      dark: '#c800c8',
      contrastText: '#000000',
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
    error: {
      main: '#ff3d71',
    },
    warning: {
      main: '#ffaa00',
    },
    info: {
      main: '#0095ff',
    },
    success: {
      main: '#00e096',
    },
  },
  typography: {
    fontFamily: "'Press Start 2P', 'Roboto', 'Helvetica', 'Arial', sans-serif",
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '0.02em',
      marginBottom: '1rem',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 700,
      letterSpacing: '0.02em',
      marginBottom: '0.8rem',
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      letterSpacing: '0.02em',
      marginBottom: '0.6rem',
    },
    h4: {
      fontSize: '1.2rem',
      fontWeight: 600,
      letterSpacing: '0.02em',
      marginBottom: '0.4rem',
    },
    button: {
      fontWeight: 600,
      letterSpacing: '0.1em',
      textTransform: 'uppercase',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: `
        @font-face {
          font-family: 'Press Start 2P';
          font-style: normal;
          font-display: swap;
          font-weight: 400;
          src: url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        }
        body {
          background: linear-gradient(135deg, #121212 0%, #1a1a2e 100%);
          min-height: 100vh;
        }
      `,
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          textTransform: 'uppercase',
          fontWeight: 600,
          boxShadow: '0 4px 0 rgba(0, 0, 0, 0.2)',
          transition: 'all 0.3s',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 6px 0 rgba(0, 0, 0, 0.2)',
          },
          '&:active': {
            transform: 'translateY(0)',
            boxShadow: '0 2px 0 rgba(0, 0, 0, 0.2)',
          },
        },
        contained: {
          background: 'linear-gradient(45deg, #00c8d4 0%, #00f5ff 100%)',
          color: '#000000',
          border: '2px solid rgba(255, 255, 255, 0.1)',
          '&:hover': {
            background: 'linear-gradient(45deg, #00d8e4 0%, #40f7ff 100%)',
          },
        },
        outlined: {
          borderColor: '#00f5ff',
          color: '#00f5ff',
          borderWidth: 2,
          '&:hover': {
            borderColor: '#80f9ff',
            borderWidth: 2,
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(30, 30, 46, 0.8)',
          backdropFilter: 'blur(10px)',
          borderRadius: 12,
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          background: 'rgba(30, 30, 46, 0.8)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
  },
});

export default theme;
