import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ffff', // Cyan for retro-futuristic neon look
      light: '#80ffff',
      dark: '#00cccc',
      contrastText: '#000000',
    },
    secondary: {
      main: '#ff00ff', // Magenta for contrast
      light: '#ff80ff',
      dark: '#cc00cc',
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
      main: '#ff5555',
    },
    warning: {
      main: '#ffaa00',
    },
    info: {
      main: '#00aaff',
    },
    success: {
      main: '#00ff7f',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontFamily: '"Orbitron", sans-serif',
      fontWeight: 700,
    },
    h2: {
      fontFamily: '"Orbitron", sans-serif',
      fontWeight: 700,
    },
    h3: {
      fontFamily: '"Orbitron", sans-serif',
      fontWeight: 600,
    },
    h4: {
      fontFamily: '"Orbitron", sans-serif',
      fontWeight: 600,
    },
    h5: {
      fontFamily: '"Orbitron", sans-serif',
      fontWeight: 500,
    },
    h6: {
      fontFamily: '"Orbitron", sans-serif',
      fontWeight: 500,
    },
    button: {
      fontFamily: '"Orbitron", sans-serif',
      fontWeight: 500,
      textTransform: 'none',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          textTransform: 'none',
          padding: '8px 16px',
          '&:hover': {
            boxShadow: '0 0 10px #00ffff, 0 0 20px rgba(0, 255, 255, 0.5)',
          },
        },
        contained: {
          background: 'linear-gradient(45deg, #00ccff 30%, #00ffff 90%)',
          color: '#000000',
          fontWeight: 600,
        },
        outlined: {
          borderColor: '#00ffff',
          color: '#00ffff',
          '&:hover': {
            borderColor: '#00ffff',
            backgroundColor: 'rgba(0, 255, 255, 0.1)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: 'rgba(30, 30, 30, 0.8)',
          borderRadius: 0,
          border: '1px solid rgba(0, 255, 255, 0.2)',
        },
      },
    },
    MuiCardHeader: {
      styleOverrides: {
        root: {
          fontFamily: '"Orbitron", sans-serif',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 4,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 0,
            '& fieldset': {
              borderColor: 'rgba(0, 255, 255, 0.3)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(0, 255, 255, 0.5)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#00ffff',
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(30, 30, 30, 0.8)',
          borderRadius: 0,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(90deg, #121212 0%, #1a1a1a 100%)',
          boxShadow: '0 0 10px rgba(0, 255, 255, 0.3)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundImage: 'linear-gradient(180deg, #121212 0%, #1a1a1a 100%)',
          borderRight: '1px solid rgba(0, 255, 255, 0.1)',
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          fontFamily: '"Orbitron", sans-serif',
          fontWeight: 500,
          backgroundColor: 'rgba(0, 255, 255, 0.1)',
        },
      },
    },
  },
});

export default theme;
