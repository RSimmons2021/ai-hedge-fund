import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Box, CssBaseline } from '@mui/material';

// Pages
import AgentSelection from './pages/AgentSelection';
import Dashboard from './pages/Dashboard';
import SimulationSetup from './pages/SimulationSetup';

// Components
import NavBar from './components/NavBar';

function App() {
  return (
    <Router>
      <CssBaseline />
      <Box className="grid-background" sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <NavBar />
        <Box component="main" sx={{ flexGrow: 1, p: 0 }}>
          <Routes>
            <Route path="/" element={<AgentSelection />} />
            <Route path="/setup" element={<SimulationSetup />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </Box>
      </Box>
    </Router>
  );
}

export default App;
