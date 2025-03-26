import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Typography, Button, Grid, Paper, Card, CardContent, CardMedia, Chip, styled } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { getAnalysts } from '../api/api';
import { Analyst } from '../types';

// Styled components for the retro-futuristic arcade look
const SelectionGrid = styled(Grid)(({ theme }) => ({
  marginTop: theme.spacing(4),
}));

const CharacterCard = styled(Card)<{ selected?: boolean; maincolor: string }>(({ theme, selected, maincolor }) => ({
  height: '100%',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  transform: selected ? 'scale(1.05)' : 'scale(1)',
  border: selected ? `3px solid ${maincolor}` : '1px solid rgba(255, 255, 255, 0.1)',
  boxShadow: selected 
    ? `0 0 15px ${maincolor}, 0 0 30px ${maincolor}50` 
    : '0 4px 20px rgba(0, 0, 0, 0.15)',
  backgroundColor: 'rgba(30, 30, 46, 0.9)',
  backdropFilter: 'blur(10px)',
  position: 'relative',
  '&:hover': {
    transform: 'scale(1.05)',
    boxShadow: `0 0 15px ${maincolor}, 0 0 30px ${maincolor}50`,
  },
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: 'inherit',
    padding: '2px',
    background: `linear-gradient(135deg, ${maincolor}, transparent)`,
    WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
    WebkitMaskComposite: 'xor',
    maskComposite: 'exclude',
    pointerEvents: 'none',
  },
}));

const CharacterImage = styled(CardMedia)(({ theme }) => ({
  height: 180,
  backgroundSize: 'contain',
  margin: '0 auto',
  position: 'relative',
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: 0,
    left: 0,
    width: '100%',
    height: '30%',
    background: 'linear-gradient(to top, rgba(30, 30, 46, 1), transparent)',
  },
}));

const AnalystName = styled(Typography)(({ theme }) => ({
  fontFamily: "'Press Start 2P', monospace",
  fontSize: '1rem',
  textAlign: 'center',
  marginBottom: theme.spacing(1),
  textShadow: '0 0 10px rgba(0, 245, 255, 0.7)',
}));

const SelectButton = styled(Button)<{ maincolor: string }>(({ theme, maincolor }) => ({
  marginTop: theme.spacing(2),
  background: `linear-gradient(45deg, ${maincolor}99 0%, ${maincolor} 100%)`,
  color: '#000',
  fontFamily: "'Press Start 2P', monospace",
  fontSize: '0.7rem',
  padding: theme.spacing(1, 2),
  '&:hover': {
    background: `linear-gradient(45deg, ${maincolor} 0%, ${maincolor}99 100%)`,
  },
}));

const StrategyChip = styled(Chip)<{ maincolor: string }>(({ theme, maincolor }) => ({
  margin: theme.spacing(0.5),
  backgroundColor: `${maincolor}40`,
  borderColor: maincolor,
  borderWidth: 1,
  borderStyle: 'solid',
  color: '#fff',
  fontSize: '0.6rem',
  height: 24,
  '& .MuiChip-label': {
    padding: theme.spacing(0, 1),
  },
}));

const DetailPane = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginTop: theme.spacing(3),
  backgroundColor: 'rgba(18, 18, 30, 0.8)',
  backdropFilter: 'blur(10px)',
  borderRadius: theme.shape.borderRadius,
  border: '1px solid rgba(255, 255, 255, 0.1)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
}));

const DetailTitle = styled(Typography)(({ theme }) => ({
  fontFamily: "'Press Start 2P', monospace",
  fontSize: '1rem',
  marginBottom: theme.spacing(2),
  color: theme.palette.primary.main,
  textShadow: `0 0 10px ${theme.palette.primary.main}`,
}));

const StartButton = styled(Button)(({ theme }) => ({
  fontSize: '1.2rem',
  padding: theme.spacing(1.5, 4),
  marginTop: theme.spacing(4),
  fontFamily: "'Press Start 2P', monospace",
  background: 'linear-gradient(45deg, #ff00ff 0%, #00f5ff 100%)',
  color: '#000',
  '&:hover': {
    background: 'linear-gradient(45deg, #00f5ff 0%, #ff00ff 100%)',
  },
  '&:disabled': {
    background: 'rgba(255, 255, 255, 0.1)',
    color: 'rgba(255, 255, 255, 0.3)',
  },
}));

const ScanlineEffect = styled(Box)(() => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  pointerEvents: 'none',
  zIndex: 10,
  backgroundImage: 'linear-gradient(to bottom, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0) 50%, rgba(255,255,255,0.03) 100%)',
  backgroundSize: '100% 8px',
  opacity: 0.15,
}));

const AgentSelection: React.FC = () => {
  const [analysts, setAnalysts] = useState<Analyst[]>([]);
  const [selectedAnalysts, setSelectedAnalysts] = useState<string[]>([]);
  const [featuredAnalyst, setFeaturedAnalyst] = useState<Analyst | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const navigate = useNavigate();

  // Fetch the analysts data
  useEffect(() => {
    const fetchAnalysts = async () => {
      try {
        const data = await getAnalysts();
        setAnalysts(data);
        if (data.length > 0) {
          setFeaturedAnalyst(data[0]);
        }
        setLoading(false);
      } catch (error) {
        console.error('Error fetching analysts:', error);
        setLoading(false);
      }
    };

    fetchAnalysts();
  }, []);

  const handleCardClick = (analyst: Analyst) => {
    setFeaturedAnalyst(analyst);
  };

  const toggleAnalystSelection = (analystId: string) => {
    setSelectedAnalysts((prev) => {
      if (prev.includes(analystId)) {
        return prev.filter((id) => id !== analystId);
      } else {
        return [...prev, analystId];
      }
    });
  };

  const handleContinue = () => {
    if (selectedAnalysts.length > 0) {
      // Get the full analyst objects for the selected analysts
      const selectedAnalystObjects = analysts.filter(analyst => selectedAnalysts.includes(analyst.id));
      
      // Navigate to setup page with selected analysts
      navigate('/setup', { state: { selectedAnalysts: selectedAnalystObjects } });
    }
  };

  const isAnalystSelected = (analystId: string) => selectedAnalysts.includes(analystId);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
        <Typography variant="h4" sx={{ fontFamily: "'Press Start 2P', monospace", color: '#00f5ff' }}>
          LOADING ANALYSTS...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ position: 'relative', overflow: 'hidden' }}>
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
          SELECT YOUR TEAM
        </Typography>
        <Typography 
          variant="body1" 
          sx={{ maxWidth: 800, mx: 'auto', color: '#b0b0b0' }}
        >
          Choose the financial analysts that will power your AI hedge fund. Each analyst has unique strengths and investment strategies.
        </Typography>
        <Typography 
          variant="body2" 
          sx={{ 
            mt: 1, 
            color: '#00f5ff', 
            fontFamily: "'Press Start 2P', monospace", 
            fontSize: '0.7rem' 
          }}
        >
          {selectedAnalysts.length} ANALYSTS SELECTED
        </Typography>
      </Box>

      <SelectionGrid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Grid container spacing={2}>
            {analysts.map((analyst) => (
              <Grid item xs={6} sm={4} md={3} key={analyst.id}>
                <motion.div
                  whileHover={{ y: -5 }}
                  whileTap={{ y: 5 }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <CharacterCard 
                    selected={isAnalystSelected(analyst.id)}
                    maincolor={analyst.color}
                    onClick={() => handleCardClick(analyst)}
                  >
                    <CharacterImage
                      image={analyst.avatarSrc || `/avatars/default.png`}
                      title={analyst.displayName}
                    />
                    <CardContent>
                      <AnalystName color="primary">
                        {analyst.displayName}
                      </AnalystName>
                      <Typography variant="body2" sx={{ fontSize: '0.75rem', color: '#b0b0b0', height: 60, overflow: 'hidden' }}>
                        {analyst.description}
                      </Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                        <SelectButton
                          maincolor={analyst.color}
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleAnalystSelection(analyst.id);
                          }}
                          size="small"
                        >
                          {isAnalystSelected(analyst.id) ? 'SELECTED' : 'SELECT'}
                        </SelectButton>
                      </Box>
                    </CardContent>
                  </CharacterCard>
                </motion.div>
              </Grid>
            ))}
          </Grid>
        </Grid>

        <Grid item xs={12} md={4}>
          <AnimatePresence mode="wait">
            {featuredAnalyst && (
              <motion.div
                key={featuredAnalyst.id}
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -50 }}
                transition={{ duration: 0.3 }}
              >
                <DetailPane>
                  <DetailTitle>
                    {featuredAnalyst.displayName}
                  </DetailTitle>
                  
                  <Typography variant="body1" sx={{ mb: 2, color: '#e0e0e0' }}>
                    {featuredAnalyst.description}
                  </Typography>
                  
                  <DetailTitle sx={{ mt: 3, fontSize: '0.8rem' }}>
                    INVESTMENT STRATEGY
                  </DetailTitle>
                  <Typography variant="body2" sx={{ color: '#b0b0b0', mb: 2 }}>
                    {featuredAnalyst.investmentStrategy}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', mb: 2, mt: 3 }}>
                    <Typography sx={{ width: '100%', mb: 1, fontFamily: "'Press Start 2P', monospace", fontSize: '0.8rem', color: '#00f5ff' }}>
                      STRENGTHS
                    </Typography>
                    {featuredAnalyst.strengths.map((strength, index) => (
                      <StrategyChip 
                        key={index} 
                        label={strength} 
                        maincolor={featuredAnalyst.color} 
                        size="small" 
                      />
                    ))}
                  </Box>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', mb: 2 }}>
                    <Typography sx={{ width: '100%', mb: 1, fontFamily: "'Press Start 2P', monospace", fontSize: '0.8rem', color: '#ff00ff' }}>
                      KEY METRICS
                    </Typography>
                    {featuredAnalyst.keyMetrics.map((metric, index) => (
                      <StrategyChip 
                        key={index} 
                        label={metric} 
                        maincolor={featuredAnalyst.color} 
                        size="small" 
                      />
                    ))}
                  </Box>
                  
                  <Box sx={{ mt: 3, textAlign: 'center' }}>
                    <Button
                      variant="outlined"
                      color="secondary"
                      onClick={() => toggleAnalystSelection(featuredAnalyst.id)}
                      sx={{ 
                        fontFamily: "'Press Start 2P', monospace",
                        fontSize: '0.7rem',
                        borderWidth: 2,
                        borderColor: featuredAnalyst.color,
                        color: featuredAnalyst.color
                      }}
                    >
                      {isAnalystSelected(featuredAnalyst.id) ? 'REMOVE FROM TEAM' : 'ADD TO TEAM'}
                    </Button>
                  </Box>
                </DetailPane>
              </motion.div>
            )}
          </AnimatePresence>
        </Grid>
      </SelectionGrid>

      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <StartButton 
          onClick={handleContinue}
          disabled={selectedAnalysts.length === 0}
        >
          CONTINUE
        </StartButton>
      </Box>

      <ScanlineEffect />
    </Box>
  );
};

export default AgentSelection;
