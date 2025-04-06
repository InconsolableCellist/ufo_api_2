import React, { useEffect, useRef, useState } from 'react';
import { useAgent } from '../context/AgentContext';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Card,
  CardContent,
  Button,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Alert
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import StarIcon from '@mui/icons-material/Star';
import FlagIcon from '@mui/icons-material/Flag';

const Goals = () => {
  const { goalData, loading, error, actions, simulationStatus } = useAgent();
  const shortTermGoalsRef = useRef(null);
  const [scrollPosition, setScrollPosition] = useState(0);

  // Save scroll position before component updates
  useEffect(() => {
    if (shortTermGoalsRef.current) {
      setScrollPosition(shortTermGoalsRef.current.scrollTop);
    }
  }, [goalData?.short_term_goals]);

  // Restore scroll position after update
  useEffect(() => {
    if (shortTermGoalsRef.current) {
      shortTermGoalsRef.current.scrollTop = scrollPosition;
    }
  }, [scrollPosition, loading]);

  // Always fetch goals when component mounts
  useEffect(() => {
    console.log('Goals component mounted, fetching goals data');
    fetchGoals();
    
    // Set up a refresh interval when viewing this component
    const refreshInterval = setInterval(() => {
      console.log('Auto-refreshing goals data');
      fetchGoals();
    }, 30000); // Refresh every 30 seconds
    
    // Clean up interval on unmount
    return () => {
      clearInterval(refreshInterval);
    };
  }, []);

  const fetchGoals = async () => {
    try {
      console.log('Fetching goals data');
      await actions.fetchGoals();
      console.log('Goals data fetched successfully');
    } catch (err) {
      console.error('Error fetching goals:', err);
    }
  };

  // Sort short-term goals to show most recent first
  const getSortedShortTermGoals = () => {
    if (!goalData?.short_term_goals || !goalData.short_term_goals.length) {
      return [];
    }
    
    // Create a copy of goals array and reverse it to show most recent first
    return [...goalData.short_term_goals].reverse();
  };

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Agent Goals</Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={fetchGoals}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      )}

      {!simulationStatus?.running && (!goalData?.short_term_goals?.length && !goalData?.long_term_goal) && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Simulation is not running. Start the simulation to access agent goals, or browse cached goals if available.
        </Alert>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Box>
          {/* Long-term goal */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <StarIcon color="primary" sx={{ mr: 1 }} />
                Long-term Goal
              </Typography>
              
              {goalData?.long_term_goal ? (
                <Paper sx={{ p: 2, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                  <Typography variant="body1">
                    {goalData.long_term_goal?.text || 'No description available'}
                    {goalData.long_term_goal?.duration && (
                      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                        Active for: {goalData.long_term_goal.duration} ({goalData.long_term_goal.cycles || 0} cycles)
                      </Typography>
                    )}
                  </Typography>
                </Paper>
              ) : (
                <Typography color="text.secondary" sx={{ fontStyle: 'italic' }}>
                  No long-term goal has been set.
                </Typography>
              )}
            </CardContent>
          </Card>
          
          {/* Short-term goals */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <FlagIcon color="secondary" sx={{ mr: 1 }} />
                Short-term Goals
              </Typography>
              
              {goalData?.short_term_goals && goalData.short_term_goals.length > 0 ? (
                <List
                  ref={shortTermGoalsRef} 
                  sx={{ 
                    maxHeight: '300px',
                    overflow: 'auto'
                  }}
                >
                  {getSortedShortTermGoals().map((goal, index) => (
                    <React.Fragment key={index}>
                      {index > 0 && <Divider component="li" />}
                      <ListItem>
                        <ListItemIcon>
                          <RadioButtonUncheckedIcon color="action" />
                        </ListItemIcon>
                        <ListItemText 
                          primary={goal.text || 'No description'} 
                          secondary={goal.duration && `Active for: ${goal.duration} (${goal.cycles || 0} cycles)`}
                        />
                      </ListItem>
                    </React.Fragment>
                  ))}
                </List>
              ) : (
                <Typography color="text.secondary" sx={{ fontStyle: 'italic' }}>
                  No short-term goals have been set.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
};

export default Goals; 