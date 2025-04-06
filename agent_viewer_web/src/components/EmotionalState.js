import React, { useEffect } from 'react';
import { useAgent } from '../context/AgentContext';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  LinearProgress,
  Card,
  CardContent,
  Button,
  Grid,
  Divider
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const COLORS = [
  '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#A569BD',
  '#5DADE2', '#48C9B0', '#F4D03F', '#EB984E', '#AF7AC5'
];

const EmotionalState = () => {
  const { emotionData, loading, error, actions } = useAgent();

  useEffect(() => {
    fetchEmotionalState();
  }, []);

  const fetchEmotionalState = async () => {
    try {
      await actions.fetchSummary();
    } catch (err) {
      console.error('Error fetching emotional state:', err);
    }
  };

  const formatEmotionData = () => {
    if (!emotionData || Object.keys(emotionData).length === 0) {
      return [];
    }

    return Object.entries(emotionData)
      .filter(([name, value]) => typeof value === 'number' && name !== 'mood')
      .map(([name, value]) => ({
        name,
        value: parseFloat(value.toFixed(2))
      }))
      .sort((a, b) => b.value - a.value); // Sort by value descending
  };

  const emotionChartData = formatEmotionData();
  const totalEmotionValue = emotionChartData.reduce((sum, item) => sum + item.value, 0);

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Emotional State</Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={fetchEmotionalState}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Paper sx={{ p: 2, mb: 2, bgcolor: '#ffebee' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : !emotionData || Object.keys(emotionData).length === 0 ? (
        <Typography>No emotional data available</Typography>
      ) : (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Emotional Intensity
                </Typography>
                <Box sx={{ mb: 4 }}>
                  {emotionChartData.map((emotion, index) => (
                    <Box key={emotion.name} sx={{ mb: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                        <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                          {emotion.name}
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                          {emotion.value.toFixed(2)}
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={emotion.value * 100}
                        sx={{
                          height: 10,
                          borderRadius: 5,
                          backgroundColor: `${COLORS[index % COLORS.length]}33`,
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: COLORS[index % COLORS.length],
                            borderRadius: 5,
                          }
                        }}
                      />
                    </Box>
                  ))}
                </Box>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Emotional Distribution
                </Typography>
                {emotionChartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={emotionChartData}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value.toFixed(2)}`}
                      >
                        {emotionChartData.map((_, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value) => [value.toFixed(2), 'Intensity']}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <Typography>No emotion data to display</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Overall Mood Analysis
                </Typography>
                
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 2 }}>
                  <Paper sx={{ p: 2, flex: '1 1 200px' }}>
                    <Typography variant="subtitle1">Dominant Emotion</Typography>
                    <Typography variant="h4" sx={{ 
                      color: emotionChartData.length > 0 ? COLORS[0] : 'inherit',
                      textTransform: 'capitalize'
                    }}>
                      {emotionChartData.length > 0 ? emotionChartData[0].name : 'None'}
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      {emotionChartData.length > 0 && totalEmotionValue > 0
                        ? `${((emotionChartData[0].value / totalEmotionValue) * 100).toFixed(1)}% of total emotional state` 
                        : ''}
                    </Typography>
                  </Paper>
                  
                  <Paper sx={{ p: 2, flex: '1 1 200px' }}>
                    <Typography variant="subtitle1">Secondary Emotion</Typography>
                    <Typography variant="h5" sx={{ 
                      color: emotionChartData.length > 1 ? COLORS[1] : 'inherit',
                      textTransform: 'capitalize'
                    }}>
                      {emotionChartData.length > 1 ? emotionChartData[1].name : 'None'}
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>
                      {emotionChartData.length > 1 && totalEmotionValue > 0
                        ? `${((emotionChartData[1].value / totalEmotionValue) * 100).toFixed(1)}% of total emotional state` 
                        : ''}
                    </Typography>
                  </Paper>
                  
                  {emotionData.mood !== undefined && (
                    <Paper sx={{ p: 2, flex: '1 1 200px' }}>
                      <Typography variant="subtitle1">Overall Mood</Typography>
                      <Typography variant="h4" sx={{ 
                        color: emotionData.mood > 0 ? '#00C49F' : emotionData.mood < 0 ? '#FF8042' : '#888'
                      }}>
                        {emotionData.mood > 0.3 ? 'Positive' : 
                         emotionData.mood < -0.3 ? 'Negative' : 'Neutral'}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        Value: {typeof emotionData.mood === 'number' ? emotionData.mood.toFixed(2) : '0.00'} (scale: -1 to 1)
                      </Typography>
                    </Paper>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default EmotionalState; 