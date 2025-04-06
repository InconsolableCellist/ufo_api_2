import React, { useState } from 'react';
import { AgentProvider } from './context/AgentContext';
import { Box, AppBar, Toolbar, Typography, Tabs, Tab, Button, Alert } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import MemoryBrowser from './components/MemoryBrowser';
import ThoughtSummaries from './components/ThoughtSummaries';
import EmotionGraph from './components/EmotionGraph';
import EmotionalState from './components/EmotionalState';
import Goals from './components/Goals';
import Journal from './components/Journal';
import ToolHistory from './components/ToolHistory';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import { useAgent } from './context/AgentContext';

// Define tab content components for other tabs as needed
const Personality = () => <Typography p={2}>Personality Content</Typography>;

// Tab panel component
const TabPanel = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      style={{ height: 'calc(100vh - 112px)', overflow: 'auto' }}
      {...other}
    >
      {value === index && children}
    </div>
  );
};

// Main app content with tabs
const AppContent = () => {
  const [tabValue, setTabValue] = useState(0);
  const { simulationStatus, loading, error, actions } = useAgent();

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    
    // Fetch data based on selected tab
    if (newValue === 3) { // Goals tab
      console.log("Goals tab selected, fetching goals data");
      actions.fetchGoals();
    } else if (newValue === 0) { // Memory Browser tab
      console.log("Memory Browser tab selected, fetching all memories");
      actions.fetchAllMemories();
    } else if (newValue === 1) { // Emotional State tab
      actions.fetchSummary();
    } else if (newValue === 4) { // Tool History tab
      console.log("Tool History tab selected, fetching tool history");
      actions.fetchToolHistory();
    } else if (newValue === 6) { // Journal tab
      actions.fetchJournal();
    } else if (newValue === 7) { // Thought Summaries tab
      actions.fetchThoughtSummaries();
    }
  };

  const handleStartSimulation = () => {
    actions.startSimulation({
      memory_path: "agent_memory.pkl",
      initial_tasks: ["Meditate on existence", "Explore emotional state"]
    });
  };

  const handleStopSimulation = () => {
    actions.stopSimulation();
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Agent State Viewer
          </Typography>
          <Typography variant="body2" sx={{ mr: 2 }}>
            {simulationStatus.running ? 
              `Running - Step: ${simulationStatus.step_count}` : 
              'Stopped'}
          </Typography>
          <Button 
            variant="contained" 
            color={simulationStatus.running ? "error" : "success"}
            startIcon={simulationStatus.running ? <StopIcon /> : <PlayArrowIcon />}
            onClick={simulationStatus.running ? handleStopSimulation : handleStartSimulation}
            disabled={loading}
          >
            {simulationStatus.running ? 'Stop' : 'Start'} Simulation
          </Button>
        </Toolbar>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          sx={{ bgcolor: 'primary.light' }}
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Memory Browser" />
          <Tab label="Emotional State" />
          <Tab label="Emotion Graph" />
          <Tab label="Goals" />
          <Tab label="Tool History" />
          <Tab label="Personality" />
          <Tab label="Journal" />
          <Tab label="Thought Summaries" />
        </Tabs>
      </AppBar>
      
      {error && (
        <Alert severity="error" sx={{ m: 2 }}>{error}</Alert>
      )}

      <TabPanel value={tabValue} index={0}>
        <MemoryBrowser />
      </TabPanel>
      <TabPanel value={tabValue} index={1}>
        <EmotionalState />
      </TabPanel>
      <TabPanel value={tabValue} index={2}>
        <EmotionGraph />
      </TabPanel>
      <TabPanel value={tabValue} index={3}>
        <Goals />
      </TabPanel>
      <TabPanel value={tabValue} index={4}>
        <ToolHistory />
      </TabPanel>
      <TabPanel value={tabValue} index={5}>
        <Personality />
      </TabPanel>
      <TabPanel value={tabValue} index={6}>
        <Journal />
      </TabPanel>
      <TabPanel value={tabValue} index={7}>
        <ThoughtSummaries />
      </TabPanel>
    </Box>
  );
};

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <AgentProvider>
        <AppContent />
      </AgentProvider>
    </ThemeProvider>
  );
}

export default App; 