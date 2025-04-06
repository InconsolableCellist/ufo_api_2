import React, { createContext, useContext, useState, useEffect } from 'react';
import agentApi from '../api/agentApi';

const AgentContext = createContext();

export const useAgent = () => useContext(AgentContext);

export const AgentProvider = ({ children }) => {
  const [simulationStatus, setSimulationStatus] = useState({ running: false, step_count: 0 });
  const [memoryData, setMemoryData] = useState({ long_term: [], short_term: [], associations: {} });
  const [emotionData, setEmotionData] = useState({});
  const [goalData, setGoalData] = useState({ short_term_goals: [], long_term_goal: null });
  const [toolData, setToolData] = useState([]);
  const [personalityData, setPersonalityData] = useState({ personality_traits: [] });
  const [journalData, setJournalData] = useState('');
  const [thoughtSummaries, setThoughtSummaries] = useState([]);
  const [summaryStatus, setSummaryStatus] = useState({ active: false, api_available: false });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Helper function to handle API errors
  const handleApiError = (err, defaultMessage) => {
    let errorMessage = defaultMessage;
    
    if (err.response) {
      // The request was made and the server responded with a status code outside of 2xx
      if (err.response.status === 404) {
        errorMessage = `API endpoint not found. Make sure the agent API is running at the configured URL. (${err.response.status})`;
      } else if (err.response.data && err.response.data.detail) {
        errorMessage = `API Error: ${err.response.data.detail} (${err.response.status})`;
      } else {
        errorMessage = `API Error: ${err.response.statusText} (${err.response.status})`;
      }
    } else if (err.request) {
      // The request was made but no response was received
      errorMessage = 'No response received from the API. Make sure the agent API is running and accessible.';
    }
    
    setError(errorMessage);
    console.error(errorMessage, err);
    return null;
  };

  // Fetch simulation status at regular interval
  useEffect(() => {
    let statusInterval;
    
    if (simulationStatus.running) {
      statusInterval = setInterval(async () => {
        try {
          const response = await agentApi.getStatus();
          setSimulationStatus(response.data);
          
          // Also update summary and goals
          fetchSummary();
          fetchGoals();
        } catch (err) {
          console.error('Error fetching status:', err);
        }
      }, 5000); // Every 5 seconds
    }
    
    return () => {
      if (statusInterval) clearInterval(statusInterval);
    };
  }, [simulationStatus.running]);

  // Start simulation
  const startSimulation = async (config = {}) => {
    setLoading(true);
    try {
      const response = await agentApi.startSimulation(config);
      setSimulationStatus(response.data);
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to start simulation');
    } finally {
      setLoading(false);
    }
  };

  // Stop simulation
  const stopSimulation = async () => {
    setLoading(true);
    try {
      const response = await agentApi.stopSimulation();
      setSimulationStatus(response.data);
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to stop simulation');
    } finally {
      setLoading(false);
    }
  };

  // Fetch summary data
  const fetchSummary = async () => {
    setLoading(true);
    try {
      const response = await agentApi.getSummary();
      const data = response.data;
      
      // Update relevant state
      setEmotionData(data.emotional_state || {});
      
      // Only update goals from summary if we don't have dedicated goals data yet
      if ((!goalData.short_term_goals || goalData.short_term_goals.length === 0) &&
          (!goalData.long_term_goal)) {
        setGoalData({
          short_term_goals: data.short_term_goals || [],
          long_term_goal: data.long_term_goal || null
        });
      }
      
      if (data.recent_thoughts) {
        setMemoryData(prev => ({ ...prev, short_term: data.recent_thoughts }));
      }
      
      setError(null);
      return data;
    } catch (err) {
      return handleApiError(err, 'Failed to fetch summary');
    } finally {
      setLoading(false);
    }
  };

  // Query memories
  const queryMemory = async (params) => {
    setLoading(true);
    try {
      const response = await agentApi.queryMemory(params);
      if (params.memory_type === 'short') {
        setMemoryData(prev => ({ ...prev, short_term: response.data.memories }));
      } else {
        setMemoryData(prev => ({ ...prev, long_term: response.data.memories }));
      }
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to query memory');
    } finally {
      setLoading(false);
    }
  };

  // Get thought summaries
  const fetchThoughtSummaries = async (limit = 10, offset = 0) => {
    setLoading(true);
    try {
      const response = await agentApi.getThoughtSummaries(limit, offset);
      setThoughtSummaries(response.data.summaries);
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to fetch thought summaries');
    } finally {
      setLoading(false);
    }
  };

  // Get goals
  const fetchGoals = async () => {
    setLoading(true);
    try {
      const response = await agentApi.getGoals();
      setGoalData({
        short_term_goals: response.data.short_term_goals || [],
        long_term_goal: response.data.long_term_goal || null
      });
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to fetch goals');
    } finally {
      setLoading(false);
    }
  };

  // Get journal
  const fetchJournal = async () => {
    setLoading(true);
    try {
      const response = await agentApi.getJournal();
      if (response.data && response.data.content) {
        setJournalData(response.data.content);
      }
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to fetch journal');
    } finally {
      setLoading(false);
    }
  };

  // Get all memories including past ones
  const fetchAllMemories = async () => {
    setLoading(true);
    try {
      const response = await agentApi.getAllMemories();
      setMemoryData({
        short_term: response.data.short_term || [],
        long_term: response.data.long_term || [],
        associations: memoryData.associations
      });
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to fetch all memories');
    } finally {
      setLoading(false);
    }
  };

  // Get tool history
  const fetchToolHistory = async () => {
    setLoading(true);
    try {
      const response = await agentApi.getToolHistory();
      setToolData(response.data.tool_history || []);
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to fetch tool history');
    } finally {
      setLoading(false);
    }
  };

  // Get thought summary status
  const fetchThoughtSummaryStatus = async () => {
    try {
      const response = await agentApi.getThoughtSummarizationStatus();
      setSummaryStatus(response.data);
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to fetch thought summary status');
    }
  };

  // Start thought summarization
  const startThoughtSummarization = async () => {
    setLoading(true);
    try {
      const response = await agentApi.startThoughtSummarization();
      setSummaryStatus(response.data);
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to start thought summarization');
    } finally {
      setLoading(false);
    }
  };

  // Stop thought summarization
  const stopThoughtSummarization = async () => {
    setLoading(true);
    try {
      const response = await agentApi.stopThoughtSummarization();
      setSummaryStatus(response.data);
      setError(null);
      return response.data;
    } catch (err) {
      return handleApiError(err, 'Failed to stop thought summarization');
    } finally {
      setLoading(false);
    }
  };

  // Update journal data
  const updateJournalData = (data) => {
    setJournalData(data);
  };
  
  const value = {
    simulationStatus,
    memoryData,
    emotionData,
    goalData,
    toolData,
    personalityData,
    journalData,
    thoughtSummaries,
    summaryStatus,
    loading,
    error,
    actions: {
      startSimulation,
      stopSimulation,
      fetchSummary,
      queryMemory,
      fetchThoughtSummaries,
      fetchThoughtSummaryStatus,
      startThoughtSummarization,
      stopThoughtSummarization,
      updateJournalData,
      fetchGoals,
      fetchJournal,
      fetchAllMemories,
      fetchToolHistory,
    }
  };

  return (
    <AgentContext.Provider value={value}>
      {children}
    </AgentContext.Provider>
  );
};

export default AgentContext; 