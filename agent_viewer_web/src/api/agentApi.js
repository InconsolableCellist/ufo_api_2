import axios from 'axios';

// Using relative paths with proxy configuration in package.json
const agentApi = {
  // Simulation control
  startSimulation: (config) => axios.post('/start', config),
  stopSimulation: () => axios.post('/stop'),
  getStatus: () => axios.get('/status'),
  nextStep: () => axios.post('/next'),
  getSummary: () => axios.get('/summary'),
  
  // Memory operations
  queryMemory: (queryParams) => {
    // Extract memory_type parameter and format it correctly
    const { memory_type, ...otherParams } = queryParams;
    
    // API uses 'long' or 'short' in the memory_type field
    const formattedParams = {
      ...otherParams,
      memory_type: memory_type === 'long_term' ? 'long' : 'short'
    };
    
    return axios.post('/memory', formattedParams);
  },
  getAllMemories: () => axios.get('/memory/all'),
  queryMemoryByType: (queryParams) => axios.post('/memory/by-type', queryParams),
  queryMemoryByTime: (queryParams) => axios.post('/memory/by-time', queryParams),
  getMemoryStats: () => axios.get('/memory/stats'),
  
  // Journal operations
  getJournal: () => axios.get('/journal'),
  
  // Goals operations
  getGoals: () => axios.get('/goals'),
  
  // Tool operations
  getToolHistory: () => axios.get('/tools/history'),
  
  // Emotion operations
  adjustEmotion: (emotionName, change) => 
    axios.post(`/emotion/adjust?emotion_name=${emotionName}&change=${change}`),
  
  // Thought summary operations
  getThoughtSummaries: (limit = 10, offset = 0) => 
    axios.get(`/thought-summaries?limit=${limit}&offset=${offset}`),
  startThoughtSummarization: () => axios.post('/thought-summaries/start'),
  stopThoughtSummarization: () => axios.post('/thought-summaries/stop'),
  getThoughtSummarizationStatus: () => axios.get('/thought-summaries/status'),
  forceSummarizeAllThoughts: () => axios.post('/thought-summaries/force-summarize'),
  
  // Stimuli operations
  addStimuli: (stimuli) => axios.post('/stimuli', stimuli),
};

export default agentApi; 