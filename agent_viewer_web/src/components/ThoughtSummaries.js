import React, { useState, useEffect } from 'react';
import { useAgent } from '../context/AgentContext';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  CardHeader,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Button,
  Pagination,
  CircularProgress,
  Chip,
  Alert,
  Stack,
  Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import RefreshIcon from '@mui/icons-material/Refresh';

const ThoughtSummaries = () => {
  const { thoughtSummaries, summaryStatus, loading, error, actions } = useAgent();
  const [page, setPage] = useState(1);
  const [limit] = useState(5);
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    loadSummaries();
    loadStatus();
  }, [page]);

  const loadSummaries = async () => {
    await actions.fetchThoughtSummaries(limit, (page - 1) * limit);
  };

  const loadStatus = async () => {
    await actions.fetchThoughtSummaryStatus();
  };

  const handleStartSummarization = async () => {
    await actions.startThoughtSummarization();
    loadStatus();
  };

  const handleStopSummarization = async () => {
    await actions.stopThoughtSummarization();
    loadStatus();
  };

  const handleRefresh = () => {
    loadSummaries();
    loadStatus();
  };

  const handleExpandChange = (summaryId) => {
    setExpanded(expanded === summaryId ? null : summaryId);
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Thought Summaries</Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="contained"
            color={summaryStatus.active ? 'error' : 'success'}
            startIcon={summaryStatus.active ? <StopIcon /> : <PlayArrowIcon />}
            onClick={summaryStatus.active ? handleStopSummarization : handleStartSummarization}
            disabled={loading}
          >
            {summaryStatus.active ? 'Stop Summarization' : 'Start Summarization'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
        </Stack>
      </Box>

      <Card sx={{ mb: 3 }}>
        <CardHeader title="Summarization Status" />
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <Chip 
              label={`API Available: ${summaryStatus.api_available ? 'Yes' : 'No'}`} 
              color={summaryStatus.api_available ? 'success' : 'error'} 
            />
            <Chip 
              label={`Active: ${summaryStatus.active ? 'Yes' : 'No'}`} 
              color={summaryStatus.active ? 'success' : 'warning'} 
            />
            <Chip label={`Queue Size: ${summaryStatus.queue_size || 0}`} color="primary" />
            <Chip label={`Total Entries: ${summaryStatus.total_entries || 0}`} color="primary" />
            <Chip label={`Summarized: ${summaryStatus.summarized_entries || 0}`} color="primary" />
          </Box>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : thoughtSummaries.length === 0 ? (
        <Alert severity="info">No thought summaries available</Alert>
      ) : (
        <>
          {thoughtSummaries.map((summary, index) => (
            <Accordion 
              key={index}
              expanded={expanded === index}
              onChange={() => handleExpandChange(index)}
              sx={{ mb: 2 }}
            >
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                  <Typography variant="subtitle1">
                    {summary.title || `Thought Summary #${index + 1}`}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                    {formatDate(summary.timestamp)}
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Box>
                  {summary.tags && summary.tags.length > 0 && (
                    <Box sx={{ mb: 2 }}>
                      {summary.tags.map((tag, i) => (
                        <Chip key={i} label={tag} size="small" sx={{ mr: 1 }} />
                      ))}
                    </Box>
                  )}
                  
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    {summary.summary || 'No summary content available'}
                  </Typography>
                  
                  {summary.original_thoughts && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>Original Thoughts:</Typography>
                      <Paper variant="outlined" sx={{ p: 2, backgroundColor: '#f5f5f5' }}>
                        <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>
                          {summary.original_thoughts.join('\n\n')}
                        </pre>
                      </Paper>
                    </Box>
                  )}
                  
                  {summary.metadata && (
                    <Box sx={{ mt: 2 }}>
                      <Divider sx={{ my: 2 }} />
                      <Typography variant="subtitle2">Metadata:</Typography>
                      <Box component="pre" sx={{ 
                        p: 1, 
                        backgroundColor: '#f0f0f0', 
                        borderRadius: 1,
                        fontSize: '0.8rem',
                        overflowX: 'auto'
                      }}>
                        {JSON.stringify(summary.metadata, null, 2)}
                      </Box>
                    </Box>
                  )}
                </Box>
              </AccordionDetails>
            </Accordion>
          ))}
          
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Pagination 
              count={Math.ceil((summaryStatus.total_entries || 0) / limit)} 
              page={page}
              onChange={(e, newPage) => setPage(newPage)}
              color="primary"
            />
          </Box>
        </>
      )}
    </Box>
  );
};

export default ThoughtSummaries; 