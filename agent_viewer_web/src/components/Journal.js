import React, { useState, useEffect, useRef } from 'react';
import { useAgent } from '../context/AgentContext';
import agentApi from '../api/agentApi';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Button,
  Alert,
  TextField,
  IconButton,
  Divider
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import SearchIcon from '@mui/icons-material/Search';

const Journal = () => {
  const { journalData, loading: contextLoading, error, actions, simulationStatus } = useAgent();
  const [searchTerm, setSearchTerm] = useState('');
  const [displayedJournal, setDisplayedJournal] = useState('');
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const paperRef = useRef(null);
  const [scrollPosition, setScrollPosition] = useState(0);

  useEffect(() => {
    fetchJournal();
  }, []);

  // Save scroll position before updating content
  useEffect(() => {
    if (paperRef.current) {
      setScrollPosition(paperRef.current.scrollTop);
    }
  }, [journalData]);

  // Restore scroll position after content update
  useEffect(() => {
    if (paperRef.current && displayedJournal) {
      paperRef.current.scrollTop = scrollPosition;
    }
  }, [displayedJournal, scrollPosition]);

  useEffect(() => {
    // Update displayed journal when journalData changes or search term changes
    if (journalData) {
      if (searchTerm) {
        try {
          const searchRegex = new RegExp(searchTerm, 'gi');
          const highlighted = journalData.replace(
            searchRegex,
            match => `<mark style="background-color: #ffeb3b;">${match}</mark>`
          );
          setDisplayedJournal(highlighted);
        } catch (error) {
          // Handle invalid regex
          setLocalError(`Invalid search pattern: ${error.message}`);
          setDisplayedJournal(journalData);
        }
      } else {
        setDisplayedJournal(journalData);
        setLocalError(null);
      }
    }
  }, [journalData, searchTerm]);

  const fetchJournal = async () => {
    setLoading(true);
    setLocalError(null);
    try {
      // Use the context function to fetch journal data
      const response = await actions.fetchJournal();
      
      // Update last updated timestamp if available
      if (response && response.last_update) {
        setLastUpdated(new Date(response.last_update));
      }
    } catch (err) {
      console.error('Error fetching journal:', err);
      const errorMessage = err.response?.status === 404
        ? 'Journal file not found. Make sure the agent has created a journal.'
        : `Error loading journal: ${err.message}`;
      
      setLocalError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyToClipboard = () => {
    if (displayedJournal) {
      // Copy the plain text version without HTML markup
      const plainText = displayedJournal.replace(/<[^>]*>/g, '');
      navigator.clipboard.writeText(plainText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleSearch = () => {
    // Search is handled in the useEffect already
    setLocalError(null);
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const isLoading = loading || contextLoading;
  const displayError = localError || error;

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h5">Agent Journal</Typography>
          {lastUpdated && (
            <Typography variant="caption" color="text.secondary">
              Last updated: {formatTimestamp(lastUpdated)}
            </Typography>
          )}
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchJournal}
            disabled={isLoading}
          >
            Refresh
          </Button>
          <Button
            variant="outlined"
            startIcon={<ContentCopyIcon />}
            onClick={handleCopyToClipboard}
            disabled={!displayedJournal || isLoading}
            color={copied ? "success" : "primary"}
          >
            {copied ? "Copied!" : "Copy All"}
          </Button>
        </Box>
      </Box>

      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <TextField
          fullWidth
          size="small"
          label="Search journal"
          variant="outlined"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ mr: 1 }}
        />
        <IconButton onClick={handleSearch} color="primary">
          <SearchIcon />
        </IconButton>
      </Box>

      {displayError && (
        <Alert severity="error" sx={{ mb: 2 }}>{displayError}</Alert>
      )}

      {!simulationStatus?.running && !displayedJournal && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Simulation is not running. Start the simulation to access the journal, or browse cached journal entries if available.
        </Alert>
      )}

      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Paper 
          ref={paperRef}
          elevation={3} 
          sx={{ 
            p: 2, 
            maxHeight: 'calc(100vh - 240px)', 
            overflow: 'auto',
            bgcolor: '#f8f9fa',
            fontFamily: '"Courier New", monospace'
          }}
        >
          {displayedJournal ? (
            <Typography 
              component="div" 
              sx={{ whiteSpace: 'pre-wrap' }}
              dangerouslySetInnerHTML={{ __html: displayedJournal }}
            />
          ) : (
            <Typography color="text.secondary" sx={{ fontStyle: 'italic' }}>
              No journal entries available. The agent may not have written in their journal yet.
            </Typography>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default Journal; 