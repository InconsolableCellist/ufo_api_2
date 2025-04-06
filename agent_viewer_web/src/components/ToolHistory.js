import React, { useState, useEffect, useRef } from 'react';
import { useAgent } from '../context/AgentContext';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Button,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import HistoryIcon from '@mui/icons-material/History';

const ToolHistory = () => {
  const { toolData, loading, error, actions, simulationStatus } = useAgent();
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [selectedTool, setSelectedTool] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const tableContainerRef = useRef(null);
  const [scrollPosition, setScrollPosition] = useState(0);

  useEffect(() => {
    fetchToolHistory();
  }, []);

  useEffect(() => {
    if (tableContainerRef.current) {
      setScrollPosition(tableContainerRef.current.scrollTop);
    }
  }, [toolData]);

  useEffect(() => {
    if (tableContainerRef.current && !loading) {
      tableContainerRef.current.scrollTop = scrollPosition;
    }
  }, [scrollPosition, loading]);

  const fetchToolHistory = async () => {
    try {
      await actions.fetchToolHistory();
    } catch (err) {
      console.error('Error fetching tool history:', err);
    }
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleRowClick = (tool) => {
    setSelectedTool(tool);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'Unknown time';
    
    if (typeof timestamp === 'string') {
      // Try to parse ISO string
      try {
        const date = new Date(timestamp);
        return date.toLocaleString();
      } catch (e) {
        return timestamp;
      }
    }
    
    return timestamp;
  };

  const formatResult = (result) => {
    if (!result) return 'No result';
    
    if (typeof result === 'object') {
      try {
        // Check for success/error indicators
        if (result.success === true) {
          return `SUCCESS: ${result.output || 'No output'}`;
        } else if (result.success === false) {
          return `FAILED: ${result.error || 'Unknown error'}`;
        } else {
          return JSON.stringify(result, null, 2);
        }
      } catch (e) {
        return String(result);
      }
    }
    
    return String(result);
  };

  const formatParams = (params) => {
    if (!params) return 'No parameters';
    
    if (typeof params === 'object') {
      try {
        return JSON.stringify(params, null, 2);
      } catch (e) {
        return String(params);
      }
    }
    
    return String(params);
  };

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center' }}>
          <HistoryIcon sx={{ mr: 1 }} />
          Tool Usage History
        </Typography>
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={fetchToolHistory}
          disabled={loading}
        >
          Refresh History
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      )}

      {!simulationStatus?.running && !toolData?.length && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Simulation is not running. Start the simulation to see tool usage history, or browse cached data if available.
        </Alert>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Paper sx={{ width: '100%', overflow: 'hidden' }}>
          <TableContainer ref={tableContainerRef} sx={{ maxHeight: 440 }}>
            <Table stickyHeader aria-label="tool history table">
              <TableHead>
                <TableRow>
                  <TableCell width="10%">#</TableCell>
                  <TableCell width="20%">Time</TableCell>
                  <TableCell width="25%">Tool</TableCell>
                  <TableCell width="45%">Result</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {toolData && toolData.length > 0 ? (
                  toolData
                    .slice()
                    .reverse()
                    .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                    .map((tool, index) => (
                      <TableRow
                        hover
                        key={index}
                        onClick={() => handleRowClick(tool)}
                        sx={{ cursor: 'pointer' }}
                      >
                        <TableCell>{page * rowsPerPage + index + 1}</TableCell>
                        <TableCell>{formatTimestamp(tool.timestamp)}</TableCell>
                        <TableCell>{tool.name}</TableCell>
                        <TableCell>
                          {formatResult(tool.result).length > 50 
                            ? `${formatResult(tool.result).substring(0, 50)}...`
                            : formatResult(tool.result)
                          }
                        </TableCell>
                      </TableRow>
                    ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={4} align="center">
                      No tool usage history available. The agent may not have used any tools yet.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[10, 25, 50]}
            component="div"
            count={toolData ? toolData.length : 0}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </Paper>
      )}

      <Dialog
        open={dialogOpen}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>{selectedTool?.name || 'Tool Details'}</DialogTitle>
        <DialogContent>
          {selectedTool && (
            <Box sx={{ mt: 2 }}>
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Basic Information</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body2" component="div">
                    <strong>Tool Name:</strong> {selectedTool.name}<br />
                    <strong>Time:</strong> {formatTimestamp(selectedTool.timestamp)}<br />
                  </Typography>
                </AccordionDetails>
              </Accordion>
              
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Parameters</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography
                    variant="body2"
                    component="pre"
                    sx={{
                      whiteSpace: 'pre-wrap',
                      backgroundColor: '#f5f5f5',
                      p: 2,
                      borderRadius: 1,
                      maxHeight: '20vh',
                      overflow: 'auto'
                    }}
                  >
                    {formatParams(selectedTool.params)}
                  </Typography>
                </AccordionDetails>
              </Accordion>
              
              <Accordion defaultExpanded>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="subtitle1">Result</Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography
                    variant="body2"
                    component="pre"
                    sx={{
                      whiteSpace: 'pre-wrap',
                      backgroundColor: '#f5f5f5',
                      p: 2,
                      borderRadius: 1,
                      maxHeight: '20vh',
                      overflow: 'auto'
                    }}
                  >
                    {formatResult(selectedTool.result)}
                  </Typography>
                </AccordionDetails>
              </Accordion>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ToolHistory; 