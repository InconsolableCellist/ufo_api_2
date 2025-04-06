import React, { useState, useEffect } from 'react';
import { useAgent } from '../context/AgentContext';
import { 
  Box, 
  Paper, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  TablePagination,
  Radio,
  RadioGroup,
  FormControlLabel,
  TextField,
  Button,
  Typography,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  Alert
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import RefreshIcon from '@mui/icons-material/Refresh';

const MemoryBrowser = () => {
  const { memoryData, loading, error, actions, simulationStatus } = useAgent();
  const [memoryType, setMemoryType] = useState('long_term');
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredMemories, setFilteredMemories] = useState([]);
  const [selectedMemory, setSelectedMemory] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  useEffect(() => {
    loadAllMemories();
  }, []);

  useEffect(() => {
    // Update filtered memories when memoryData changes
    if (memoryType === 'long_term') {
      setFilteredMemories(memoryData.long_term || []);
    } else {
      setFilteredMemories(memoryData.short_term || []);
    }
  }, [memoryData, memoryType]);

  const loadAllMemories = async () => {
    try {
      await actions.fetchAllMemories();
    } catch (err) {
      console.error('Error loading all memories:', err);
    }
  };

  const loadMemories = async () => {
    try {
      await actions.queryMemory({ memory_type: memoryType });
    } catch (err) {
      console.error('Error loading memories:', err);
    }
  };

  const handleSearch = () => {
    const memories = memoryType === 'long_term' ? memoryData.long_term : memoryData.short_term;
    if (!memories) return;

    const query = searchQuery.toLowerCase();
    if (!query) {
      setFilteredMemories(memories);
      return;
    }

    const filtered = memories.filter(memory => 
      memory.toLowerCase().includes(query)
    );
    setFilteredMemories(filtered);
  };

  const handleMemoryTypeChange = (event) => {
    setMemoryType(event.target.value);
    setPage(0);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleRowClick = (memory) => {
    setSelectedMemory(memory);
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    setDialogOpen(false);
  };

  const extractTimestamp = (memory) => {
    const match = typeof memory === 'string' ? memory.match(/\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/) : null;
    return match ? match[0] : "";
  };

  const getPreview = (memory, maxLength = 50) => {
    if (typeof memory !== 'string') return String(memory);
    
    const content = memory.replace(/\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/, '').trim();
    return content.length > maxLength ? `${content.substring(0, maxLength)}...` : content;
  };

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <RadioGroup
          row
          name="memory-type"
          value={memoryType}
          onChange={handleMemoryTypeChange}
        >
          <FormControlLabel value="long_term" control={<Radio />} label="Long-term" />
          <FormControlLabel value="short_term" control={<Radio />} label="Short-term" />
        </RadioGroup>
        
        <TextField
          size="small"
          label="Search memories"
          variant="outlined"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          sx={{ mx: 2, flexGrow: 1 }}
        />
        
        <Button 
          variant="contained" 
          startIcon={<SearchIcon />}
          onClick={handleSearch}
          sx={{ mr: 1 }}
        >
          Search
        </Button>
        
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={loadAllMemories}
          disabled={loading}
        >
          Refresh All
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      )}

      {!simulationStatus?.running && (!filteredMemories || filteredMemories.length === 0) && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Simulation is not running. Start the simulation to access memories, or browse cached data if available.
        </Alert>
      )}

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Paper sx={{ width: '100%', overflow: 'hidden' }}>
          <TableContainer sx={{ maxHeight: 440 }}>
            <Table stickyHeader aria-label="memory table">
              <TableHead>
                <TableRow>
                  <TableCell>#</TableCell>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Preview</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredMemories.length > 0 ? (
                  filteredMemories
                    .slice()
                    .reverse()
                    .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                    .map((memory, index) => (
                      <TableRow 
                        hover 
                        key={index} 
                        onClick={() => handleRowClick(memory)}
                        sx={{ cursor: 'pointer' }}
                      >
                        <TableCell>{page * rowsPerPage + index + 1}</TableCell>
                        <TableCell>{extractTimestamp(memory)}</TableCell>
                        <TableCell>{getPreview(memory)}</TableCell>
                      </TableRow>
                    ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={3} align="center">
                      No memories found. Try initializing the agent or changing memory type.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[10, 25, 50]}
            component="div"
            count={filteredMemories.length}
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
        <DialogTitle>Memory Details</DialogTitle>
        <DialogContent>
          <Typography variant="body1" component="pre" sx={{ 
            whiteSpace: 'pre-wrap',
            backgroundColor: '#f5f5f5',
            p: 2,
            borderRadius: 1,
            maxHeight: '60vh',
            overflow: 'auto'
          }}>
            {selectedMemory}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MemoryBrowser; 