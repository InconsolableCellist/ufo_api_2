# Agent Viewer Web

A modern web-based viewer for the agent simulation with 3D visualization capabilities.

## Features

- **Memory Browser**: View and search through the agent's long-term and short-term memories
- **Emotional State**: Monitor the agent's current emotional state
- **3D Emotion Visualization**: Interactive 3D visualization of emotional states using Three.js
- **Goals**: Track the agent's current goals and objectives
- **Tool History**: View a log of tools used by the agent
- **Personality**: Examine the agent's personality traits
- **Journal**: Read the agent's journal entries
- **Thought Summaries**: Review summaries of the agent's thought processes

## Getting Started

### Prerequisites

- Node.js 14.x or later
- npm 6.x or later

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd agent_viewer_web
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

   Alternatively, you can create a start script with the proper environment variables:
   ```
   # Create a start.sh file
   echo '#!/bin/bash
   export PORT=3000
   export BROWSER=none
   export DANGEROUSLY_DISABLE_HOST_CHECK=true
   export HOST=0.0.0.0
   npm start' > start.sh
   
   # Make it executable
   chmod +x start.sh
   
   # Run it
   ./start.sh
   ```

4. The application will open in your browser at http://localhost:3000

## Configuration

The application is configured to connect to the agent API at `http://bestiary:8081`. If your API server runs on a different address, update the `API_BASE_URL` in `src/api/agentApi.js`.

## Troubleshooting

If you encounter API connection issues:

1. Make sure the agent API server is running at the configured URL
2. Check that the proxy setting in `package.json` points to the correct API URL
3. If you see a 404 error for the `/memory` endpoint:
   - Verify that the API is correctly handling memory queries
   - The API may expect `memory_type` to be 'long' or 'short' instead of 'long_term' or 'short_term'

## Development

- The project uses React for the UI components
- Material-UI is used for styling and UI components
- Three.js is used for 3D visualization
- Axios is used for API communication

## API Integration

The application integrates with the following Agent API endpoints:

- `/start` - Start the simulation
- `/stop` - Stop the simulation
- `/status` - Get simulation status
- `/summary` - Get agent summary
- `/memory` - Query agent's memories
- `/thought-summaries` - Get thought summaries
- `/journal` - Get the agent's journal entries
- And more...

## CORS Configuration

Make sure the API server allows cross-origin requests from your development server. You might need to configure CORS headers on the API side if you're experiencing connection issues.

## License

[MIT License](LICENSE)

## Using the Application

### Main Features

#### Memory Browser
Navigate through the agent's long-term and short-term memories. You can search for specific memories and filter them based on various criteria.

#### Emotional State
View the agent's current emotional state with a detailed breakdown of different emotions and their intensities.

#### 3D Emotion Visualization
Interact with a 3D visualization of the agent's emotional state, where emotions are represented in a spatial arrangement based on their relationships.

#### Goals
Track the agent's goals, including:
- Long-term goals that guide overall behavior
- Short-term goals with completion status
Use the refresh button to get the latest goal updates from the agent.

#### Journal
Read the agent's journal entries chronologically. This component features:
- Full text search capabilities to find specific mentions
- Copy functionality to extract journal contents
- Auto-refreshing when the agent adds new entries

The journal component fetches the agent's journal data from the `/journal` API endpoint, which returns the journal content and last update timestamp. 