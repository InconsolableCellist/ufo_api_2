#!/bin/bash

# Set environment variables
export PORT=3000
export BROWSER=none  # Prevent auto-opening browser
export DANGEROUSLY_DISABLE_HOST_CHECK=true  # Allow connecting from other devices on the network

# Set development server to bind to all interfaces
export HOST=0.0.0.0

# Start the development server
cd "$(dirname "$0")" && npm start 