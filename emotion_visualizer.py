#!/usr/bin/env python3
import pickle
import json
import http.server
import socketserver
import webbrowser
import os
import time
from threading import Thread

# File paths
MEMORY_PATH = "agent_memory.pkl"
JSON_OUTPUT_DIR = "web_visualizer"
JSON_DATA_PATH = os.path.join(JSON_OUTPUT_DIR, "memory_data.json")
HTML_PATH = os.path.join(JSON_OUTPUT_DIR, "index.html")

def create_web_directory():
    """Create directory for web files if it doesn't exist"""
    if not os.path.exists(JSON_OUTPUT_DIR):
        os.makedirs(JSON_OUTPUT_DIR)
        print(f"Created directory: {JSON_OUTPUT_DIR}")

def convert_memory_to_json():
    """Convert the memory pickle file to JSON format"""
    print(f"Loading memory data from {MEMORY_PATH}...")
    
    try:
        with open(MEMORY_PATH, 'rb') as f:
            memory_data = pickle.load(f)
            
        # Extract relevant data
        json_data = {
            'memories': [],
            'emotions': {}
        }
        
        # Process long-term memories
        print("Processing long-term memories...")
        for i, memory in enumerate(memory_data.get('long_term', [])):
            memory_obj = {
                'id': i,
                'content': str(memory),
                'emotions': {}
            }
            
            # Add emotional associations if available
            if 'associations' in memory_data and memory in memory_data['associations']:
                for emotion, value in memory_data['associations'][memory].items():
                    if isinstance(value, (int, float)):
                        memory_obj['emotions'][emotion] = value
                        
                        # Track all emotions for the legend
                        if emotion not in json_data['emotions']:
                            json_data['emotions'][emotion] = []
                        json_data['emotions'][emotion].append({'memory_id': i, 'value': value})
            
            json_data['memories'].append(memory_obj)
            
        # Write the JSON file
        with open(JSON_DATA_PATH, 'w') as f:
            json.dump(json_data, f)
            
        print(f"Converted {len(json_data['memories'])} memories to JSON")
        print(f"Found {len(json_data['emotions'])} different emotions")
        return True
    
    except Exception as e:
        print(f"Error converting memory data: {e}")
        return False

def create_html_file():
    """Create the HTML file with Three.js visualization"""
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Memory 3D Visualizer</title>
    <style>
        body { 
            margin: 0; 
            overflow: hidden; 
            font-family: Arial, sans-serif;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            max-width: 300px;
            z-index: 100;
        }
        #legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            z-index: 100;
        }
        #memoryDetail {
            position: absolute;
            bottom: 10px;
            left: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            max-height: 200px;
            overflow-y: auto;
            display: none;
            z-index: 100;
        }
        .color-box {
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 5px;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            border-radius: 5px;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div id="loading">Loading memory data...</div>
    <div id="info">
        <h2>Agent Memory Visualizer</h2>
        <p>Each sphere represents a memory with emotional associations.</p>
        <p>Colors represent different emotions.</p>
        <p>Size represents emotional intensity.</p>
        <p>Click on a sphere to view memory details.</p>
        <p>Use mouse to rotate, scroll to zoom, right-click to pan.</p>
    </div>
    <div id="legend"></div>
    <div id="memoryDetail"></div>
    
    <!-- Import Three.js from CDN -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Main visualization script
        let scene, camera, renderer, controls;
        let memoryPoints = [];
        let memoryData = null;
        let colorMap = {};
        
        // Initialize the scene
        function init() {
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x111111);
            
            // Create camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 50;
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            
            // Add controls for camera movement
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);
            
            // Handle window resize
            window.addEventListener('resize', onWindowResize);
            
            // Load data and create visualization
            loadData();
        }
        
        // Load JSON data
        function loadData() {
            fetch('memory_data.json')
                .then(response => response.json())
                .then(data => {
                    memoryData = data;
                    createVisualization(data);
                    document.getElementById('loading').style.display = 'none';
                })
                .catch(error => {
                    console.error('Error loading data:', error);
                    document.getElementById('loading').textContent = 'Error loading data. Please check console.';
                });
        }
        
        // Create the 3D visualization
        function createVisualization(data) {
            // Create color mapping for emotions
            const emotions = Object.keys(data.emotions);
            const colorScale = generateColorScale(emotions.length);
            
            emotions.forEach((emotion, index) => {
                colorMap[emotion] = colorScale[index];
            });
            
            // Create legend
            createLegend(emotions);
            
            // Create memory points
            data.memories.forEach(memory => {
                if (Object.keys(memory.emotions).length > 0) {
                    createMemoryPoint(memory, data.memories.length);
                }
            });
            
            // Start animation loop
            animate();
        }
        
        // Create a memory point (sphere)
        function createMemoryPoint(memory, totalMemories) {
            // Calculate position based on emotional values
            const emotions = Object.entries(memory.emotions);
            if (emotions.length === 0) return;
            
            // Calculate total emotional intensity
            let totalIntensity = 0;
            emotions.forEach(([emotion, value]) => {
                totalIntensity += value;
            });
            
            // Determine primary emotion (highest value)
            let primaryEmotion = emotions.reduce((max, current) => 
                current[1] > max[1] ? current : max, ['', 0]);
            
            // Create sphere geometry (size based on total emotional intensity)
            const radius = 0.5 + (totalIntensity * 2);
            const geometry = new THREE.SphereGeometry(radius, 32, 32);
            
            // Create material with color based on primary emotion
            const color = colorMap[primaryEmotion[0]] || 0xffffff;
            const material = new THREE.MeshPhongMaterial({ 
                color: color,
                transparent: true,
                opacity: 0.8,
                emissive: color,
                emissiveIntensity: 0.2
            });
            
            // Create mesh
            const sphere = new THREE.Mesh(geometry, material);
            
            // Position the sphere in 3D space
            // Use first two emotions for X and Y, or random if not enough emotions
            let x = 0, y = 0, z = 0;
            
            if (emotions.length >= 1) {
                const emotionX = emotions[0][1];
                x = (emotionX * 40) - 20; // Scale and center
            }
            
            if (emotions.length >= 2) {
                const emotionY = emotions[1][1];
                y = (emotionY * 40) - 20; // Scale and center
            }
            
            // Use memory ID for Z-axis (chronological ordering)
            z = (memory.id * 0.2) - (totalMemories * 0.1);
            
            sphere.position.set(x, y, z);
            
            // Store memory data with the sphere
            sphere.userData = {
                memoryId: memory.id,
                content: memory.content,
                emotions: memory.emotions
            };
            
            // Add to scene and tracking array
            scene.add(sphere);
            memoryPoints.push(sphere);
        }
        
        // Generate an array of distinct colors
        function generateColorScale(count) {
            const colors = [];
            for (let i = 0; i < count; i++) {
                const hue = i / count;
                const color = new THREE.Color().setHSL(hue, 1, 0.5);
                colors.push(color.getHex());
            }
            return colors;
        }
        
        // Create legend for emotion colors
        function createLegend(emotions) {
            const legend = document.getElementById('legend');
            legend.innerHTML = '<h3>Emotions</h3>';
            
            emotions.forEach(emotion => {
                const color = colorMap[emotion];
                const colorHex = '#' + color.toString(16).padStart(6, '0');
                
                const item = document.createElement('div');
                item.innerHTML = `<span class="color-box" style="background-color: ${colorHex}"></span>${emotion}`;
                legend.appendChild(item);
            });
        }
        
        // Handle window resize
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        
        // Raycaster for detecting clicks on memory points
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        // Handle mouse clicks
        function onMouseClick(event) {
            // Calculate mouse position in normalized device coordinates
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            
            // Update the raycaster with the camera and mouse position
            raycaster.setFromCamera(mouse, camera);
            
            // Check for intersections with memory points
            const intersects = raycaster.intersectObjects(memoryPoints);
            
            if (intersects.length > 0) {
                const memory = intersects[0].object.userData;
                showMemoryDetail(memory);
            } else {
                hideMemoryDetail();
            }
        }
        
        // Show memory details
        function showMemoryDetail(memory) {
            const detailElement = document.getElementById('memoryDetail');
            
            // Format emotions
            let emotionsHtml = '';
            for (const [emotion, value] of Object.entries(memory.emotions)) {
                const colorHex = '#' + colorMap[emotion].toString(16).padStart(6, '0');
                emotionsHtml += `<span style="color: ${colorHex}"><b>${emotion}:</b> ${value.toFixed(2)}</span><br>`;
            }
            
            detailElement.innerHTML = `
                <h3>Memory #${memory.memoryId}</h3>
                <div style="margin-bottom: 10px;">
                    <h4>Emotional Context:</h4>
                    ${emotionsHtml}
                </div>
                <div>
                    <h4>Content:</h4>
                    <p>${memory.content}</p>
                </div>
            `;
            
            detailElement.style.display = 'block';
        }
        
        // Hide memory details
        function hideMemoryDetail() {
            document.getElementById('memoryDetail').style.display = 'none';
        }
        
        // Add click event listener
        window.addEventListener('click', onMouseClick);
        
        // Initialize the application
        init();
    </script>
</body>
</html>
'''
    
    try:
        with open(HTML_PATH, 'w') as f:
            f.write(html_content)
        print(f"Created HTML file: {HTML_PATH}")
        return True
    except Exception as e:
        print(f"Error creating HTML file: {e}")
        return False

class MemoryVisualizer:
    def __init__(self, port=8081):
        self.port = port
        self.server = None
        self.httpd = None
        
    def start_server(self):
        """Start the HTTP server to serve the visualization"""
        try:
            # Change directory to the web files
            os.chdir(JSON_OUTPUT_DIR)
            
            # Create server
            handler = http.server.SimpleHTTPRequestHandler
            self.httpd = socketserver.TCPServer(("", self.port), handler)
            
            print(f"Starting server at http://localhost:{self.port}")
            print("Press Ctrl+C to stop the server")
            
            # Start server in a thread
            self.server = Thread(target=self.httpd.serve_forever)
            self.server.daemon = True
            self.server.start()
            
            return True
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
            
    def stop_server(self):
        """Stop the HTTP server"""
        if self.httpd:
            self.httpd.shutdown()
            print("Server stopped")

def main():
    # Create directory structure
    create_web_directory()
    
    # Convert memory data to JSON
    if not convert_memory_to_json():
        print("Failed to convert memory data to JSON. Exiting.")
        return
    
    # Create HTML file
    if not create_html_file():
        print("Failed to create HTML file. Exiting.")
        return
    
    # Start HTTP server
    visualizer = MemoryVisualizer()
    if not visualizer.start_server():
        print("Failed to start server. Exiting.")
        return
    
    # Open browser
    print("Opening browser...")
    webbrowser.open(f"http://localhost:8081")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        visualizer.stop_server()
        print("Exiting.")

if __name__ == "__main__":
    main() 