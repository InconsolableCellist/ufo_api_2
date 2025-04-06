import React, { useRef, useEffect, useState } from 'react';
import { useAgent } from '../context/AgentContext';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Box, Typography, CircularProgress, Button, Slider, FormControlLabel, Switch, Alert } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';

const EmotionGraph = () => {
  const { emotionData, loading, error, actions } = useAgent();
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const pointsRef = useRef({});
  const frameIdRef = useRef(null);
  
  const [rotationSpeed, setRotationSpeed] = useState(0.001);
  const [autoRotate, setAutoRotate] = useState(true);
  const [showLabels, setShowLabels] = useState(true);

  useEffect(() => {
    // Load initial data
    fetchEmotionData();
    
    return () => {
      // Cleanup on unmount
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);
      }
      
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
      
      if (sceneRef.current) {
        // Remove all objects from the scene
        while (sceneRef.current.children.length > 0) {
          const object = sceneRef.current.children[0];
          sceneRef.current.remove(object);
        }
      }
    };
  }, []);

  useEffect(() => {
    // Initialize Three.js when the container is ready and not in loading state
    if (containerRef.current && !loading && !rendererRef.current) {
      initThree();
    }
  }, [containerRef.current, loading]);

  useEffect(() => {
    if (sceneRef.current && emotionData && Object.keys(emotionData).length > 0) {
      updateEmotionVisualization();
    }
  }, [emotionData, showLabels]);

  const fetchEmotionData = async () => {
    try {
      await actions.fetchSummary();
    } catch (err) {
      console.error('Error fetching emotion data:', err);
    }
  };

  const initThree = () => {
    if (!containerRef.current) return;
    
    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;
    
    // Create camera
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 5;
    cameraRef.current = camera;
    
    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Create controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;
    
    // Add axes for reference
    const axesHelper = new THREE.AxesHelper(3);
    scene.add(axesHelper);
    
    // Add central sphere (representing the agent)
    const sphereGeometry = new THREE.SphereGeometry(0.3, 32, 32);
    const sphereMaterial = new THREE.MeshLambertMaterial({ color: 0x4444ff });
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    scene.add(sphere);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current) return;
      
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Animation loop
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);
      
      if (autoRotate) {
        scene.rotation.y += rotationSpeed;
      }
      
      controls.update();
      renderer.render(scene, camera);
    };
    
    animate();
  };

  const updateEmotionVisualization = () => {
    if (!sceneRef.current) return;
    
    // Remove old emotion points
    Object.keys(pointsRef.current).forEach(emotionName => {
      const { point, line, label } = pointsRef.current[emotionName];
      if (point) sceneRef.current.remove(point);
      if (line) sceneRef.current.remove(line);
      if (label) sceneRef.current.remove(label);
    });
    
    pointsRef.current = {};
    
    // Create new emotion points
    const emotions = Object.entries(emotionData).filter(([name, value]) => 
      typeof value === 'number' && name !== 'mood'
    );
    const emotionCount = emotions.length;
    
    if (emotionCount === 0) return;
    
    emotions.forEach(([emotionName, value], index) => {
      // Calculate position on a sphere
      const phi = Math.acos(-1 + (2 * index) / emotionCount);
      const theta = Math.sqrt(emotionCount * Math.PI) * phi;
      
      // Base distance
      const baseDistance = 2;
      // Intensity affects the distance from center
      const distance = baseDistance * (0.5 + value * 0.5);
      
      const x = distance * Math.sin(phi) * Math.cos(theta);
      const y = distance * Math.sin(phi) * Math.sin(theta);
      const z = distance * Math.cos(phi);
      
      // Create point
      const geometry = new THREE.SphereGeometry(0.1, 16, 16);
      
      // Use emotion value to determine color (red for high, blue for low)
      const color = new THREE.Color().setHSL(
        (1 - value) * 0.6, // hue (red to blue)
        0.8, // saturation
        0.5  // lightness
      );
      
      const material = new THREE.MeshLambertMaterial({ color });
      const point = new THREE.Mesh(geometry, material);
      point.position.set(x, y, z);
      sceneRef.current.add(point);
      
      // Create line connecting to center
      const lineGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(x, y, z)
      ]);
      const lineMaterial = new THREE.LineBasicMaterial({ 
        color: color,
        transparent: true,
        opacity: 0.5
      });
      const line = new THREE.Line(lineGeometry, lineMaterial);
      sceneRef.current.add(line);
      
      // Create text label
      if (showLabels) {
        try {
          const canvas = document.createElement('canvas');
          const context = canvas.getContext('2d');
          canvas.width = 256;
          canvas.height = 128;
          
          context.fillStyle = 'rgba(255, 255, 255, 0.8)';
          context.fillRect(0, 0, canvas.width, canvas.height);
          
          context.font = '24px Arial';
          context.fillStyle = '#000000';
          context.textAlign = 'center';
          context.textBaseline = 'middle';
          context.fillText(emotionName, canvas.width / 2, 40);
          
          context.font = '20px Arial';
          context.fillText(value.toFixed(2), canvas.width / 2, 70);
          
          const texture = new THREE.CanvasTexture(canvas);
          const labelMaterial = new THREE.SpriteMaterial({ map: texture });
          const label = new THREE.Sprite(labelMaterial);
          
          // Position label slightly offset from the point
          label.position.set(x * 1.2, y * 1.2, z * 1.2);
          label.scale.set(1, 0.5, 1);
          
          sceneRef.current.add(label);
          
          // Store reference
          pointsRef.current[emotionName] = { point, line, label };
        } catch (err) {
          console.error('Error creating label:', err);
          // Store reference without label
          pointsRef.current[emotionName] = { point, line, label: null };
        }
      } else {
        // Store reference without label
        pointsRef.current[emotionName] = { point, line, label: null };
      }
    });
  };

  const handleRefresh = async () => {
    try {
      await actions.fetchSummary();
    } catch (err) {
      console.error('Error refreshing emotion data:', err);
    }
  };

  return (
    <Box sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Emotion Visualization</Typography>
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />} 
          onClick={handleRefresh}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
      )}
      
      <Box sx={{ display: 'flex', mb: 2, gap: 4 }}>
        <Box sx={{ width: 300 }}>
          <Typography>Rotation Speed</Typography>
          <Slider
            value={rotationSpeed * 1000}
            min={0}
            max={10}
            step={0.1}
            onChange={(_, value) => setRotationSpeed(value / 1000)}
            disabled={!autoRotate}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value / 10}`}
          />
        </Box>
        
        <FormControlLabel
          control={
            <Switch
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
            />
          }
          label="Auto Rotate"
        />
        
        <FormControlLabel
          control={
            <Switch
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
            />
          }
          label="Show Labels"
        />
      </Box>
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexGrow: 1 }}>
          <CircularProgress />
        </Box>
      ) : (!emotionData || Object.keys(emotionData).filter(k => typeof emotionData[k] === 'number').length === 0) ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flexGrow: 1 }}>
          <Typography color="text.secondary">No emotion data available. Start the agent simulation first.</Typography>
        </Box>
      ) : (
        <Box 
          ref={containerRef} 
          sx={{ 
            flexGrow: 1, 
            width: '100%', 
            borderRadius: 1,
            overflow: 'hidden',
            border: '1px solid #ddd'
          }}
        />
      )}
    </Box>
  );
};

export default EmotionGraph; 