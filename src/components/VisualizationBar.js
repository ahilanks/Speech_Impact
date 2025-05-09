import React, { useEffect, useRef, useCallback } from 'react';
import styled from 'styled-components';
import * as d3 from 'd3';

const VisualizationContainer = styled.div`
  width: 100%;
  height: 180px;
  background-color: var(--surface);
  border-radius: var(--radius-md);
  position: relative;
  overflow: hidden;
  margin-bottom: 20px;
`;

const TimelineMarker = styled.div`
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background-color: var(--secondary);
  z-index: 10;
  pointer-events: none;
  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: -8px;
    width: 18px;
    height: 18px;
    background-color: var(--secondary);
    border-radius: 50%;
  }
`;

const TimestampContainer = styled.div`
  position: absolute;
  bottom: 10px;
  left: 0;
  right: 0;
  display: flex;
  justify-content: space-between;
  padding: 0 20px;
  color: var(--text-secondary);
  font-size: 12px;
`;

const PlayButtonOverlay = styled.div`
  position: absolute;
  top: 15px;
  left: 15px;
  z-index: 20;
`;

const SpeedControlContainer = styled.div`
  position: absolute;
  top: 20px;
  left: 70px;
  z-index: 20;
`;

const SpeedToggleButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(30, 30, 30, 0.8);
  color: white;
  border: none;
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: rgba(40, 40, 40, 0.9);
  }
  
  span {
    color: ${props => props.active ? 'var(--primary)' : 'white'};
  }
`;

const PlayButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: var(--primary);
  color: white;
  border: none;
  border-radius: 50%;
  width: 42px;
  height: 42px;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0,0,0,0.3);
  
  &:hover {
    background-color: var(--primary-light);
  }
`;

const TimeDisplay = styled.div`
  position: absolute;
  top: 20px;
  right: 20px;
  font-size: 14px;
  color: var(--text-primary);
  font-variant-numeric: tabular-nums;
  background-color: rgba(30, 30, 30, 0.8);
  padding: 4px 8px;
  border-radius: 4px;
  z-index: 20;
`;

const VisualizationBar = ({ 
  clappingData, 
  currentTime,
  duration, 
  onTimeChange,
  isPlaying,
  onPlayPause,
  playbackSpeed,
  onSpeedChange 
}) => {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const markerRef = useRef(null);
  const isDraggingRef = useRef(false);

  // Define event handlers with useCallback to prevent unnecessary re-creations
  const updateTimeFromPosition = useCallback((e) => {
    if (!containerRef.current || !duration) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const newTime = percentage * duration;
    onTimeChange(newTime);
  }, [duration, onTimeChange]);

  const startDrag = useCallback((e) => {
    // Don't start drag if clicking on play button
    if (e.target.closest('button')) return;
    
    isDraggingRef.current = true;
    updateTimeFromPosition(e);
  }, [updateTimeFromPosition]);
  
  const drag = useCallback((e) => {
    if (!isDraggingRef.current) return;
    updateTimeFromPosition(e);
  }, [updateTimeFromPosition]);
  
  const endDrag = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  // Draw the visualization on mount and when data changes
  useEffect(() => {
    if (!containerRef.current || !clappingData?.length || !duration) return;
    
    const currentContainer = containerRef.current;
    const width = currentContainer.clientWidth;
    const height = currentContainer.clientHeight - 30; // Leave space for timestamps
    
    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    // Clear previous content
    svg.selectAll('*').remove();
    
    // X scale - time
    const xScale = d3.scaleLinear()
      .domain([0, duration])
      .range([0, width]);
    
    // Maximum intensity for scaling
    const maxIntensity = d3.max(clappingData, d => d.intensity);
    
    // Y scale - intensity
    const yScale = d3.scaleLinear()
      .domain([0, maxIntensity * 1.2]) // Add 20% padding at top
      .range([height, 10]);
    
    // Generate the area for each Gaussian curve
    clappingData.forEach(event => {
      const eventDuration = event.endTime - event.startTime;
      const centerTime = event.startTime + (eventDuration / 2);
      
      // Standard deviation (controls the width of the curve)
      const sigma = Math.max(eventDuration / 4, 2); // Ensure minimum width
      
      // Generate points for the Gaussian curve
      const points = [];
      const rangeStart = Math.max(0, centerTime - sigma * 3);
      const rangeEnd = Math.min(duration, centerTime + sigma * 3);
      
      for (let t = rangeStart; t <= rangeEnd; t += 0.5) {
        const gaussian = Math.exp(-0.5 * Math.pow((t - centerTime) / sigma, 2));
        const intensity = event.intensity * gaussian;
        points.push({ x: t, y: intensity });
      }
      
      // Create area path
      const area = d3.area()
        .x(d => xScale(d.x))
        .y0(height)
        .y1(d => yScale(d.y))
        .curve(d3.curveBasis);
      
      // Add area to SVG
      svg.append('path')
        .datum(points)
        .attr('fill', 'rgba(126, 87, 194, 0.6)')
        .attr('stroke', 'rgba(146, 87, 194, 0.8)')
        .attr('stroke-width', 1)
        .attr('d', area);
    });
    
    // Add baseline
    svg.append('line')
      .attr('x1', 0)
      .attr('y1', height)
      .attr('x2', width)
      .attr('y2', height)
      .attr('stroke', '#6c3f9c')
      .attr('stroke-width', 1);
    
    // Add click event to container
    currentContainer.addEventListener('mousedown', startDrag);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', endDrag);
    
    return () => {
      currentContainer.removeEventListener('mousedown', startDrag);
      document.removeEventListener('mousemove', drag);
      document.removeEventListener('mouseup', endDrag);
    };
  }, [clappingData, duration, startDrag, drag, endDrag]);
  
  // Update marker position when currentTime changes
  useEffect(() => {
    if (!containerRef.current || !markerRef.current || !duration) return;
    
    const position = (currentTime / duration) * containerRef.current.clientWidth;
    markerRef.current.style.left = `${position}px`;
  }, [currentTime, duration]);
  
  // Define available speeds
  const availableSpeeds = [0.75, 1, 1.25, 1.5, 1.75, 2];
  
  // Handle cycling through speeds
  const cycleSpeed = () => {
    const currentIndex = availableSpeeds.indexOf(playbackSpeed);
    const nextIndex = (currentIndex + 1) % availableSpeeds.length;
    onSpeedChange(availableSpeeds[nextIndex]);
  };

  // Format time as MM:SS
  const formatTime = (timeInSeconds) => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <VisualizationContainer ref={containerRef}>
      <svg ref={svgRef}></svg>
      <TimelineMarker ref={markerRef} />
      
      <PlayButtonOverlay>
        <PlayButton onClick={onPlayPause}>
          {isPlaying ? (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5,3 19,12 5,21" />
            </svg>
          )}
        </PlayButton>
      </PlayButtonOverlay>
      
      <SpeedControlContainer>
        <SpeedToggleButton 
          onClick={cycleSpeed} 
          title="Change playback speed"
        >
          <span>{playbackSpeed}x</span>
        </SpeedToggleButton>
      </SpeedControlContainer>
      
      <TimeDisplay>
        {formatTime(currentTime)} / {formatTime(duration)}
      </TimeDisplay>
      
      <TimestampContainer>
        <span>00:00</span>
        {duration && <span>{formatTime(duration / 4)}</span>}
        {duration && <span>{formatTime(duration / 2)}</span>}
        {duration && <span>{formatTime(3 * duration / 4)}</span>}
        {duration && <span>{formatTime(duration)}</span>}
      </TimestampContainer>
    </VisualizationContainer>
  );
};

export default VisualizationBar; 