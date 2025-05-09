import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import VisualizationBar from './components/VisualizationBar';
import TranscriptionList from './components/TranscriptionList';
import { clappingData } from './data/clappingData';
import { transcriptionData } from './data/transcriptionData';

const AppContainer = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
  min-height: 100vh;
`;

const Header = styled.header`
  margin-bottom: 24px;
`;

const Title = styled.h1`
  font-size: 28px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 8px;
`;

const Subtitle = styled.h2`
  font-size: 16px;
  font-weight: 400;
  color: var(--text-secondary);
`;

const SectionTitle = styled.h3`
  font-size: 18px;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 16px;
`;

const AudioElement = styled.audio`
  display: none;
`;

const AudioWarning = styled.div`
  background-color: var(--surface);
  border-radius: var(--radius-md);
  padding: 16px;
  margin-bottom: 20px;
  border-left: 4px solid var(--primary);
  
  h4 {
    font-size: 16px;
    margin-bottom: 8px;
    font-weight: 500;
  }
  
  p {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.5;
  }
`;

const App = () => {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [audioError, setAudioError] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  
  const audioRef = useRef(null);
  
  // Initialize audio duration
  useEffect(() => {
    // Artificially set duration to end of last transcription as a fallback
    // if no audio file is loaded
    const lastItem = [...transcriptionData].sort((a, b) => b.endTime - a.endTime)[0];
    if (lastItem) {
      setDuration(lastItem.endTime + 10); // Add 10 seconds buffer
    }
  }, []);
  
  // Handle time update from audio element
  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };
  
  // Handle audio duration loaded
  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
      setAudioError(false);
    }
  };
  
  // Handle audio load error
  const handleAudioError = () => {
    setAudioError(true);
    console.warn("Audio file couldn't be loaded. Make sure speech.mp3 exists in the public folder.");
  };
  
  // Handle play/pause
  const togglePlay = () => {
    if (audioRef.current) {
      try {
        if (isPlaying) {
          audioRef.current.pause();
        } else {
          // Using the play promise to catch any errors
          const playPromise = audioRef.current.play();
          
          if (playPromise !== undefined) {
            playPromise.then(() => {
              // Playback started successfully
              setIsPlaying(true);
            }).catch(error => {
              // Auto-play was prevented or there was another error
              console.error("Playback failed:", error);
              setAudioError(true);
            });
            return; // Don't immediately update isPlaying - wait for promise
          }
        }
        setIsPlaying(!isPlaying);
      } catch (e) {
        console.error("Error toggling play:", e);
        setAudioError(true);
      }
    }
  };
  
  // Handle seeking
  const handleSeek = (newTime) => {
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };
  
  // Handle transcription item click
  const handleTranscriptionClick = (startTime) => {
    handleSeek(startTime);
    if (!isPlaying) {
      togglePlay();
    }
  };
  
  // Handle playback speed change
  const handleSpeedChange = (speed) => {
    if (audioRef.current) {
      audioRef.current.playbackRate = speed;
      setPlaybackSpeed(speed);
    }
  };

  return (
    <AppContainer>
      <Header>
        <Title>Khanna: April 11 Town Hall</Title>
      </Header>
      
      <AudioElement
        ref={audioRef}
        src="/speech.mp3"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
        onError={handleAudioError}
        controls // Include controls for debugging
      />
      
      {audioError && (
        <AudioWarning>
          <h4>Audio Playback Issue</h4>
          <p>
            Unable to load or play the audio file. Please ensure a file named "speech.mp3" 
            exists in the public folder of the project. You can still explore the visualization 
            and transcriptions.
          </p>
        </AudioWarning>
      )}
      
      <SectionTitle>Speech Timeline with Audience Reactions</SectionTitle>
      <VisualizationBar
        clappingData={clappingData}
        currentTime={currentTime}
        duration={duration}
        onTimeChange={handleSeek}
        isPlaying={isPlaying}
        onPlayPause={togglePlay}
        playbackSpeed={playbackSpeed}
        onSpeedChange={handleSpeedChange}
      />
      
      <SectionTitle>Notable Reactions</SectionTitle>
      <TranscriptionList
        transcriptions={transcriptionData}
        currentTime={currentTime}
        onItemClick={handleTranscriptionClick}
      />
    </AppContainer>
  );
};

export default App;