import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

const ListContainer = styled.div`
  width: 100%;
  background-color: var(--surface);
  border-radius: var(--radius-md);
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 500px;
  max-height: 60vh;
  overflow: hidden;
`;

const SearchInput = styled.input`
  width: 100%;
  padding: 12px 16px;
  border-radius: var(--radius-sm);
  background-color: var(--surface-light);
  border: 1px solid #6c3f9c;
  color: var(--text-primary);
  font-size: 14px;
  
  &:focus {
    outline: none;
    border-color: var(--primary);
  }
`;

const TranscriptionItem = styled.div`
  padding: 16px;
  background-color: ${props => props.active ? 'var(--primary)' : 'var(--surface-light)'};
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background-color 0.1s ease;
  
  &:hover {
    background-color: ${props => props.active ? 'var(--primary)' : '#3a3a3a'};
  }
`;

const TranscriptionText = styled.div`
  font-size: 15px;
  line-height: 1.5;
  margin-bottom: 8px;
`;

const HighlightedSpan = styled.span`
  color: #ffffff;
  font-weight: 600;
  text-shadow: 0 0 8px rgba(255, 255, 255, 0.4);
`;

const NormalSpan = styled.span`
  color: ${props => props.active ? 'rgba(255, 255, 255, 0.7)' : 'var(--text-primary)'};
`;

const TranscriptionMeta = styled.div`
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: ${props => props.active ? 'rgba(255, 255, 255, 0.8)' : 'var(--text-secondary)'};
`;

const IntensityIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
`;

const IntensityBar = styled.div`
  width: 60px;
  height: 6px;
  background-color: #444;
  border-radius: 3px;
  overflow: hidden;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: ${props => (props.intensity / 16) * 100}%;
    background-color: ${props => props.active ? '#b794f6' : 'var(--secondary)'};
    border-radius: 3px;
  }
`;

const ScrollableList = styled.div`
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
`;

const EmptyState = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  color: var(--text-secondary);
`;

const ControlsRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 8px;
`;

const SortButton = styled.button`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background-color: ${props => props.active ? 'var(--primary)' : 'var(--surface-light)'};
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  
  svg {
    width: 16px;
    height: 16px;
  }
`;

const TranscriptionList = ({
  transcriptions,
  currentTime,
  onItemClick,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredTranscriptions, setFilteredTranscriptions] = useState(transcriptions);
  const [sortType, setSortType] = useState('time'); // 'time' or 'intensity'
  
  // Search filtering and sorting effect
  useEffect(() => {
    let filtered = [...transcriptions];
    
    // Apply search filter
    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(item => 
        item.transcription.toLowerCase().includes(term)
      );
    }
    
    // Apply sorting
    if (sortType === 'time') {
      filtered.sort((a, b) => a.startTime - b.startTime);
    } else if (sortType === 'intensity') {
      filtered.sort((a, b) => b.intensity - a.intensity);
    }
    
    setFilteredTranscriptions(filtered);
  }, [searchTerm, transcriptions, sortType]);
  
  // Format time as MM:SS
  const formatTime = (timeInSeconds) => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };
  
  // Find which transcription is currently active
  const findActiveTranscription = () => {
    return filteredTranscriptions.find(
      item => currentTime >= item.startTime && currentTime <= item.endTime
    );
  };
  
  const activeTranscription = findActiveTranscription();
  
  // Generate highlighted text for the active transcription
  const getHighlightedText = (text, active) => {
    if (!active) return <NormalSpan>{text}</NormalSpan>;
    
    // This is a simplified version; a real implementation would need
    // word-by-word timing data to highlight the exact words
    
    // Calculate progress with a slight lead to compensate for perceived lag
    // Add an offset (1 second) to make highlighting appear sooner
    const leadCompensation = 1.0;
    let progress = (currentTime + leadCompensation - active.startTime) / (active.endTime - active.startTime);
    
    // Ensure progress stays within bounds
    progress = Math.max(0, Math.min(1, progress)); 
    
    const words = text.split(' ');
    const highlightIndex = Math.floor(words.length * progress);
    
    return (
      <>
        <HighlightedSpan>{words.slice(0, highlightIndex).join(' ')}</HighlightedSpan>
        {highlightIndex > 0 && highlightIndex < words.length && ' '}
        <NormalSpan active={true}>{words.slice(highlightIndex).join(' ')}</NormalSpan>
      </>
    );
  };

  return (
    <ListContainer>
      <div>
        <SearchInput
          type="text"
          placeholder="Search transcriptions..."
          value={searchTerm}
          onChange={e => setSearchTerm(e.target.value)}
        />
        
        <ControlsRow>
          <SortButton 
            active={sortType === 'time'} 
            onClick={() => setSortType('time')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            Time
          </SortButton>
          
          <SortButton 
            active={sortType === 'intensity'} 
            onClick={() => setSortType('intensity')}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 20V10" />
              <path d="M18 20V4" />
              <path d="M6 20v-4" />
            </svg>
            Intensity
          </SortButton>
        </ControlsRow>
      </div>
      
      <ScrollableList>
        {filteredTranscriptions.length > 0 ? (
          filteredTranscriptions.map((item) => {
            const isActive = activeTranscription && activeTranscription.startTime === item.startTime;
            
            return (
              <TranscriptionItem
                key={`${item.startTime}-${item.endTime}`}
                active={isActive}
                onClick={() => onItemClick(item.startTime)}
              >
                <TranscriptionText>
                  {getHighlightedText(item.transcription, isActive ? item : null)}
                </TranscriptionText>
                
                <TranscriptionMeta active={isActive}>
                  <span>{formatTime(item.startTime)} - {formatTime(item.endTime)}</span>
                  <IntensityIndicator>
                    <span>Intensity:</span>
                    <IntensityBar intensity={item.intensity} active={isActive} />
                    <span>{item.intensity.toFixed(1)}</span>
                  </IntensityIndicator>
                </TranscriptionMeta>
              </TranscriptionItem>
            );
          })
        ) : (
          <EmptyState>No matching transcriptions found</EmptyState>
        )}
      </ScrollableList>
    </ListContainer>
  );
};

export default TranscriptionList; 