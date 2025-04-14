# Speech Impact Visualizer

A modern React application that visualizes audience reactions (clapping) during a speech, along with searchable transcriptions. This application allows you to analyze when audience reactions occurred and how intense they were.

## Features

- **Interactive Timeline Visualization**: View Gaussian curves representing clapping intensity throughout the speech
- **Audio Playback**: Play the speech audio with a marker showing the current position
- **Searchable Transcriptions**: Search through speech transcriptions to find specific content
- **Highlighted Text**: As audio plays, the active transcription has words highlighted in sync with the audio
- **Dark Modern UI**: Clean, dark-themed interface with rounded corners for easy readability

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   npm install
   ```
   or
   ```
   yarn
   ```

3. Add your speech audio file:
   - Place your audio file as `speech.mp3` in the `public` folder

4. Start the development server:
   ```
   npm start
   ```
   or
   ```
   yarn start
   ```

5. Open your browser and navigate to `http://localhost:3000`

## Usage

- **Timeline Navigation**: Click or drag on the visualization bar to navigate to different parts of the speech
- **Play/Pause**: Use the play/pause button to control audio playback
- **Volume Control**: Adjust the volume using the slider
- **Search Transcriptions**: Type in the search box to filter transcriptions
- **Jump to Transcription**: Click on any transcription item to jump to that part of the speech

## Data Format

The application uses two main data files:

- `src/data/clappingData.js`: Contains timestamps and intensity of clapping events
- `src/data/transcriptionData.js`: Contains transcriptions with their timestamps and associated clapping intensity

## Customization

To use your own data:

1. Replace the content in the data files with your own speech transcription and audience reaction data
2. Place your audio file in the `public` folder as `speech.mp3`

## License

This project is open source and available under the MIT License.

## Acknowledgements

- Built with React, styled-components, and D3.js
- Uses modern JavaScript (ES6+) features 