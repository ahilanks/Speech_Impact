#!/bin/bash

echo "🚀 Starting Speech Impact Visualizer development server..."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed. Please install Node.js and npm first."
    exit 1
fi

# Check if node_modules exists, if not run npm install
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    
    if [ $? -ne 0 ]; then
        echo "❌ Error installing dependencies. Please check your network connection and try again."
        exit 1
    fi
fi

# Add OpenSSL legacy provider for Node.js v17+ compatibility
echo "🔒 Setting up OpenSSL legacy provider for compatibility..."
export NODE_OPTIONS=--openssl-legacy-provider

# Start the development server
echo "🔧 Starting development server..."
npm start 