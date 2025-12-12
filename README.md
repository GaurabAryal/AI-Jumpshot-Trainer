Video of testing this out & the build process: https://www.youtube.com/watch?v=r5JTmFDXivk&t=15s  


# AI Basketball Shot Training Application

A desktop Python application that provides real-time basketball shot training with AI coach Klay Thompson. The app captures video, detects shots automatically, analyzes form using Google Gemini, and provides personalized critiques.

## Features

- Real-time video stream with pose detection and body mesh overlay
- Automatic shot detection using computer vision
- AI-powered shot analysis via Google Gemini
- Klay Thompson persona critiques overlaid on saved videos
- Session management with comprehensive summaries
- Support for external cameras

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

3. Run the application:
```bash
python src/main.py
```

## Usage

1. Start the application
2. Select your camera from the dropdown
3. Click "Start Session" to begin a training session
4. Shoot basketballs - the app will automatically detect shots
5. Each shot will be analyzed and saved with Klay's critique
6. Click "End Session" to view the session summary

## Project Structure

- `src/` - Main application source code
- `data/sessions/` - Session metadata JSON files
- `data/videos/` - Recorded shot videos



OG inspo from: https://github.com/farzaa/gemini-bball
