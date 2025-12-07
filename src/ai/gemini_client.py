"""Google Gemini API client for video analysis."""
import os
import time
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    """Client for Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (uses GEMINI_API_KEY env var if None)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=self.api_key)
        # Use gemini-robotics-er-1.5-preview as requested
        model_name = 'gemini-robotics-er-1.5-preview'
        
        try:
            self.model = genai.GenerativeModel(model_name)
            print(f"[GeminiClient] Using model: {model_name}")
        except Exception as e:
            print(f"[GeminiClient] Failed to load {model_name}: {e}")
            # Fallback to other models if gemini-robotics-er-1.5-preview doesn't work
            fallback_models = [
                'gemini-2.5-flash',
                'gemini-2.5-pro',
                'gemini-2.0-flash',
                'gemini-2.0-flash-001',
            ]
            
            self.model = None
            for fallback_name in fallback_models:
                try:
                    self.model = genai.GenerativeModel(fallback_name)
                    print(f"[GeminiClient] Using fallback model: {fallback_name}")
                    break
                except Exception as fallback_error:
                    print(f"[GeminiClient] Failed to load {fallback_name}: {fallback_error}")
                    continue
            
            if self.model is None:
                raise ValueError("Could not initialize any model. Please check your API key and available models.")
        
    def analyze_video(self, video_path: str, prompt: str) -> str:
        """Analyze a video using Gemini.
        
        Args:
            video_path: Path to video file
            prompt: Analysis prompt
            
        Returns:
            Analysis response text
        """
        video_file = None
        try:
            # Upload video file
            print(f"[GeminiClient] Uploading video file: {video_path}")
            video_file = genai.upload_file(path=video_path)
            
            # Wait for file to become ACTIVE (required before use)
            # Video files need processing time before they can be used
            print(f"[GeminiClient] Waiting for file to become ACTIVE...")
            max_wait_time = 60  # Maximum 60 seconds
            wait_interval = 2  # Check every 2 seconds
            elapsed_time = 0
            
            # Check if file object has state attribute directly
            file_ready = False
            if hasattr(video_file, 'state'):
                state = video_file.state
                state_name = state.name if hasattr(state, 'name') else str(state)
                if state_name == "ACTIVE":
                    file_ready = True
                    print(f"[GeminiClient] File is already ACTIVE")
            
            # Poll file state until ACTIVE
            while not file_ready and elapsed_time < max_wait_time:
                try:
                    # Get updated file state
                    current_file = genai.get_file(video_file.name)
                    state = current_file.state
                    state_name = state.name if hasattr(state, 'name') else str(state)
                    
                    if state_name == "ACTIVE":
                        file_ready = True
                        print(f"[GeminiClient] File is ACTIVE, proceeding with analysis")
                        break
                    elif state_name == "FAILED":
                        raise Exception(f"File upload failed with state: {state_name}")
                    
                    # File is still processing
                    if elapsed_time % 6 == 0:  # Print every 6 seconds to avoid spam
                        print(f"[GeminiClient] File state: {state_name}, waiting... ({elapsed_time}s)")
                    
                except Exception as state_error:
                    # If we can't get file state, wait a bit and try to proceed
                    if elapsed_time < 10:
                        if elapsed_time % 4 == 0:
                            print(f"[GeminiClient] Checking file state... ({elapsed_time}s)")
                    else:
                        # After 10 seconds, try to proceed (file might be ready)
                        print(f"[GeminiClient] Proceeding after wait period (file may be ready)")
                        file_ready = True
                        break
                
                time.sleep(wait_interval)
                elapsed_time += wait_interval
            
            if not file_ready:
                print(f"[GeminiClient] Warning: Could not confirm ACTIVE state, proceeding anyway...")
            
            # Generate content
            print(f"[GeminiClient] Generating content with Gemini...")
            response = self.model.generate_content([prompt, video_file])
            
            # Clean up uploaded file
            try:
                genai.delete_file(video_file.name)
                print(f"[GeminiClient] Cleaned up uploaded file")
            except Exception as cleanup_error:
                print(f"[GeminiClient] Warning: Failed to delete file: {cleanup_error}")
            
            return response.text
            
        except Exception as e:
            # Clean up file on error
            if video_file:
                try:
                    genai.delete_file(video_file.name)
                except:
                    pass
            return f"Error analyzing video: {str(e)}"
    
    def analyze_text(self, text: str, prompt: str) -> str:
        """Analyze text using Gemini.
        
        Args:
            text: Text to analyze
            prompt: Analysis prompt
            
        Returns:
            Analysis response text
        """
        try:
            full_prompt = f"{prompt}\n\n{text}"
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Error analyzing text: {str(e)}"

