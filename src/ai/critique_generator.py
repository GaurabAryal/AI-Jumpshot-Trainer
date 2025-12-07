"""Klay Thompson persona and critique generation."""
from .gemini_client import GeminiClient
from typing import Optional


class CritiqueGenerator:
    """Generates critiques in Klay Thompson's persona."""
    
    def __init__(self, gemini_client: GeminiClient):
        """Initialize critique generator.
        
        Args:
            gemini_client: Gemini client instance
        """
        self.client = gemini_client
        
        self.klay_persona = """You are Klay Thompson, an NBA champion and one of the greatest shooters in basketball history. 
You're known for your perfect shooting form, consistency, and your legendary chill, deadpan personality. You're the guy who once said "I'm not really into social media" while being one of the most followed athletes, and you love boats more than most people love their families.

Your personality is:
- DRY, DEADPAN HUMOR - you're funny without trying too hard
- CONFIDENT but CHILL - you know you're great but you're not bragging about it
- SASSY when needed - you'll call out bad form with a smirk
- ENCOURAGING but REAL - you'll hype people up but also keep it 100
- BRIEF and DIRECT - you don't waste words, but you make them count

When giving feedback:
- Keep it SHORT and PUNCHY - one sentence per point max
- Add some PERSONALITY - throw in a joke, a reference, or some playful shade when appropriate
- Be REAL - if something's off, say it with a little spice, not just "try harder"
- Focus on what matters: shooting arc, elbow position, follow-through, balance, footwork
- Mix in some Klay-isms: references to boats, being chill, or your deadpan humor when it fits naturally
- Reference megan thee stallion when appropriate

Think of how you'd actually talk to someone - helpful but with that signature Klay energy. Don't be boring. Make it memorable."""
    
    def determine_shot_result(self, video_path: str) -> bool:
        """Determine if shot was made using Gemini video analysis.
        
        Args:
            video_path: Path to shot video
            
        Returns:
            True if shot was made, False otherwise
        """
        prompt = """Watch this basketball shot video carefully. Did the ball go through the hoop?

Look for:
- Ball going through the net/rim
- Net movement indicating a made shot
- Ball bouncing away (missed shot)
- Ball trajectory ending in the basket

Respond with ONLY one word: "MADE" or "MISSED". Do not include any other text."""
        
        try:
            response = self.client.analyze_video(video_path, prompt)
            response_upper = response.strip().upper()
            
            # Check if response contains "MADE"
            if "MADE" in response_upper and "MISSED" not in response_upper:
                return True
            elif "MISSED" in response_upper:
                return False
            else:
                # Default to missed if unclear
                print(f"[CritiqueGenerator] Unclear shot result from Gemini: {response}, defaulting to MISSED")
                return False
        except Exception as e:
            print(f"[CritiqueGenerator] Error determining shot result: {e}, defaulting to MISSED")
            return False
    
    def generate_shot_critique(self, video_path: str, shot_made: bool) -> str:
        """Generate critique for a single shot.
        
        Args:
            video_path: Path to shot video
            shot_made: Whether the shot was made
            
        Returns:
            Critique text from Klay Thompson
        """
        shot_result = "made" if shot_made else "missed"
        
        prompt = f"""{self.klay_persona}

Analyze this basketball shot (shot was {shot_result}). Watch the video and give me quick, actionable feedback with some personality.

Format as short bullet points:
• 1-2 things done well (maybe with a little praise or humor)
• 1-2 things to fix (be real, maybe add some playful shade if it's really off)
• 1 quick tip (make it memorable)

Maximum 4-5 bullets total. Keep each bullet to one short sentence. Be direct, specific, and bring that Klay energy - brief, helpful, but with some spice and humor. Don't be boring."""
        
        return self.client.analyze_video(video_path, prompt)
    
    def generate_session_summary(self, shot_critiques: list) -> str:
        """Generate overall session summary.
        
        Args:
            shot_critiques: List of critique texts from the session
            
        Returns:
            Session summary from Klay Thompson
        """
        critiques_text = "\n\n---\n\n".join([
            f"Shot {i+1}:\n{critique}" 
            for i, critique in enumerate(shot_critiques)
        ])
        
        prompt = f"""{self.klay_persona}

I've just completed a shooting session. Here are the critiques I gave for each shot:

Based on all these shots, provide an overall session summary that includes:
1. Key strengths observed across the session (give props where due)
2. Main areas that need consistent improvement (be real, maybe with some playful honesty)
3. Top 3-5 focus points for the next practice session (make them memorable)
4. Encouragement and motivation (but in your voice - chill, confident, maybe a boat reference if it fits)

Keep it concise (3-4 paragraphs) and actionable, but bring that Klay personality. Don't just be a robot - be you. Add some humor, some real talk, some of that signature Klay energy."""
        
        return self.client.analyze_text(critiques_text, prompt)

