"""Test script to find available Gemini models."""
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def test_gemini_models():
    """Test which Gemini models are available."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables")
        return
    
    print(f"API Key found: {api_key[:10]}...")
    print("\n" + "="*60)
    print("Testing Gemini Models")
    print("="*60 + "\n")
    
    genai.configure(api_key=api_key)
    
    # First, list all available models
    print("Fetching available models...")
    try:
        models = genai.list_models()
        model_list = list(models)
        print(f"\nFound {len(model_list)} models:\n")
        
        available_models = []
        for model in model_list:
            model_name = model.name
            # Extract just the model identifier (e.g., 'gemini-pro' from 'models/gemini-pro')
            if '/' in model_name:
                model_id = model_name.split('/')[-1]
            else:
                model_id = model_name
            
            # Check if it supports generateContent
            supported_methods = []
            if hasattr(model, 'supported_generation_methods'):
                supported_methods = list(model.supported_generation_methods)
            elif hasattr(model, 'supported_methods'):
                supported_methods = list(model.supported_methods)
            
            supports_generate = 'generateContent' in supported_methods or len(supported_methods) == 0
            
            print(f"  Model: {model_id}")
            print(f"    Full name: {model_name}")
            if supported_methods:
                print(f"    Supported methods: {supported_methods}")
            print(f"    Supports generateContent: {supports_generate}")
            if supports_generate or len(supported_methods) == 0:
                available_models.append(model_id)
            print()
        
        print("\n" + "="*60)
        print("Testing models that support generateContent...")
        print("="*60 + "\n")
        
        # Test each model that supports generateContent
        working_models = []
        for model_id in available_models:
            print(f"Testing {model_id}...")
            try:
                model = genai.GenerativeModel(model_id)
                # Try a simple text generation
                response = model.generate_content("Say 'Hello' in one word.")
                print(f"  ✓ {model_id} WORKS! Response: {response.text}")
                working_models.append(model_id)
            except Exception as e:
                print(f"  ✗ {model_id} FAILED: {str(e)}")
            print()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        if working_models:
            print(f"\n✓ Working models: {', '.join(working_models)}")
            print(f"\nRecommended model: {working_models[0]}")
        else:
            print("\n✗ No working models found!")
            print("Please check your API key and account permissions.")
        
    except Exception as e:
        print(f"ERROR listing models: {e}")
        print("\nTrying fallback: testing common model names...")
        
        # Fallback: test common model names
        common_models = [
            'gemini-pro',
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-2.0-flash-exp',
            'gemini-1.0-pro',
        ]
        
        working_models = []
        for model_name in common_models:
            print(f"Testing {model_name}...")
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Say 'Hello' in one word.")
                print(f"  ✓ {model_name} WORKS! Response: {response.text}")
                working_models.append(model_name)
            except Exception as e:
                print(f"  ✗ {model_name} FAILED: {str(e)}")
        
        if working_models:
            print(f"\n✓ Working models: {', '.join(working_models)}")
            print(f"\nRecommended model: {working_models[0]}")
        else:
            print("\n✗ No working models found!")

if __name__ == "__main__":
    test_gemini_models()


