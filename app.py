import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from google.generativeai.errors import APIError

# --- Configuration and Initialization ---

# Azure App Service will securely provide the API key via Application Settings.
API_KEY = os.environ.get("GOOGLE_API_KEY")

# Hardcoded model name as requested
MODEL_TO_USE = 'gemini-2.5-flash' 

if not API_KEY:
    # In a cloud environment, print a fatal message and allow the host 
    # (like Gunicorn/Azure) to handle the startup failure.
    print("FATAL: GOOGLE_API_KEY environment variable not found. The application cannot start.")

try:
    # Only configure if the key is available to avoid runtime errors on startup
    if API_KEY:
        genai.configure(api_key=API_KEY)
        # The GenerativeModel instance now explicitly uses gemini-2.5-flash
        model = genai.GenerativeModel(MODEL_TO_USE)
    else:
        # Create a placeholder object or handle startup failure gracefully
        model = None 
except Exception as e:
    # Handle configuration failure if key is present but invalid
    print(f"ERROR: Failed to configure Google Generative AI: {e}")
    model = None

# Initialize Flask app
# The name must be 'app' for Azure/Gunicorn to easily find it.
app = Flask(__name__)

# Configure CORS
# In production, replace "*" with your specific front-end origin(s)
CORS(app, resources={r"/api/*": {"origins": "*", "supports_credentials": True}})

# --- Core LLM Logic ---

def execute_will_i_regret_buying(query: str):
    """
    Skill: Will I regret buying this - Generates the regret analysis using the Gemini API.
    """
    if not model:
        raise Exception("AI model failed to initialize due to missing or invalid API key.")

    prompt = f"""
You are a smart shopping assistant.

ITEM CONTEXT: {query}

TASK:
1) Analyze factors:
   - Style (trendy vs timeless)
   - Price (fairness)
   - Material quality
   - Return policy
   - Personal relevance
   - Reviews & brand reputation
   - Redundancy (similar items already owned)
   - Lasting desire (will I still want it in a week?)
2) Output:
   - Main title: "You are X% likely to regret buying [ITEM] ([Do it or Don't do it]!)"
   - Table: Factor | What I See | Regret Risk (🔴,🟡,🟢)
   - Comparable purchases based on browsing history
   - Tone: friendly, honest, human

Return in readable, structured format.
"""
    response = model.generate_content(prompt)
    return response.text

# --- Health Check Route ---

@app.route('/check', methods=['GET'])
def check():
    """
    Simple health check route used to confirm the backend server is running and accessible.
    """
    status_code = 200
    if not model:
        # Return a warning status if the model failed to initialize
        status_code = 503
        message = "backend is running, but AI model failed to initialize."
    else:
        message = "backend is running"

    return jsonify({
        'status': 'ok' if status_code == 200 else 'error', 
        'message': message, 
        'model': MODEL_TO_USE
    }), status_code

# --- Main API Route ---

@app.route('/api/execute', methods=['POST'])
def execute():
    # 1. Basic Input Validation and Parsing
    if not model:
        # Fail fast if the model isn't configured
        return jsonify({'error': 'AI service not initialized. Check API key configuration.'}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON payload.'}), 400

    query = data.get('query')
    # Validate that query is a non-empty string
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({'error': 'Missing or empty "query" field in the request.'}), 400
    
    # 2. Execution and Specific Error Handling
    try:
        # Call the core logic
        result = execute_will_i_regret_buying(query)
        
        return jsonify({'success': True, 'result': result})
        
    except APIError as e:
        # Handle specific Gemini API errors (e.g., rate limits, invalid prompts)
        print(f"Gemini API Error: {e}")
        # Use a 503 Service Unavailable or 500 Internal Server Error
        return jsonify({'error': f'AI Service Unavailable or request error: {e}'}), 503
        
    except Exception as e:
        # Catch all other unexpected errors (e.g., network, internal logic)
        print(f"Internal Server Error: {e}")
        return jsonify({'error': 'An unexpected internal server error occurred.'}), 500
