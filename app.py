
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import argparse

if "GOOGLE_API_KEY" not in os.environ:
    print("Please set the GOOGLE_API_KEY environment variable.")
    exit()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

app = Flask(__name__)
CORS(app)

def execute_will_i_regret_buying(query):
    """
    Skill: Will I regret buying this
    """
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

@app.route('/api/execute', methods=['POST'])
def execute():
    data = request.get_json()
    query = data.get('query', '')
    try:
        result = execute_will_i_regret_buying(query)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the will_i_regret_buying agent as a Flask app.")
    parser.add_argument("--port", type=int, default=5008, help="Port to run the Flask app on.")
    args = parser.parse_args()
    app.run(port=args.port)
