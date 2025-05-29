import os
import re
import mimetypes
import cv2
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai

# Setup
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini Flash
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# Helper functions
def load_image(path):
    return Image.open(path)

def extract_key_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(UPLOAD_FOLDER, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(Image.open(frame_path))

    cap.release()
    return frames

def load_media(file):
    return {
        "mime_type": file.mimetype,
        "data": base64.b64encode(file.read()).decode()
    }

def process_media(file):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith("image"):
        visuals = [load_image(file_path)]
    elif mime_type and mime_type.startswith("video"):
        visuals = extract_key_frames(file_path)
    else:
        return None, None
    
    return visuals, mime_type

# Routes
@app.route("/generate-questions", methods=["POST"])
def generate_questions_endpoint():
    file = request.files.get("file")
    concept = request.form.get("concept")

    if not file or not concept:
        return jsonify({"error": "File and concept are required"}), 400

    visuals, mime_type = process_media(file)
    if not visuals:
        return jsonify({"error": "Unsupported file type"}), 400

    prompt = f"""
You are an intelligent tutor AI.

Analyze the visual input carefully and combine that understanding with the given concept: "{concept}"

Then generate 5 simple questions relevant that relate the concept to the visual scene and 3 in 7 should be mcqs.

Each question should reflect how the concept can be applied or understood in the context of what is seen in the image/video.
    and dont give any special characters in generated output but can use numbers.
"""
    response = model.generate_content([prompt] + visuals)
    return jsonify({"questions": response.text.strip()})

@app.route('/answer_question', methods=['POST'])
def answer_question():
    data = request.get_json()
    question = data.get('question')
    description = data.get('description')

    if not question or not description:
        return jsonify({"error": "Question and description are required"}), 400

    # Generate answer
    answer_prompt = f"""Analyze the concept '{description}' and answer the following question:
{question}

Provide a clear and concise answer in 2-3 sentences."""
    answer_response = model.generate_content(answer_prompt)
    clean_answer = answer_response.text.strip()

    # Generate recommended questions
    questions_prompt = f"""Generate 5 word three concise follow-up questions based on the concept and Q/A below.
Format each question on a separate line starting with '1.', '2.', '3.'.

Concept: {description}
Question: {question}
Answer: {clean_answer}

Recommended Follow-up Questions:
1."""
    
    questions_response = model.generate_content(questions_prompt)
    raw_questions = questions_response.text.strip()

    # Parse questions
    questions = []
    for line in raw_questions.split('\n'):
        line = line.strip()
        # Extract first three numbered questions
        if len(questions) >= 3:
            break
        if re.match(r'^\d+[.)]', line):
            q = re.sub(r'^\d+[.)]\s*', '', line)
            questions.append(q)

    # Ensure we always return exactly three questions
    while len(questions) < 3:
        questions.append("Could not generate question")

    return jsonify({
        'answer': clean_answer,
        'recommended_questions': questions[:3]
    })


def parse_questions(questions_string):
    """Parse questions string into structured format"""
    questions = []
    blocks = [b.strip() for b in questions_string.split('\n\n') if b.strip()]
    
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if not lines:
            continue
            
        # Extract question number and text
        q_match = re.match(r'^(\d+)\.\s+(.*)', lines[0])
        if not q_match:
            continue
            
        q_number = int(q_match.group(1))
        q_text = q_match.group(2)
        
        # Determine question type
        is_multiple_choice = any(
            re.match(r'^[a-d]\)\s', line) for line in lines[1:]
        ) or "true or false" in q_text.lower()
        
        # Handle options for multiple choice
        options = {}
        if is_multiple_choice:
            for line in lines[1:]:
                opt_match = re.match(r'^([a-d])\)\s*(.*)', line)
                if opt_match:
                    options[opt_match.group(1)] = opt_match.group(2)
            
            # Add true/false options if missing
            if "true or false" in q_text.lower() and not options:
                options = {'a': 'True', 'b': 'False'}
        
        questions.append({
            "question_number": q_number,
            "question_type": "multiple_choice" if is_multiple_choice else "open_ended",
            "question": q_text,
            "options": options if options else None
        })
    
    return sorted(questions, key=lambda x: x['question_number'])

def generate_answers(context, questions):
    """Generate answers for all questions using Gemini"""
    results = []
    for q in questions:
        # Build the prompt
        prompt = f"""
        CONTEXT: {context}
        QUESTION: {q['question']}
        """
        
        # Add instructions based on question type
        if q['question_type'] == 'multiple_choice':
            options_str = '\n'.join([f"{k}) {v}" for k, v in q['options'].items()])
            prompt += f"\nOPTIONS:\n{options_str}\n"
            prompt += "INSTRUCTION: Answer with ONLY the letter of the correct choice (e.g. 'a')"
        else:
            prompt += "\nINSTRUCTION: Answer concisely in 1-2 sentences"
        
        # Get Gemini response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=500
            )
        )
        
        # Clean and store answer
        answer = response.text.strip()
        if q['question_type'] == 'multiple_choice':
            # Extract just the letter for MCQs
            letter_match = re.search(r'^\s*([a-d])\s*$', answer, re.IGNORECASE)
            if letter_match:
                answer = letter_match.group(1).lower()
        
        # Add to results
        q['answer'] = answer
        results.append(q)
    
    return results

@app.route('/generate_answers', methods=['POST'])
def api_handler():
    try:
        # Get request data
        data = request.json
        context = data.get('context', '')
        questions_str = data.get('questions', '')
        
        if not context:
            return jsonify({"error": "Missing context"}), 400
        if not questions_str:
            return jsonify({"error": "Missing questions"}), 400
        
        # Parse and process questions
        questions = parse_questions(questions_str)
        results = generate_answers(context, questions)
        
        return jsonify({"questions": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check
@app.route("/", methods=["GET"])
def health_check():
    return "API is running. Available endpoints: POST /generate-questions, /generate_questions, /answer_question, /generate_answers"

if __name__ == "__main__":
    app.run(debug=True)
