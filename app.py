import os
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

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    file = request.files['media']
    description = request.form['description']
    media = load_media(file)

    prompt = f"""You are a visual content educator.
Analyze the provided visual input and the concept '{description}'.
Generate 7 concise, relevant, and concept-related questions based on the visual context.
Number each question clearly and the questions 5 words.
Output format:
1. Question 1
2. Question 2
..."""

    response = model.generate_content([
        prompt,
        media
    ])

    raw_output = response.text.strip()
    questions = [q.strip().split(". ", 1)[-1] for q in raw_output.split("\n") if q.strip()]
    return jsonify({'questions': questions})

@app.route('/answer_question', methods=['POST'])
def answer_question():
    data = request.get_json()
    question = data.get('question')
    description = data.get('description')

    if not question or not description:
        return jsonify({"error": "Question and description are required"}), 400

    prompt = f"""Analyze the concept '{description}' and answer the following question:
{question}

Provide a clear and concise answer."""
    
    response = model.generate_content(prompt)
    clean_answer = response.text.strip()
    return jsonify({'answer': clean_answer})

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate_answers', methods=['POST'])
def generate_answers():
    data = request.get_json()
    description = data.get('description', '')
    questions_passage = data.get('questions', '')

    if not questions_passage or not description:
        return jsonify({"error": "Both questions and description are required"}), 400

    # Parse multi-line questions with options
    raw_lines = [line.strip() for line in questions_passage.split('\n') if line.strip()]
    questions = []
    current_question = None

    for line in raw_lines:
        # Detect numbered questions (e.g., "1.", "2)")
        if re.match(r'^\d+[.)]\s', line):
            if current_question:
                questions.append(current_question)
            # Remove numbering prefix
            line = re.sub(r'^\d+[.)]\s*', '', line)
            current_question = line
        elif current_question is not None:
            # Append options/continuations to current question
            current_question += '\n' + line

    if current_question:
        questions.append(current_question)

    # Generate answers with error handling
    answers = []
    for question in questions:
        try:
            prompt = f"""Analyze the concept "{description}" and answer this question:
            {question}
            
            For multiple-choice questions, provide the letter answer (e.g., "b)") 
            followed by a brief explanation."""
            
            response = model.generate_content(prompt)
            answers.append({
                "question": question,
                "answer": response.text.strip() if response.text else "No response from model"
            })
        except Exception as e:
            answers.append({
                "question": question,
                "answer": f"Error generating answer: {str(e)}"
            })

    return jsonify({'answers': answers})


# Health check
@app.route("/", methods=["GET"])
def health_check():
    return "API is running. Available endpoints: POST /generate-questions, /generate_questions, /answer_question, /generate_answers"

if __name__ == "__main__":
    app.run(debug=True)
