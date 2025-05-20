import os
import base64
import mimetypes
import cv2
from flask import Flask, request, jsonify
from PIL import Image
import google.generativeai as genai

# Setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini
genai.configure(api_key="AIzaSyAtZdcm9nN--eMNlWoiF0wRuTwE70mBkV4")
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Helpers
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

# Endpoint 1: Generate Detailed Questions
@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    file = request.files.get('media')
    description = request.form.get('description')

    if not file or not description:
        return jsonify({"error": "Media and description are required"}), 400

    media = load_media(file)

    prompt = f"""You are a visual content educator.
Analyze the provided visual input and the concept '{description}'.
Generate 7 concise, relevant, and concept-related questions based on the visual context.
Number each question clearly and keep each question 5 words.
Output format:
1. Question 1
2. Question 2
..."""

    response = model.generate_content([prompt, media])
    raw_output = response.text.strip()
    questions = [q.strip().split(". ", 1)[-1] for q in raw_output.split("\n") if q.strip()]
    return jsonify({'questions': questions})

# 🆕 Endpoint 2: Generate Short Recommended Questions
@app.route('/generate_recommended_questions', methods=['POST'])
def generate_recommended_questions():
    file = request.files.get('media')
    description = request.form.get('description')

    if not file or not description:
        return jsonify({"error": "Media and description are required"}), 400

    media = load_media(file)

    prompt = f"""You are a visual question recommender.
Analyze the visual input and the concept '{description}'.
Generate 7 very short recommended questions or prompts (1 to 4 words only) that relate to the visual and concept.
These should be simple triggers or cues.
Output format:
1. Short question
2. Short question
..."""

    response = model.generate_content([prompt, media])
    raw_output = response.text.strip()
    questions = [q.strip().split(". ", 1)[-1] for q in raw_output.split("\n") if q.strip()]
    return jsonify({'recommended_questions': questions})

# Endpoint 3: Answer a Single Question
@app.route('/answer_question', methods=['POST'])
def answer_question():
    data = request.get_json()
    question = data.get('question')
    description = data.get('description')
    media_data = data.get('media')

    if not question or not description or not media_data:
        return jsonify({"error": "Question, description, and media are required"}), 400

    media = {
        "mime_type": media_data['mime_type'],
        "data": media_data['data']
    }

    prompt = f"""Analyze the visual input and the concept '{description}'.
Answer the question clearly and concisely:
"{question}"."""

    response = model.generate_content([prompt, media])
    clean_answer = response.text.strip().replace('\n', ' ')
    return jsonify({'answer': clean_answer})

# Endpoint 4: Answer Multiple Questions
@app.route('/generate_answers', methods=['POST'])
def generate_answers():
    questions = request.form.getlist('questions')
    description = request.form.get('description')
    file = request.files.get('media')

    if not questions or not description or not file:
        return jsonify({"error": "Questions, description, and media are required"}), 400

    media = load_media(file)
    answers = []

    for question in questions:
        prompt = f"Analyze the concept '{description}' and answer the question: {question}"
        response = model.generate_content([prompt, media])
        answers.append({
            "question": question,
            "answer": response.text.strip()
        })

    return jsonify({'answers': answers})

# Health check
@app.route('/', methods=['GET'])
def health_check():
    return "API is running. Endpoints: /generate_questions, /generate_recommended_questions, /answer_question, /generate_answers"

if __name__ == '__main__':
    app.run(debug=True)
