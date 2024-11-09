import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from openai import OpenAI
import os
import faiss
import numpy as np
from dotenv import load_dotenv
import pickle
import uuid
from gtts import gTTS
import io
import base64
import speech_recognition as sr

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Set your secret key here
app.config['SESSION_TYPE'] = 'filesystem'

# Enable CORS for cross-origin requests
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Path to store cached embeddings
EMBEDDINGS_CACHE_FILE = 'embeddings_cache.pkl'

# Initialize embedding cache
embedding_cache = {}

# In-memory storage to replace Redis
class InMemoryStorage:
    def __init__(self):
        self.storage = {}

    def rpush(self, key, value):
        if key not in self.storage:
            self.storage[key] = []
        self.storage[key].append(value)

    def lindex(self, key, index):
        if key in self.storage and -len(self.storage[key]) <= index < len(self.storage[key]):
            return self.storage[key][index]
        return None

# Initialize in-memory storage
in_memory_storage = InMemoryStorage()

def load_embeddings_from_cache():
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None, None

def get_embedding(text, model="text-embedding-ada-002"):
    if text in embedding_cache:
        return embedding_cache[text]
    
    if isinstance(text, str) and len(text) > 0:
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        embedding_cache[text] = embedding
        return embedding
    else:
        raise ValueError(f"Invalid input for embedding: {text}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_query = data.get('query')
        
        user_id = session.get('user_id')
        if not user_id:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id
        
        in_memory_storage.rpush(f'session:{user_id}:queries', user_query)

        if user_query.lower() == "what was my previous question?":
            previous_question = in_memory_storage.lindex(f'session:{user_id}:queries', -2)
            if previous_question:
                return jsonify({'response': f"Your previous question was: {previous_question}"})
            else:
                return jsonify({'response': "I don't have any previous questions."})

        faiss_index, sections = load_embeddings_from_cache()

        if faiss_index is None or sections is None:
            return jsonify({'response': "Embeddings not found. Please generate them first."})

        query_embedding = get_embedding(user_query)

        _, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=3)

        relevant_sections = [sections[i] for i in indices[0] if i != -1]
        if relevant_sections:
            relevant_content = ' '.join(relevant_sections)
            gpt4_response = generate_gpt4_response(relevant_content, user_query)
            return jsonify({'response': gpt4_response})
        else:
            return jsonify({'response': "No relevant content found."})

    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({'response': "Sorry, we're experiencing technical difficulties. Please try again later."})

def generate_gpt4_response(context, query):
    prompt = f"""
You are an AI assistant created by Excellence Driving. Relevant context: {context}
User Query: {query}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content

@socketio.on('audio_data')
def handle_audio_data(data):
    audio_data = base64.b64decode(data['audio'])
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_data)
    
    r = sr.Recognizer()
    with sr.AudioFile("temp_audio.wav") as source:
        audio = r.record(source)
    
    try:
        text = r.recognize_google(audio)
        emit('transcription', {'text': text})
    except sr.UnknownValueError:
        emit('transcription', {'text': "Sorry, I couldn't understand that."})
    except sr.RequestError as e:
        emit('transcription', {'text': "Error processing speech. Try again."})

    os.remove("temp_audio.wav")

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text')
    tts = gTTS(text=text, lang='en')
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return send_file(audio_io, mimetype='audio/mp3')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
