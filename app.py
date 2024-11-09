import eventlet
eventlet.monkey_patch()
 
from flask import Flask, render_template, request, jsonify, session, send_file
from flask_socketio import SocketIO, emit
from openai import OpenAI
import os
import faiss
import numpy as np
from dotenv import load_dotenv
import pickle
import uuid
import speech_recognition as sr
from gtts import gTTS
import io
 
# Load environment variables from .env file
load_dotenv()
 
# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Set your secret key here
app.config['SESSION_TYPE'] = 'filesystem'
socketio = SocketIO(app)
 
# Path to store cached embeddings
EMBEDDINGS_CACHE_FILE = 'embeddings_cache.pkl'
 
# Store user queries and their embeddings in a dictionary for caching
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
 
# Initialize the in-memory storage
in_memory_storage = InMemoryStorage()
 
def load_embeddings_from_cache():
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None, None
 
def get_embedding(text, model="text-embedding-ada-002"):
    # Check if the embedding is already cached
    if text in embedding_cache:
        return embedding_cache[text]
 
    if isinstance(text, str) and len(text) > 0:
        # Use OpenAI's embedding API to get the embedding
        response = client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        # Cache the embedding for future use
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
 
        # Generate a unique user ID for the session
        user_id = session.get('user_id')
        if not user_id:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id
        # Store the user query in in-memory storage
        in_memory_storage.rpush(f'session:{user_id}:queries', user_query)
 
        # Check if the user is asking about their previous question
        if user_query.lower() == "what was my previous question?":
            previous_question = in_memory_storage.lindex(f'session:{user_id}:queries', -2)  # Get the second last question
            if previous_question:
                return jsonify({'response': f"Your previous question was: {previous_question}"})
            else:
                return jsonify({'response': "I don't have any previous questions."})
 
        # Load cached embeddings if available
        faiss_index, sections = load_embeddings_from_cache()
 
        if faiss_index is None or sections is None:
            return jsonify({'response': "Embeddings not found. Please generate them first."})
 
        # Get embedding for user query using the cache or generate new one
        query_embedding = get_embedding(user_query)
 
        try:
            # Retrieve relevant content using FAISS
            _, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=3)
 
            # Gather the most relevant sections based on FAISS index results
            relevant_sections = [sections[i] for i in indices[0] if i != -1]
 
            if relevant_sections:
                relevant_content = ' '.join(relevant_sections)
                # Use your custom prompt for generating a response
                gpt4_response = generate_gpt4_response(relevant_content, user_query)
                return jsonify({'response': gpt4_response})
            else:
                return jsonify({'response': "No relevant content found."})
        except Exception as faiss_error:
            app.logger.error(f"FAISS search error: {str(faiss_error)}")
            return jsonify({'response': "Sorry, we couldn't process your query. Please try again."})
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({'response': "Sorry, we're experiencing technical difficulties. Please try again later."})
 
def generate_gpt4_response(context, query):
    prompt = f"""
You are an AI assistant created by Excellence Driving to provide prompt and helpful customer service. Your knowledge is limited to information about Excellence Driving, its driving classes, licensing services, policies, and company details.
 
Relevant context: {context}
 
**Excellence Driving Services:**
1. Driving Classes: Beginner, advanced defensive, specialized (seniors/teens), refresher courses.
2. Licensing Services: Learner's permit, written test prep, behind-the-wheel prep, license renewal.
 
**Excellence Driving Policies:**
- 48-hour advance booking required
- Late cancellation fee may apply
- Valid learner's permit required for behind-the-wheel lessons
- Payment due at booking
 
When responding to user queries:
1. Provide clear, concise, and helpful answers about Excellence Driving's services or policies.
2. For specific driving techniques, offer general guidance and recommend discussing details with an instructor.
3. For non-Excellence Driving topics, politely redirect to the local Department of Motor Vehicles.
 
IMPORTANT: Keep your responses brief and to the point. Aim for no more than 2-3 short sentences unless absolutely necessary.
 
User Query: {query}
 
How may I assist you with Excellence Driving's services?
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
 
@socketio.on('start_recording')
def handle_recording():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        emit('transcription', {'text': text})
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
 
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get('text')
    tts = gTTS(text=text, lang='en')
    # Save the audio to a BytesIO object
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return send_file(audio_io, mimetype='audio/mp3')
 
if __name__ == '__main__':
    socketio.run(app, debug=True)
