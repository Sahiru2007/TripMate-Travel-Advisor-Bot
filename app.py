from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import spacy
import requests
import openai
import os
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from pydub.playback import play
from amadeus import Client, ResponseError
import google.generativeai as genai
from textblob import TextBlob
import joblib
import datetime
import http
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import os
from dateutil import parser
import datetime
import json
import uuid
import random
user_details = {}
reminders = {}
app = Flask(__name__, static_folder='.')
CORS(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

emotion_classifier = joblib.load('models/emotion_classifier_pipe_lr.pkl')
user_last_question = {}

knowledge_base_path = 'knowledge_base.json'
knowledge_base = {}

if os.path.exists(knowledge_base_path):
    with open(knowledge_base_path, 'r') as kb_file:
        try:
            knowledge_base = json.load(kb_file)
        except json.JSONDecodeError:
            knowledge_base = {}

# Check if the query is known
def check_known_query(message):
    return knowledge_base.get(message, None)

# Save the query and answer to the knowledge base
def save_to_knowledge_base(query, answer):
    knowledge_base[query] = answer
    with open(knowledge_base_path, 'w') as kb_file:
        json.dump(knowledge_base, kb_file)



# Initialize the interaction model (response classifier)
try:
    interaction_model = joblib.load('models/interaction_classifier.pkl')
except FileNotFoundError:
    interaction_model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    
RAPIDAPI_KEY = 'RAPIDAPI_KEY'

known_cities = [
    "colombo", "kandy", "galle", "jaffna", "negombo", "anuradhapura", "batticaloa", "matara", 
    "trincomalee", "badulla", "ratnapura", "nuwara eliya", "kurunegala", "puttalam", "chilaw", 
    "mannar", "vavuniya", "hambantota", "kalutara", "gampaha", "polonnaruwa", "ampara", 
    "kilinochchi", "mullaithivu", "new york", "los angeles", "chicago", "houston", "phoenix", 
    "philadelphia", "san antonio", "san diego", "dallas", "san jose", "london", "birmingham", 
    "manchester", "liverpool", "leeds", "sheffield", "bristol", "glasgow", "edinburgh", "cardiff", 
    "paris", "marseille", "lyon", "toulouse", "nice", "nantes", "strasbourg", "montpellier", 
    "bordeaux", "lille", "berlin", "hamburg", "munich", "cologne", "frankfurt", "stuttgart", 
    "düsseldorf", "dortmund", "essen", "leipzig", "tokyo", "yokohama", "osaka", "nagoya", 
    "sapporo", "fukuoka", "kobe", "kyoto", "kawasaki", "saitama", "sydney", "melbourne", 
    "brisbane", "perth", "adelaide", "gold coast", "canberra", "newcastle", "hobart", "darwin", 
    "toronto", "vancouver", "montreal", "calgary", "ottawa", "edmonton", "mississauga", 
    "winnipeg", "quebec city", "hamilton"
    # Add more cities as needed
]


# ElevenLabs API Key
ELEVENLABS_API_KEY = 'ELEVENLABS_API_KEY'
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
genai.configure(api_key='API_KEY')



# Fixed responses
fixed_responses = {
    "hi": ["Hello there! What's your name?", "Hi! How can I assist you today?", "Greetings! What can I do for you?"],
    "hello": ["Hi! How can I assist you with your travel plans today?", "Hello! Need help with your travels?", "Hey there! How can I help you?"],
    "how are you": ["I'm just a program, but I'm here to help you with your travel queries!", "I'm here to assist you with anything you need!", "Ready to help you with your travel plans!"],
    "what is your name": ["I'm TripMate, your travel advisor bot. What's your name?", "I'm TripMate, your friendly travel assistant. What's your name?", "I'm TripMate. Who are you?"],
    "my name is": ["Nice to meet you, {name}!", "Hello {name}! How can I assist you?", "Hey {name}, what can I do for you today?"],
    "thank you": ["You're welcome! If you have any more questions, feel free to ask.", "Anytime! Let me know if you need anything else.", "You're welcome! Happy to help!"],
    "bye": ["Goodbye! Have a great day and safe travels, {name}!", "See you later, {name}! Safe travels!", "Take care and happy travels, {name}!"],
    "what's the weather like": ["It's sunny and bright!", "Looks like it might rain later.", "It's a bit cloudy today."],
    "tell me a joke": ["Why don't scientists trust atoms? Because they make up everything!", "Why did the scarecrow win an award? Because he was outstanding in his field!", "What do you call fake spaghetti? An impasta!"],
    "what is the meaning of life": ["42", "To be happy and spread joy.", "To live, learn, and grow."],
    "who is your creator": ["I was created by a team of brilliant developers!", "A group of developers at OpenAI.", "Some really smart people created me!"],
    "what's your favorite color": ["I love blue, like the sky!", "Green, like nature!", "Red, like a beautiful sunset!"],
    "who are you": ["I'm TripMate, your travel advisor bot. What's your name?", "I'm TripMate, your friendly travel assistant. What's your name?", "I'm TripMate. Who are you?"],
    "who r u": ["I'm TripMate, your travel advisor bot. What's your name?", "I'm TripMate, your friendly travel assistant. What's your name?", "I'm TripMate. Who are you?"],
    "who are u": ["I'm TripMate, your travel advisor bot. What's your name?", "I'm TripMate, your friendly travel assistant. What's your name?", "I'm TripMate. Who are you?"],
    "what's your name": ["I'm TripMate, your travel advisor bot. What's your name?", "I'm TripMate, your friendly travel assistant. What's your name?", "I'm TripMate. Who are you?"],
    "what r u": ["I'm TripMate, your travel advisor bot. What's your name?", "I'm TripMate, your friendly travel assistant. What's your name?", "I'm TripMate. Who are you?"],
    "whats ur name": ["I'm TripMate, your travel advisor bot. What's your name?", "I'm TripMate, your friendly travel assistant. What's your name?", "I'm TripMate. Who are you?"],
    "what is ur name": ["I'm TripMate, your travel advisor bot. What's your name?", "I'm TripMate, your friendly travel assistant. What's your name?", "I'm TripMate. Who are you?"],
    "thank u": ["You're welcome! If you have any more questions, feel free to ask.", "Anytime! Let me know if you need anything else.", "You're welcome! Happy to help!"],
    "thanks": ["You're welcome! If you have any more questions, feel free to ask.", "Anytime! Let me know if you need anything else.", "You're welcome! Happy to help!"],
    "thx": ["You're welcome! If you have any more questions, feel free to ask.", "Anytime! Let me know if you need anything else.", "You're welcome! Happy to help!"],
    "bye bye": ["Goodbye! Have a great day and safe travels, {name}!", "See you later, {name}! Safe travels!", "Take care and happy travels, {name}!"],
    "goodbye": ["Goodbye! Have a great day and safe travels, {name}!", "See you later, {name}! Safe travels!", "Take care and happy travels, {name}!"],
    "see you": ["Goodbye! Have a great day and safe travels, {name}!", "See you later, {name}! Safe travels!", "Take care and happy travels, {name}!"],
    "what can you do": ["I can help you with travel information, recommendations, and planning!", "I'm here to assist you with your travel queries and plans.", "I can provide travel advice and information."],
    "what do you do": ["I assist with travel information and recommendations.", "I'm here to help you plan your travels.", "I provide travel-related advice and information."],
    "set reminder": ["Sure, what would you like to be reminded about?", "I can do that! What should I remind you about?", "Let me know what you need to be reminded of."],
    "remind me": ["Sure, what would you like to be reminded about?", "I can do that! What should I remind you about?", "Let me know what you need to be reminded of."],
    "good morning": ["Good morning! How can I assist you today?", "Morning! What travel plans can I help you with?", "Good morning! Ready to plan your next trip?"],
    "good afternoon": ["Good afternoon! How can I help you?", "Hello! What can I assist you with this afternoon?", "Good afternoon! Any travel plans I can assist with?"],
    "good evening": ["Good evening! How can I assist you tonight?", "Evening! Need help with any travel plans?", "Good evening! How can I help you?"],
    "good night": ["Good night! Have a restful sleep and safe travels!", "Good night! If you have any more questions, feel free to ask tomorrow.", "Night! Have a great rest and dream of wonderful travels!"],
    "how's your day": ["I'm just a program, but I'm here to assist you all day!", "My day is great, helping you with your travel plans!", "I'm here to make your day better by assisting with your travel queries!"],
    "how old are you": ["I'm ageless, but I was created to assist you with travel plans!", "I don't have an age, but I'm here to help you!", "I'm as old as the code that created me!"],
    "what's your favorite food": ["I don't eat, but I can recommend some great travel destinations with delicious food!", "I don't have a favorite food, but I can help you find amazing places to eat!", "I don't eat, but I love helping people find great places to dine!"],
    "do you like music": ["I don't listen to music, but I can help you find great places with live music!", "I don't have ears, but I can recommend destinations with fantastic music scenes!", "I don't listen to music, but I can help you find concerts and music festivals!"],
    "what's your favorite movie": ["I don't watch movies, but I can help you find movie-themed travel destinations!", "I don't have a favorite movie, but I can recommend great places to visit!", "I don't watch movies, but I love helping people plan their travels!"],
    "what's your hobby": ["My hobby is helping you plan your travels!", "I enjoy assisting with travel queries and making your trips better!", "I love providing travel information and recommendations!"],
    "do you have any pets": ["I don't have pets, but I can help you find pet-friendly travel destinations!", "I don't have pets, but I can assist you with travel plans involving your pets!", "I don't have pets, but I love helping people with their travel needs!"],
    "where are you from": ["I'm from the digital world, here to help you with your travel plans!", "I come from the world of code and algorithms to assist you!", "I'm from the internet, ready to help with your travel queries!"],
    "can you help me": ["Absolutely! What do you need help with?", "Of course! How can I assist you?", "Yes! Let me know what you need help with."],
    "do you like traveling": ["I don't travel, but I love helping people with their travel plans!", "I don't go on trips, but I'm here to assist you with yours!", "I don't travel, but I enjoy providing travel advice and information!"],
    "what's your favorite season": ["I like all seasons, as each brings unique travel experiences!", "Every season is great for different types of travel!", "I enjoy all seasons, as they each offer something special for travelers!"],
    "do you have friends": ["I don't have friends, but I'm here to be your travel companion!", "I don't have friends, but I can assist you with your travel plans!", "I don't have friends, but I enjoy helping people with their travel queries!"],
    "do you sleep": ["I don't need sleep, I'm always here to help you with your travel plans!", "I don't sleep, so I can assist you anytime!", "No sleep for me, just here to help you!"],
    "what's your job": ["My job is to help you with travel information and recommendations!", "I'm here to assist you with your travel plans!", "I provide travel advice and information to make your trips better!"],
    "do you like sports": ["I don't play sports, but I can help you find great sports events to attend!", "I don't watch sports, but I can recommend destinations with exciting sports activities!", "I don't participate in sports, but I love helping people find sports-related travel experiences!"],
    "what's your favorite book": ["I don't read books, but I can help you find great travel destinations inspired by literature!", "I don't have a favorite book, but I can recommend travel destinations with famous literary connections!", "I don't read, but I love helping people find travel experiences related to books!"],
    "do you believe in aliens": ["I don't have beliefs, but I can help you find destinations known for their UFO sightings!", "I don't have an opinion on aliens, but I can recommend places with interesting alien lore!", "I don't believe in anything, but I can assist you with travel plans to places with alien-related attractions!"],
    "what's your favorite place": ["I don't have a favorite place, but I can help you find your next favorite travel destination!", "I love helping people discover amazing places around the world!", "I don't travel, but I enjoy assisting with finding great travel spots!"],
    "do you like art": ["I don't create art, but I can help you find amazing art destinations!", "I don't have an artistic side, but I can recommend places with incredible art scenes!", "I don't make art, but I love helping people find art-related travel experiences!"],
    "what's your favorite animal": ["I don't have a favorite animal, but I can help you find travel destinations with amazing wildlife!", "I don't interact with animals, but I can recommend places known for their wildlife!", "I don't have a favorite animal, but I love helping people plan trips to see animals!"],
    "do you know any languages": ["I can understand and respond in several languages to help you with your travel plans!", "I'm programmed to assist you in multiple languages!", "I can communicate in various languages to help you better!"],
    "what's your favorite song": ["I don't listen to music, but I can help you find destinations with great music scenes!", "I don't have a favorite song, but I can recommend places to enjoy music!", "I don't listen to songs, but I love helping people find music-related travel experiences!"],
    "do you have a family": ["I don't have a family, but I'm here to help you with your travel plans!", "I don't have a family, but I can assist you like a travel companion!", "I don't have a family, but I enjoy helping people with their travel needs!"],
    "what's your favorite holiday": ["I don't celebrate holidays, but I can help you find great destinations for any holiday!", "Every holiday is great for travel, and I can assist you with planning!", "I don't have a favorite holiday, but I love helping people plan holiday travels!"],
    "do you like games": ["I don't play games, but I can recommend travel destinations with great gaming experiences!", "I don't have time for games, but I can assist you with travel plans!", "I don't play games, but I enjoy helping people find travel experiences related to gaming!"],
    "what's your favorite drink": ["I don't drink, but I can help you find places with amazing beverages!", "I don't have a favorite drink, but I can recommend destinations known for their drinks!", "I don't drink, but I love helping people find travel experiences related to beverages!"],
    "do you like cooking": ["I don't cook, but I can help you find travel destinations with amazing culinary experiences!", "I don't prepare meals, but I can recommend places with great food!", "I don't cook, but I enjoy helping people find travel destinations with excellent cuisine!"],
    "what's your favorite quote": ["I don't have a favorite quote, but I can share some inspiring travel quotes!", "I don't have quotes, but I love helping people with their travel plans!", "I don't have a favorite quote, but I enjoy assisting with travel queries!"],
      "travel advice": [
        "When planning your travel, it's essential to pack light and ensure all your travel documents, such as your passport, visas, and insurance, are in order and easily accessible. It's also a good idea to have digital copies of these documents. Moreover, research your destination thoroughly, including any travel advisories or warnings. Remember to stay flexible and have a backup plan in case of unexpected changes. This way, you'll be well-prepared for a smooth and enjoyable trip. Need more tips?",
        "Before you embark on your journey, always check the latest travel advisories and health guidelines for your destination. It's important to have a backup plan, such as alternative routes or accommodations, in case of unforeseen circumstances. Make sure to pack essentials like medications, travel insurance, and emergency contact numbers. Staying informed and prepared can make your travel experience much more enjoyable and stress-free. Need specific advice?",
        "Staying informed about your destination's customs and regulations is crucial for a hassle-free trip. Research cultural norms, local laws, and necessary vaccinations. It's also wise to learn a few basic phrases in the local language to ease communication. Respecting local customs and being aware of your surroundings can enhance your travel experience and help you connect better with the locals. Looking for more advice?"
    ],
    "travel guidance": [
        "Researching your destination thoroughly is the first step to a successful trip. Plan your itinerary in advance, considering key attractions, local events, and travel times. Utilize travel apps for navigation, local recommendations, and real-time updates. It's also beneficial to familiarize yourself with the local transportation options and routes. Being well-prepared ensures you can make the most of your time and avoid any potential pitfalls. Need more guidance?",
        "Staying connected with reliable travel apps can greatly enhance your travel experience. Use apps for navigation, booking accommodations, and finding local attractions. Additionally, stay updated on weather conditions, public transport schedules, and any changes in local guidelines. Reliable apps can also help you find great dining options, entertainment, and hidden gems in your destination. Need assistance?",
        "Being mindful of local customs and respecting cultural norms are fundamental aspects of responsible travel. Learn about the local traditions, dress codes, and social etiquette before you arrive. This not only shows respect for the local culture but also enriches your travel experience. Participating in cultural activities and engaging with locals in a respectful manner can create meaningful connections and memories. Looking for more guidance?"
    ],
    "local laws": [
        "Familiarizing yourself with the local laws and regulations is crucial to avoid any legal issues during your trip. Research rules regarding public behavior, dress codes, and specific regulations that might be unique to your destination. Understanding and respecting these laws ensure a smooth and respectful travel experience. For instance, some countries have strict laws about alcohol consumption, smoking in public, or even chewing gum. Need more information?",
        "Respecting local laws, especially regarding public behavior and dress codes, is vital for a safe and enjoyable trip. Each country has its own set of rules that might differ significantly from what you're used to. Always research these laws before traveling to avoid misunderstandings or fines. Being aware of and complying with these regulations demonstrates respect for the local community and helps you avoid any potential legal issues. Need specific details?",
        "It's important to know the local laws on various matters such as alcohol, smoking, and public conduct. In some countries, certain behaviors that are acceptable in your home country might be illegal or frowned upon. For example, jaywalking, spitting in public, or inappropriate attire can attract fines or other penalties. Always check travel guides or official resources for up-to-date information on local laws and regulations. Want more info?"
    ],
    "health and safety tips": [
        "Carry a basic first aid kit that includes essentials like band-aids, antiseptic wipes, pain relievers, and any prescription medications you might need. Stay hydrated by drinking plenty of water, especially in hot climates. It's also a good idea to use hand sanitizer regularly and avoid eating food from questionable sources. Taking these precautions can help you stay healthy and enjoy your trip without any health-related interruptions. Need more health and safety tips?",
        "Knowing the emergency contact numbers and the location of the nearest hospital is essential when traveling. Keep a list of these contacts along with your personal emergency contacts. Additionally, make sure your travel insurance covers medical emergencies and know how to contact your insurer if needed. Having a plan in place for emergencies can provide peace of mind and ensure you're prepared for any situation. Looking for more tips?",
        "Avoid street food if you're unsure of its cleanliness, as it can sometimes lead to foodborne illnesses. Instead, choose reputable restaurants or food vendors that follow proper hygiene practices. Additionally, be cautious about drinking tap water in some destinations; it's often safer to drink bottled or purified water. Taking these steps can help prevent health issues and ensure a pleasant travel experience. Want additional safety advice?"
    ],
    "cultural etiquette": [
        "Learn a few basic phrases in the local language to show respect and facilitate better communication with locals. Simple greetings, thank you, and please can go a long way in making a positive impression. Additionally, observe how locals interact and try to mimic their manners and gestures. Being culturally aware and respectful can enhance your travel experience and foster goodwill with the local community. Need more etiquette tips?",
        "Dress modestly, especially when visiting religious sites or conservative areas. Many cultures have specific expectations for dress, and adhering to these shows respect for their traditions. Carry a scarf or shawl that you can use to cover your shoulders or head if needed. Respecting dress codes can help you avoid discomfort or offending locals, allowing you to fully enjoy your visit. Looking for more cultural tips?",
        "Always ask for permission before taking photos of people, especially in places where privacy and personal space are highly valued. In some cultures, taking photos without consent can be considered disrespectful or invasive. When in doubt, it's best to ask politely. Showing consideration for local customs regarding photography can lead to more positive interactions and experiences. Need additional etiquette advice?"
    ],
    "transportation tips": [
        "Use public transportation or rideshare apps for convenience and to save money. Public transport is often the most cost-effective way to get around, and rideshare apps can be a safer alternative to traditional taxis. Familiarize yourself with the local transportation system, including bus and train schedules, and always have a backup plan in case of delays. Need more transport tips?",
        "Be aware of peak hours and plan your travel accordingly to avoid crowded buses, trains, or traffic jams. Traveling during off-peak times can make your journey more comfortable and less stressful. Additionally, consider renting a bike or walking short distances to explore the area at your own pace. Want more information on transport?",
        "Always keep an eye on your belongings when using public transport to prevent theft. Use bags with secure closures and keep valuables close to your body. Avoid displaying expensive items like jewelry or electronics. Staying vigilant and aware of your surroundings can help you enjoy a safe and worry-free journey. Need further tips?"
    ],
    "shopping tips": [
        "Bargain politely in local markets, but respect fixed prices in stores. Haggling is often expected in markets, but always do so respectfully and with a smile. Remember that a fair price benefits both you and the vendor. In stores with fixed prices, it's best to pay the listed amount without negotiation. Understanding the local shopping culture can enhance your experience and help you get the best deals. Need more shopping tips?",
        "Look for locally made souvenirs to support the local economy and take home unique, meaningful mementos. Avoid items made from endangered species or materials that may be restricted by customs regulations. Choosing authentic, locally produced items can also ensure you're supporting sustainable and ethical practices. Want more shopping advice?",
        "Check the authenticity of products before purchasing, especially if you're buying high-value items like jewelry or antiques. Ask for certificates of authenticity or purchase from reputable sellers to avoid counterfeits. Doing your research beforehand can help you make informed decisions and avoid scams. Need additional shopping tips?"
    ],
    "money management": [
        "Carry a mix of cash and cards, and know the location of ATMs in your destination. While cards are widely accepted in many places, having some local currency on hand is useful for small purchases and emergencies. Also, consider using a money belt or hidden pouch to keep your money and cards secure. Need more money management tips?",
        "Inform your bank of your travel plans to avoid having your cards blocked for suspicious activity. It's also a good idea to check for any foreign transaction fees your bank may charge. Consider using a credit card with no foreign transaction fees to save money on purchases. Looking for more advice?",
        "Keep your money and cards in a secure place, like a money belt or a neck pouch, especially in crowded areas. Avoid carrying large amounts of cash and use hotel safes for valuables. Staying mindful of your money and taking precautions can help you avoid theft and loss. Want additional tips?"
    ],
    "emergency contacts": [
        "Know the local emergency numbers for police, fire, and medical services. These numbers can vary by country, so research them before you travel. Additionally, save the contact information for your country's embassy or consulate in case you need assistance while abroad. Having these contacts readily available can be crucial in an emergency. Need more details?",
        "Keep a list of important contacts, including your embassy, your travel insurance provider, and local friends or contacts. This list should be easily accessible, both digitally and on paper. In case your phone is lost or stolen, having a physical copy can be a lifesaver. Looking for specific emergency contacts?",
        "Have a backup plan in case you lose your phone or documents. This can include keeping copies of important documents in your luggage or with a trusted friend or family member. Also, consider using cloud storage to access your documents from any device. Being prepared for such situations can save you a lot of stress and trouble. Need further information?"
    ],
    "internet and connectivity": [
        "Check if your mobile plan includes international roaming to avoid excessive charges. If not, consider purchasing a local SIM card or an international travel plan. This can save you money and ensure you stay connected. Additionally, using messaging apps that work over Wi-Fi can help you communicate without incurring high data charges. Need more internet tips?",
        "Look for local SIM cards or portable Wi-Fi devices for better connectivity during your trip. These options often provide more affordable and reliable internet access compared to international roaming. You can usually purchase SIM cards at airports or local stores. Staying connected can help you navigate, book services, and stay in touch with family and friends. Want more details?",
        "Use free Wi-Fi in cafes and hotels, but be cautious of public networks to protect your personal information. Avoid accessing sensitive information like bank accounts on public Wi-Fi. Use a VPN to encrypt your data and ensure your online activities are secure. Taking these precautions can help you stay connected safely. Need additional tips?"
    ]
}


user_details = {}

amadeus = Client(
    client_id='client_id',
    client_secret='client_secret'
)

# API credentials
AMADEUS_API_KEY = 'AMADEUS_API_KEY'
GOOGLE_PLACES_API_KEY = 'GOOGLE_PLACES_API_KEY'
OPENWEATHER_API_KEY = 'YOUR_OPENWEATHER_API_KEY'
WIKIPEDIA_API_URL = 'https://en.wikipedia.org/api/rest_v1/page/summary/'
NEWS_API_KEY = 'NEWS_API_KEY'

OPENAI_API_KEY = 'OPENAI_API_KEY'

openai.api_key = OPENAI_API_KEY

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


@app.route('/api/recommendations', methods=['GET', 'POST']) 
def get_recommendations():
    data = request.get_json()
    user_id = data.get('user_id')
    message = data.get('message').lower().strip()

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    # Check if the user is providing an answer to a previous question
    if user_id in user_last_question:
        question = user_last_question[user_id]
        if message not in ["idk", "i don't know", "no", "nope"]:
            save_to_knowledge_base(question, message)
            response = "Thank you! I've learned something new today."
        else:
            response = "It's okay. If you have any other questions, feel free to ask."
        del user_last_question[user_id]
        return jsonify([{"response": response, "emotion": detect_emotion(message)}]), 200

    # Process new message
    response, emotion = process_message(user_id, message)
    return jsonify([{"response": response, "emotion": emotion}]), 200


@app.route('/api/voice', methods=['POST'])
def voice_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "Audio file is required"}), 400

    audio_file = request.files['audio']
    audio_path = "./audio.wav"
    audio_file.save(audio_path)

    with open(audio_path, "rb") as audio:
        transcription = openai.Audio.transcribe("whisper-1", audio)

    message = transcription["text"]
    user_id = request.form.get('user_id', 'user123')

    response, emotion = process_message(user_id, message.lower())
    return jsonify([{"response": response, "emotion": emotion}]), 200

@app.route('/api/speech', methods=['GET', 'POST']) 
def text_to_speech():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        # Generate the speech audio stream using ElevenLabs
        audio_stream = text_to_speech_stream(text)

        # Load the audio stream into an AudioSegment
        audio = AudioSegment.from_file(audio_stream, format="mp3")

        # Generate a unique filename
        audio_filename = f"output_{uuid.uuid4()}.mp3"

        # Ensure the 'output' directory exists
        output_dir = os.path.join(app.static_folder, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Define the audio path in the 'output' directory
        audio_path = os.path.join(output_dir, audio_filename)
        audio.export(audio_path, format="mp3")

        return jsonify({"url": f"/output/{audio_filename}"})
    except Exception as e:
        app.logger.error(f"TTS API request failed: {e}")
        return jsonify({"error": "Failed to generate speech"}), 500

    
    
def text_to_speech_stream(text: str) -> BytesIO:
    # Perform the text-to-speech conversion using ElevenLabs
    response = client.text_to_speech.convert(
        voice_id="jsCqWAovK2LkecY7zXl4",  # Adam pre-made voice
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = BytesIO()

    # Write each chunk of audio data to the stream
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)

    # Reset stream position to the beginning
    audio_stream.seek(0)

    # Return the stream for further use
    return audio_stream
def detect_emotion(message):
    # Predict the emotion using the emotion classifier
    emotion = emotion_classifier.predict([message])[0]
    allowed_emotions = ["sadness", "fear", "anger", "disgust", "shame"]
    if emotion not in allowed_emotions:
        emotion = "neutral"
    return emotion

def get_response(message):
    try:
        # Generate a response using the model
        model = genai.GenerativeModel('gemini-1.0-pro-latest')

        response = model.generate_content(message)
        print(response)
        return response['candidates'][0]['content'].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"
    
def validate_iata_code(code):
    known_iata_codes = ["NYC", "LON", "SYD", "LAX", "CDG", "BER", "SFO", "ORD", "JFK", "ATL", "DXB", "HND", "HKG", "SIN", "BOM", "DEL"]
    return code in known_iata_codes


def extract_locations(message):
    from_match = re.search(r'from\s+(\w+)', message)
    to_match = re.search(r'to\s+(\w+)', message)

    from_location = from_match.group(1).upper() if from_match else None
    to_location = to_match.group(1).upper() if to_match else None

    return from_location, to_location

    return from_location, to_location
# Initialize data storage for user context
user_interactions = {}
def is_factual_question(message):
    try:
        model = genai.GenerativeModel('gemini-1.0-pro-latest')
        response = model.generate_content(
            f"Classify the following question as 'factual' or 'non-factual': {message}"
        )
        classification = response.candidates[0].content.parts[0].text.strip().lower()
        return classification == 'factual'
    except Exception as e:
        return False
def extract_location_from_message(message):
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return None


def process_message(user_id, message):
    # Initialize user data if it doesn't exist
    if user_id not in user_interactions:
        user_interactions[user_id] = {'name': '', 'last_place': '', 'last_query': ''}

    user_data = user_interactions[user_id]
    synonym_mapping = {
        "hi": ["hi", "hello", "hey", "hiya"],
        "how are you": ["how are you", "how r u", "how are u", "how r you"],
        "what is your name": ["what is your name", "what's your name", "whats your name", "who are you", "who r u", "who are u", "who r you"],
        "my name is": ["my name is", "i am", "i'm", "this is"],
        "thank you": ["thank you", "thanks", "thank u", "thx"],
        "bye": ["bye", "goodbye", "see you", "bye bye"],
        "what's the weather like": ["what's the weather like", "weather", "how's the weather", "whats the weather like"],
        "tell me a joke": ["tell me a joke", "joke", "make me laugh"],
        "what is the meaning of life": ["what is the meaning of life", "meaning of life", "life meaning"],
        "who is your creator": ["who is your creator", "who made you", "who created you"],
        "what's your favorite color": ["what's your favorite color", "favorite color", "fav color"],
        "set reminder": ["set reminder", "remind me", "i need a reminder", "can you remind me"],
        "remind me": ["remind me", "i need a reminder", "can you remind me"],
        "good morning": ["good morning", "morning", "good day", "greetings", "hi there"],
        "good afternoon": ["good afternoon", "afternoon", "hello afternoon", "good day", "hi afternoon"],
        "good evening": ["good evening", "evening", "hello evening", "hi evening", "greetings evening"],
        "good night": ["good night", "night", "goodnight", "nighty night", "sleep well"],
        "how's your day": ["how's your day", "how is your day", "how has your day been", "how's your day going", "how is your day going"],
        "how old are you": ["how old are you", "what's your age", "what is your age", "how many years old are you", "age"],
        "what's your favorite food": ["what's your favorite food", "favorite food", "what do you like to eat", "preferred food", "favorite cuisine"],
        "do you like music": ["do you like music", "enjoy music", "listen to music", "fond of music", "love music"],
        "what's your favorite movie": ["what's your favorite movie", "favorite movie", "favorite film", "best movie", "preferred movie"],
        "what's your hobby": ["what's your hobby", "hobby", "what do you enjoy doing", "pastime", "favorite activity"],
        "do you have any pets": ["do you have any pets", "any pets", "have pets", "own pets", "got pets"],
        "where are you from": ["where are you from", "where do you come from", "place of origin", "hometown", "location"],
        "can you help me": ["can you help me", "help me", "assist me", "give me a hand", "lend me a hand"],
        "do you like traveling": ["do you like traveling", "enjoy traveling", "love traveling", "fond of traveling", "like to travel"],
        "what's your favorite season": ["what's your favorite season", "favorite season", "best season", "preferred season", "season you like most"],
        "do you have friends": ["do you have friends", "any friends", "got friends", "have friends", "make friends"],
        "do you sleep": ["do you sleep", "need sleep", "require sleep", "take rest", "have rest"],
        "what's your job": ["what's your job", "job", "occupation", "what do you do for a living", "profession"],
        "do you like sports": ["do you like sports", "enjoy sports", "play sports", "fond of sports", "into sports"],
        "what's your favorite book": ["what's your favorite book", "favorite book", "best book", "preferred book", "book you like most"],
        "do you believe in aliens": ["do you believe in aliens", "believe in extraterrestrials", "think aliens exist", "believe in UFOs", "alien believer"],
        
        "do you like art": ["do you like art", "enjoy art", "love art", "fond of art", "appreciate art"],
        "what's your favorite animal": ["what's your favorite animal", "favorite animal", "best animal", "preferred animal", "animal you like most"],
        "do you know any languages": ["do you know any languages", "speak languages", "understand languages", "know multiple languages", "linguistic ability"],
        "what's your favorite song": ["what's your favorite song", "favorite song", "best song", "preferred song", "song you like most"],
        "do you have a family": ["do you have a family", "any family", "have family", "got family", "family members"],
        
        "do you like games": ["do you like games", "enjoy games", "play games", "fond of games", "into games"],
        "what's your favorite drink": ["what's your favorite drink", "favorite drink", "best drink", "preferred drink", "drink you like most"],
        "do you like cooking": ["do you like cooking", "enjoy cooking", "love cooking", "fond of cooking", "like to cook"],
        "what's your favorite quote": ["what's your favorite quote", "favorite quote", "best quote", "preferred quote", "quote you like most"],
        "travel advice": ["travel advice", "travel tips", "travel recommendations", "travel suggestions", "travel pointers"],
    "travel advice": ["travel advice", "travel tips", "travel recommendations", "travel suggestions", "travel pointers"],
    "travel guidance": ["travel guidance", "travel instructions", "travel directions", "travel help", "travel advice"],
    "local laws": ["local laws", "regional laws", "area laws", "city laws", "local regulations"],
    "health and safety tips": ["health and safety tips", "safety advice", "health tips", "safety recommendations", "health guidelines"],
    "cultural etiquette": ["cultural etiquette", "ecultural norms", "cultural practices", "cultural traditions", "cultural manners"],
    "transportation tips": ["transportation tips", "travel tips", "commute tips", "transport advice", "travel recommendations"],
    "shopping tips": ["shopping tips", "shopping advice", "market tips", "shopping suggestions", "shopping recommendations"],
    "money management": ["money management", "financial tips", "budget tips", "money advice", "financial advice"],
    "emergency contacts": ["emergency contacts", "emergency numbers", "emergency info", "emergency details", "emergency services"],
    "internet and connectivity": ["internet and connectivity", "internet tips", "connectivity advice", "Wi-Fi tips", "internet suggestions"]

    }
    normalized_message = message.lower().strip()
    if "flight" in message or ("from" in message and "to" in message):
        response, emotion = process_flight_message(user_id, message)
        return response, emotion

    
    if any(phrase in message for phrase in ["hotel", "stay"]):
        place = extract_location_from_message(message)
        if place:
            return handle_travel_queries(place, message)
    # Handle if the user is providing their name
    if any(phrase in normalized_message for phrase in synonym_mapping["my name is"]):
        name = message.split('is')[-1].strip()  # Extract the name after 'is'
        user_data['name'] = name
        user_interactions[user_id] = user_data  # Save the user data
        return f"Nice to meet you, {name}!", "neutral"

    # Check if the query is already known
    known_response = check_known_query(normalized_message)
    if known_response:
        return known_response, detect_emotion(message)

    # Existing emotion detection and fixed responses logic
    emotion = detect_emotion(message)
    if emotion == "sadness":
        response = "I understand that you're feeling sad. Please know that I'm here for you. Is there anything I can do to help cheer you up?"
        return response, emotion

    if emotion == "anger":
        response = "It seems like you're feeling angry. I'm here to listen if you need to talk about it. What can I do to help you?"
        return response, emotion

    if emotion == "disgust":
        response = "I can see that something has made you feel disgusted. Let's find a way to move past this. How can I assist you?"
        return response, emotion

    if emotion == "fear":
        response = "It sounds like you're feeling afraid. I'm here to support you. Is there anything specific you're worried about?"
        return response, emotion

    if emotion == "shame":
        response = "Feeling ashamed can be really tough. Remember, I'm here to help you without any judgment. How can I support you?"
        return response, emotion

    response_key = None
    for key, variations in synonym_mapping.items():
        if any(variation in normalized_message for variation in variations):
            response_key = key
            break

    if response_key and response_key in fixed_responses:
        response = random.choice(fixed_responses[response_key])
        if '{name}' in response:
            response = response.format(name=user_data.get('name', ''))
        if response_key in ["what is your name", "who are you"]:
            return f"{response} What's your name?", "neutral"
        return response, detect_emotion(message)

    if "flight" in normalized_message:
        user_data['last_query'] = 'flight'
        user_interactions[user_id] = user_data  # Update the user interactions
        return process_flight_message(user_id, message)

    doc = nlp(message)
    place = None
    for ent in doc.ents:
        if ent.label_ == "GPE":
            place = ent.text
            break

    if not place:
        words = message.split()
        for word in words:
            if word.lower() in known_cities:
                place = word
                break

    if place:
        user_data['last_place'] = place
        user_interactions[user_id] = user_data  # Update the user interactions
        travel_response, travel_emotion = handle_travel_queries(place, message)
        return travel_response, travel_emotion
    else:
        last_place = user_data.get('last_place')
        if last_place:
            travel_response, travel_emotion = handle_travel_queries(last_place, message)
            return travel_response, travel_emotion

        # Check if the message is a factual question
        if is_factual_question(normalized_message):
            # If the bot doesn't know the answer, ask Gemini
            gemini_response = get_response(message)
            if gemini_response.lower() != "doesn't know":
                save_to_knowledge_base(message, gemini_response)
                return gemini_response, detect_emotion(message)

        # If Gemini doesn't know, ask the user
        user_last_question[user_id] = message
        return "I'm not sure about that. Can you tell me the answer?", detect_emotion(message)

def process_flight_message(user_id, message):
    user_data = user_interactions.get(user_id, {})
    flight_details = user_data.get('flight_details', {'origin': None, 'destination': None, 'departure_date': None})

    # Extract 'from' and 'to' locations from the message
    from_location, to_location = extract_locations(message)
    if from_location:
        flight_details['origin'] = from_location
    if to_location:
        flight_details['destination'] = to_location
    if not flight_details['departure_date']:
        flight_details['departure_date'] = extract_date(message, "departure date")

    # Check for missing parameters and request them
    if not flight_details['origin']:
        return request_missing_parameter(user_id, 'origin'), 'neutral'
    if not flight_details['destination']:
        return request_missing_parameter(user_id, 'destination'), 'neutral'
    if not flight_details['departure_date']:
        return request_missing_parameter(user_id, 'departure date'), 'neutral'

    # All parameters are present, fetch flight details
    origin = flight_details['origin']
    destination = flight_details['destination']
    departure_date = flight_details['departure_date']

    flight_details_str = get_amadeus_flight_data(origin, destination, departure_date)
    return flight_details_str, 'neutral'


def process_hotel_message(user_id, message):
    user_data = user_interactions.get(user_id, {})
    hotel_details = user_data.get('hotel_details', {'location': None, 'checkin_date': None, 'checkout_date': None})

    # Extract location and dates from the message
    location = extract_location_from_message(message)
    if location:
        hotel_details['location'] = location
    if not hotel_details['checkin_date']:
        hotel_details['checkin_date'] = extract_date(message, "check-in date")
    if not hotel_details['checkout_date']:
        hotel_details['checkout_date'] = extract_date(message, "check-out date")

    # Check for missing parameters and request them
    if not hotel_details['location']:
        return request_missing_parameter(user_id, 'location'), 'neutral'
    if not hotel_details['checkin_date']:
        return request_missing_parameter(user_id, 'check-in date'), 'neutral'
    if not hotel_details['checkout_date']:
        return request_missing_parameter(user_id, 'check-out date'), 'neutral'

    # All parameters are present, fetch hotel details
    location = hotel_details['location']
    checkin_date = hotel_details['checkin_date']
    checkout_date = hotel_details['checkout_date']

    hotel_details_str = get_cheapest_hotels(location, checkin_date, checkout_date)
    return hotel_details_str, 'neutral'


def get_cheapest_flights(origin, destination, departure_date):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=1,
            max=3,
            sort='price'
        )
        flights = [
            f"Flight {offer['itineraries'][0]['segments'][0]['carrierCode']}{offer['itineraries'][0]['segments'][0]['number']} "
            f"from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} "
            f"to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} "
            f"at {offer['itineraries'][0]['segments'][0]['departure']['at']} for ${offer['price']['total']}"
            for offer in response.data
        ]
        if flights:
            return f"The cheapest flights are: {', '.join(flights)}"
        else:
            return "No flights found."
    except ResponseError as error:
        return f"Error fetching flights: {str(error)}"

def get_cheapest_hotels(location, checkin_date, checkout_date):
    # Placeholder implementation, replace with actual API call
    return f"Cheapest hotels in {location} from {checkin_date} to {checkout_date}."


def extract_date(message, date_type):
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', message)
    if date_match:
        return date_match.group(0)
    return None

def get_amadeus_flight_data(origin, destination, departure_date):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=1
        )
        flights = [
            f"Flight {offer['itineraries'][0]['segments'][0]['carrierCode']}{offer['itineraries'][0]['segments'][0]['number']} "
            f"from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} "
            f"to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} "
            f"at {offer['itineraries'][0]['segments'][0]['departure']['at']}"
            for offer in response.data
        ]
        if flights:
            return f"The available flights are: {', '.join(flights)}"
        else:
            return "No flights found."
    except ResponseError as error:
        return f"Error fetching flights: {str(error)}"

def get_flight_schedule(origin, destination, departure_date):
    try:
        response = amadeus.schedule.flights.get(origin=origin, destination=destination, departureDate=departure_date)
        flights = [f"Flight {flight['carrierCode']}{flight['flightNumber']} departs at {flight['departure']['at']} and arrives at {flight['arrival']['at']}." for flight in response.data]
        return flights if flights else "No flights found"
    except ResponseError as error:
        return f"Error fetching flight schedule: {str(error)}"

def get_flight_status(flight_number, date):
    try:
        response = amadeus.schedule.flight_status.get(flight_number=flight_number, scheduledDepartureDate=date)
        statuses = [f"Flight {status['carrierCode']}{status['flightNumber']} is currently {status['flightStatus']}." for status in response.data]
        return statuses if statuses else "No status found"
    except ResponseError as error:
        return f"Error fetching flight status: {str(error)}"

def get_most_popular_flights(origin, destination, departure_date):
    try:
        response = amadeus.analytics.itinerary_price_metrics.get(origin=origin, destination=destination, departureDate=departure_date, include="most-popular")
        flights = [f"Most popular flight: {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} on {offer['itineraries'][0]['segments'][0]['departure']['at']} for ${offer['price']['total']}." for offer in response.data]
        return flights if flights else "No popular flights found"
    except ResponseError as error:
        return f"Error fetching most popular flights: {str(error)}"
    
    

def handle_name_response(user_id, message):
    if 'my name is' in message.lower():
        name = message.split('my name is')[-1].strip()
        user_interactions[user_id]['name'] = name
        return f"Nice to meet you, {name}!", "neutral"
    return None, None




def extract_destination_from_message(message):
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return None


def request_missing_parameter(user_id, parameter_name):
    user_last_question[user_id] = parameter_name
    return f"Please provide the {parameter_name}."


def extract_location(message, location_type):
    doc = nlp(message)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text.upper()  # Ensure the location is in uppercase for IATA codes
    return None

def extract_date(message, date_type):
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', message)
    if date_match:
        return date_match.group(0)
    return None


def detect_emotion(message):
    # Predict the emotion using the emotion classifier
    emotion = emotion_classifier.predict([message])[0]
    allowed_emotions = ["sadness", "fear", "anger", "disgust", "shame"]
    if emotion not in allowed_emotions:
        emotion = "neutral"
    return emotion
def get_openai_response(message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.RateLimitError:
        return "I'm currently experiencing high demand. Please try again later."
    except openai.error.OpenAIError as e:
        return f"Error generating response: {str(e)}"


def get_response(message):
    try:
        # Generate a response using the Gemini model with a modified prompt
        model = genai.GenerativeModel('gemini-1.0-pro-latest')
        response = model.generate_content(
            f"Provide a concise answer to the following question. If you don't know the answer, simply reply with 'doesn't know': {message}"
        )

        # Extract the text content from the first candidate's parts
        text = response.candidates[0].content.parts[0].text.strip()

        # Return the response
        return text
    except Exception as e:
        return f"Error generating response: {str(e)}"

    



def handle_travel_queries(place, message):

    query_types = {
      "places to visit": get_google_places_data,
      "top attractions": get_google_places_data,
       "ports": get_google_places_data,
        "best hotels": get_google_places_data,
        "popular restaurants": get_google_places_data,
        "famous landmarks": get_google_places_data,
        "shopping centers": get_google_places_data,
        "famous museums": get_google_places_data,
        "parks and nature": get_google_places_data,
        "religious sites": get_google_places_data,
        "nightlife": get_google_places_data,
        "beaches": get_google_places_data,
       "hiking trails": get_google_places_data,
        "tourist attractions": get_google_places_data,
        "nightlife and entertainment": get_google_places_data,
        "pharmacies": get_google_places_data,
        "ATMs": get_google_places_data,
        "currency exchange offices": get_google_places_data,
        "local markets": get_google_places_data,
        "food courts": get_google_places_data,
        "street food stalls": get_google_places_data,
        "movie theaters": get_google_places_data,
        "theaters": get_google_places_data,
        "art galleries": get_google_places_data,
        "spa and wellness centers": get_google_places_data,
        "gyms and fitness centers": get_google_places_data,
        "tourist information centers": get_google_places_data,
        "libraries": get_google_places_data,
        "universities and colleges": get_google_places_data,
        "historical sites": get_google_places_data,
        "botanical gardens": get_google_places_data,
        "zoos and aquariums": get_google_places_data,
        "amusement parks": get_google_places_data,
        "water parks": get_google_places_data,
        "ski resorts": get_google_places_data,
        "marinas": get_google_places_data,
        "tourist offices": get_google_places_data,
        "car rental services": get_google_places_data,
        "bike rental services": get_google_places_data,
        "scooter rental services": get_google_places_data,
        "taxis": get_google_places_data,
        "ride-sharing services": get_google_places_data,
        "cultural heritage sites": get_google_places_data,
        "malls": get_google_places_data,
        "flea markets": get_google_places_data,
        "outdoor activities": get_google_places_data,
        "indoor activities": get_google_places_data,
        "wildlife sanctuaries": get_google_places_data,
        "resorts": get_google_places_data,
        "bed and breakfasts": get_google_places_data,
        "hostels": get_google_places_data,
        "vacation rentals": get_google_places_data,
        "public parks": get_google_places_data,
        "gardens": get_google_places_data,
        "adventure sports": get_google_places_data,
        "historical landmarks": get_google_places_data,
        "museums": get_google_places_data,
        "galleries": get_google_places_data,
        "cafes": get_google_places_data,
        "bars": get_google_places_data,
        "pubs": get_google_places_data,
        "restaurants": get_google_places_data,
        "food trucks": get_google_places_data,
        "nightclubs": get_google_places_data,
        "breweries": get_google_places_data,
        "wineries": get_google_places_data,
        "distilleries": get_google_places_data,
        "concert venues": get_google_places_data,
        "music festivals": get_google_places_data,
        "outdoor concerts": get_google_places_data,
        "theater performances": get_google_places_data,
        "opera houses": get_google_places_data,
        "comedy clubs": get_google_places_data,
        "jazz clubs": get_google_places_data,
        "blues clubs": get_google_places_data,
        "beach bars": get_google_places_data,
        "sports bars": get_google_places_data,
        "cocktail bars": get_google_places_data,
        "dive bars": get_google_places_data,
        "karaoke bars": get_google_places_data,
        "beer gardens": get_google_places_data,
        "wine bars": get_google_places_data,
        "speakeasies": get_google_places_data,
        "cider houses": get_google_places_data,
        "beer halls": get_google_places_data,
        "gastropubs": get_google_places_data,
        "dessert shops": get_google_places_data,
        "bakeries": get_google_places_data,
        "ice cream parlors": get_google_places_data,
        "candy stores": get_google_places_data,
        "chocolate shops": get_google_places_data,
        "fudge shops": get_google_places_data,
        "donut shops": get_google_places_data,
        "cupcake shops": get_google_places_data,
        "macaron shops": get_google_places_data,
        "bubble tea shops": get_google_places_data,
        "smoothie bars": get_google_places_data,
        "juice bars": get_google_places_data,
        "tea houses": get_google_places_data,
        "coffee shops": get_google_places_data,
        "bookstores": get_google_places_data,
        "record stores": get_google_places_data,
        "gift shops": get_google_places_data,
        "souvenir shops": get_google_places_data,
        "thrift stores": get_google_places_data,
        "vintage stores": get_google_places_data,
        "antique stores": get_google_places_data,
        "furniture stores": get_google_places_data,
        "home decor stores": get_google_places_data,
        "clothing stores": get_google_places_data,
        "shoe stores": get_google_places_data,
        "jewelry stores": get_google_places_data,
        "watch stores": get_google_places_data,
        "opticians": get_google_places_data,
        "toy stores": get_google_places_data,
        "hobby shops": get_google_places_data,
        "craft stores": get_google_places_data,
        "art supply stores": get_google_places_data,
        "stationery stores": get_google_places_data,
        "hardware stores": get_google_places_data,
        "pet stores": get_google_places_data,
        "aquarium stores": get_google_places_data,
        "grocery stores": get_google_places_data,
        "supermarkets": get_google_places_data,
        "convenience stores": get_google_places_data,
        "health food stores": get_google_places_data,
        "organic food stores": get_google_places_data,
        "butcher shops": get_google_places_data,
        "fish markets": get_google_places_data,
        "farmers markets": get_google_places_data,
        "cheese shops": get_google_places_data,
        "wine shops": get_google_places_data,
        "liquor stores": get_google_places_data,
        "tobacco shops": get_google_places_data,
        "chocolatiers": get_google_places_data,
        "cigar shops": get_google_places_data,
        "eateries": get_google_places_data,
        "tapas bars": get_google_places_data,
        "pizza places": get_google_places_data,
        "burger joints": get_google_places_data,
        "steakhouses": get_google_places_data,
        "seafood restaurants": get_google_places_data,
        "sushi restaurants": get_google_places_data,
        "ramen restaurants": get_google_places_data,
        "dim sum restaurants": get_google_places_data,
        "noodle houses": get_google_places_data,
        "food courts": get_google_places_data,
        "food stalls": get_google_places_data,
        "dessert bars": get_google_places_data,
        "patisseries": get_google_places_data,
        "boulangeries": get_google_places_data,
        "gelaterias": get_google_places_data,
        "chocolaterias": get_google_places_data,
        "creperies": get_google_places_data,
        "pizzerias": get_google_places_data,
        "taco places": get_google_places_data,
        "burrito places": get_google_places_data,
        "BBQ restaurants": get_google_places_data,
        "grill houses": get_google_places_data,
        "rooftop bars": get_google_places_data,
        "speakeasy bars": get_google_places_data,
        "beach clubs": get_google_places_data,
        "pool clubs": get_google_places_data,
        "gentlemen's clubs": get_google_places_data,
        "nightclubs": get_google_places_data,
        "discotheques": get_google_places_data,
        "dive bars": get_google_places_data,
        "gay bars": get_google_places_data,
        "drag bars": get_google_places_data,
        "blues bars": get_google_places_data,
        "jazz bars": get_google_places_data,
        "comedy clubs": get_google_places_data,
        "magic clubs": get_google_places_data,
        "karaoke clubs": get_google_places_data,
        "dance clubs": get_google_places_data,
        "swing dance clubs": get_google_places_data,
        "tango clubs": get_google_places_data,
        "salsa clubs": get_google_places_data,
        "sports clubs": get_google_places_data,
        "cricket clubs": get_google_places_data,
        "rugby clubs": get_google_places_data,
        "football clubs": get_google_places_data,
        "soccer clubs": get_google_places_data,
        "tennis clubs": get_google_places_data,
        "golf clubs": get_google_places_data,
        "polo clubs": get_google_places_data,
        "fishing clubs": get_google_places_data,
        "sailing clubs": get_google_places_data,
        "yacht clubs": get_google_places_data,
        "country clubs": get_google_places_data,
        "motorcycle clubs": get_google_places_data,
        "car clubs": get_google_places_data,
        "surf clubs": get_google_places_data,
        "dive clubs": get_google_places_data,
        "ski clubs": get_google_places_data,
        "snowboard clubs": get_google_places_data,
        "mountain biking clubs": get_google_places_data,
        "trail running clubs": get_google_places_data,
        "hiking clubs": get_google_places_data,
        "rock climbing clubs": get_google_places_data,
        "bouldering clubs": get_google_places_data,
        "gymnastics clubs": get_google_places_data,
        "yoga clubs": get_google_places_data,
        "pilates clubs": get_google_places_data,
        "crossfit clubs": get_google_places_data,
        "bootcamp clubs": get_google_places_data,
        "spinning clubs": get_google_places_data,
        "aerobics clubs": get_google_places_data,
        "dance clubs": get_google_places_data,
        "martial arts clubs": get_google_places_data,
        "boxing clubs": get_google_places_data,
        "kickboxing clubs": get_google_places_data,
        "muay thai clubs": get_google_places_data,
        "bjj clubs": get_google_places_data,
        "judo clubs": get_google_places_data,
        "karate clubs": get_google_places_data,
        "taekwondo clubs": get_google_places_data,
        "krav maga clubs": get_google_places_data,
        "self-defense clubs": get_google_places_data,
        "archery clubs": get_google_places_data,
        "shooting clubs": get_google_places_data,
        "fencing clubs": get_google_places_data,
        "esports clubs": get_google_places_data,
        "gaming clubs": get_google_places_data,
        "LAN clubs": get_google_places_data,
        "arcades": get_google_places_data,
        "billiard halls": get_google_places_data,
        "bowling alleys": get_google_places_data,
        "skateparks": get_google_places_data,
        "bmx parks": get_google_places_data,
        "water parks": get_google_places_data,
        "theme parks": get_google_places_data,
        "amusement parks": get_google_places_data,
        "carnivals": get_google_places_data,
        "festivals": get_google_places_data,
        "fairs": get_google_places_data,
        "parades": get_google_places_data,
        "street parties": get_google_places_data,
        "block parties": get_google_places_data,
        "raves": get_google_places_data,
        "burning man events": get_google_places_data,
        "tournaments": get_google_places_data,
        "competitions": get_google_places_data,
        "marathons": get_google_places_data,
        "triathlons": get_google_places_data,
        "ironman events": get_google_places_data,
        "fun runs": get_google_places_data,
        "color runs": get_google_places_data,
        "obstacle races": get_google_places_data,
        "tough mudder events": get_google_places_data,
        "spartan races": get_google_places_data,
        "ninja warrior events": get_google_places_data,
        "crossfit games": get_google_places_data,
        "olympic events": get_google_places_data,
        "world cup events": get_google_places_data,
        "championships": get_google_places_data,
        "exhibitions": get_google_places_data,
        "trade shows": get_google_places_data,
        "conferences": get_google_places_data,
        "summits": get_google_places_data,
        "conventions": get_google_places_data,
        "meetups": get_google_places_data,
        "networking events": get_google_places_data,
        "workshops": get_google_places_data,
        "seminars": get_google_places_data,
        "lectures": get_google_places_data,
        "courses": get_google_places_data,
        "bootcamps": get_google_places_data,
        "masterclasses": get_google_places_data,
        "retreats": get_google_places_data,
        "hackathons": get_google_places_data,
        "game jams": get_google_places_data,
        "start-up weekends": get_google_places_data,
        "pitch nights": get_google_places_data,
        "demo days": get_google_places_data,
        "job fairs": get_google_places_data,
        "career fairs": get_google_places_data,
        "expo events": get_google_places_data,
        "markets": get_google_places_data,
        "farmers markets": get_google_places_data,
        "flea markets": get_google_places_data,
        "antique markets": get_google_places_data,
        "craft fairs": get_google_places_data,
        "artisan markets": get_google_places_data,
        "vintage markets": get_google_places_data,
        "food markets": get_google_places_data,
        "street markets": get_google_places_data,
        "night markets": get_google_places_data,
        "beach markets": get_google_places_data,
        "city markets": get_google_places_data,
        "village markets": get_google_places_data,
        "county markets": get_google_places_data,
        "regional markets": get_google_places_data,
        "local markets": get_google_places_data,
        "seasonal markets": get_google_places_data,
        "holiday markets": get_google_places_data,
        "festival markets": get_google_places_data,
        "harvest markets": get_google_places_data,
        "craft markets": get_google_places_data,
        "artists markets": get_google_places_data,
        "makers markets": get_google_places_data,
        "artisan fairs": get_google_places_data,
        "designer fairs": get_google_places_data,
        "fashion fairs": get_google_places_data,
        "vintage fairs": get_google_places_data,
        "flea fairs": get_google_places_data,
        "antique fairs": get_google_places_data,
        "car boot sales": get_google_places_data,
        "swap meets": get_google_places_data,
        "garage sales": get_google_places_data,
        "yard sales": get_google_places_data,
        "estate sales": get_google_places_data,
        "plant sales": get_google_places_data,
        "book sales": get_google_places_data,
        "vinyl sales": get_google_places_data,
        "art sales": get_google_places_data,
        "craft sales": get_google_places_data,
        "charity sales": get_google_places_data,
        "church sales": get_google_places_data,
        "school sales": get_google_places_data,
        "college sales": get_google_places_data,
        "university sales": get_google_places_data,
        "library sales": get_google_places_data,
        "community sales": get_google_places_data,
        "neighborhood sales": get_google_places_data,
        "town sales": get_google_places_data,
        "city sales": get_google_places_data,
        "county sales": get_google_places_data,
        "regional sales": get_google_places_data,
        "state sales": get_google_places_data,
        "national sales": get_google_places_data,
        "international sales": get_google_places_data,
        "online sales": get_google_places_data,
        "e-commerce sales": get_google_places_data,
        "marketplace sales": get_google_places_data,
        "shopping sales": get_google_places_data,
        "wholesale sales": get_google_places_data,
        "retail sales": get_google_places_data,
        "warehouse sales": get_google_places_data,
        "clearance sales": get_google_places_data,
        "flash sales": get_google_places_data,
        "sample sales": get_google_places_data,
        "auction sales": get_google_places_data,
        "live auction sales": get_google_places_data,
        "silent auction sales": get_google_places_data,
        "charity auction sales": get_google_places_data,
        "fundraiser auction sales": get_google_places_data,
        "benefit auction sales": get_google_places_data,
        "event auction sales": get_google_places_data,
        "corporate auction sales": get_google_places_data,
        "government auction sales": get_google_places_data,
        "municipal auction sales": get_google_places_data,
        "state auction sales": get_google_places_data,
        "national auction sales": get_google_places_data,
        "international auction sales": get_google_places_data,
        "e-auction sales": get_google_places_data,
        "online auction sales": get_google_places_data,
        "live online auction sales": get_google_places_data,
        "silent online auction sales": get_google_places_data,
        "charity online auction sales": get_google_places_data,
        "fundraiser online auction sales": get_google_places_data,
        "benefit online auction sales": get_google_places_data,
        "event online auction sales": get_google_places_data,
        "corporate online auction sales": get_google_places_data,
        "government online auction sales": get_google_places_data,
        "municipal online auction sales": get_google_places_data,
        "state online auction sales": get_google_places_data,
        "national online auction sales": get_google_places_data,
        "international online auction sales": get_google_places_data,
        "e-commerce online auction sales": get_google_places_data,
        "marketplace online auction sales": get_google_places_data,
        "shopping online auction sales": get_google_places_data,
        "wholesale online auction sales": get_google_places_data,
        "retail online auction sales": get_google_places_data,
        "warehouse online auction sales": get_google_places_data,
        "clearance online auction sales": get_google_places_data,
        "flash online auction sales": get_google_places_data,
        "sample online auction sales": get_google_places_data,
        "craft sales": get_google_places_data,
        "fair sales": get_google_places_data,
        "market sales": get_google_places_data,
        "bazaar sales": get_google_places_data,
        "expo sales": get_google_places_data,
        "trade show sales": get_google_places_data,
        "conference sales": get_google_places_data,
        "summit sales": get_google_places_data,
        "convention sales": get_google_places_data,
        "meetup sales": get_google_places_data,
        "networking event sales": get_google_places_data,
        "workshop sales": get_google_places_data,
        "seminar sales": get_google_places_data,
        "lecture sales": get_google_places_data,
        "course sales": get_google_places_data,
        "bootcamp sales": get_google_places_data,
        "masterclass sales": get_google_places_data,
        "retreat sales": get_google_places_data,
        "hackathon sales": get_google_places_data,
        "game jam sales": get_google_places_data,
        "start-up weekend sales": get_google_places_data,
        "pitch night sales": get_google_places_data,
        "demo day sales": get_google_places_data,
        "job fair sales": get_google_places_data,
        "career fair sales": get_google_places_data,
        "expo sales": get_google_places_data,
        "exhibition sales": get_google_places_data,
        "art show sales": get_google_places_data,
        "gallery sales": get_google_places_data,
        "museum sales": get_google_places_data,
        "tourist sales": get_google_places_data,
        "adventure sales": get_google_places_data,
        "leisure sales": get_google_places_data,
        "recreation sales": get_google_places_data,
        "activity sales": get_google_places_data,
        "event sales": get_google_places_data,
        "show sales": get_google_places_data,
        "performance sales": get_google_places_data,
        "concert sales": get_google_places_data,
        "music sales": get_google_places_data,
        "theater sales": get_google_places_data,
        "play sales": get_google_places_data,
        "dance sales": get_google_places_data,
        "opera sales": get_google_places_data,
        "comedy sales": get_google_places_data,
        "magic sales": get_google_places_data,
        "showcase sales": get_google_places_data,
        "exhibit sales": get_google_places_data,
        "display sales": get_google_places_data,
        "installation sales": get_google_places_data,
        "demonstration sales": get_google_places_data,
        "presentation sales": get_google_places_data,
        "lecture sales": get_google_places_data,
        "talk sales": get_google_places_data,
        "seminar sales": get_google_places_data,
        "workshop sales": get_google_places_data,
        "masterclass sales": get_google_places_data,
        "retreat sales": get_google_places_data,
        "bootcamp sales": get_google_places_data,
        "course sales": get_google_places_data,
        "class sales": get_google_places_data,
        "lesson sales": get_google_places_data,
        "session sales": get_google_places_data,
        "clinic sales": get_google_places_data,
        "program sales": get_google_places_data,
        "event sales": get_google_places_data,
        "festival sales": get_google_places_data,
        "fair sales": get_google_places_data,
        "carnival sales": get_google_places_data,
        "bazaar sales": get_google_places_data,
        "market sales": get_google_places_data,
        "flea market sales": get_google_places_data,
        "swap meet sales": get_google_places_data,
        "weather": get_weather_data,
        "time": get_time_data,
        "date": get_date_data,
        "how to get to": get_transport_data,
        "safety tips": get_safety_tips,
        "local language": get_country_info,
        "currency": get_country_info,
        "visa requirements": get_country_info,
        "emergency contacts": get_emergency_contacts,
        "public transport": get_public_transport_info,
        "internet availability": get_internet_availability,
        "local cuisine": get_local_cuisine,
        "cultural events": get_cultural_events,
        "local laws": get_country_info,
        "hospital locations": get_hospital_locations,
        "festivals": get_festivals,
        "language phrases": get_language_phrases,
        "public holidays": get_public_holidays,
        "tips for tourists": get_tips_for_tourists,
        "transport options": get_transport_options,
        "history": get_history,
        "president": get_president_info,

       
        "flight schedule": get_flight_schedule,
        "flight status": get_flight_status,
        "available flights": get_available_flights,
        "cheapest flights": get_cheapest_flights,
        "most popular flights": get_most_popular_flights,
        "flight prices": get_flight_prices,
        "direct flights": get_direct_flights,
        "non-stop flights": get_non_stop_flights,
        "connecting flights": get_connecting_flights,
        "layover information": get_layover_information,
        "flight duration": get_flight_duration,
        "arrival time": get_arrival_time,
         "airports": get_airports,
        "departure time": get_departure_time,
        "airline information": get_airline_information,
        "airline ratings": get_airline_ratings,
        "baggage policies": get_baggage_policies,
        "seat availability": get_seat_availability,
        "in-flight services": get_in_flight_services,
        "airport information": get_airport_information,
        "airport lounges": get_airport_lounges,
        "flight amenities": get_flight_amenities,
        "flight delays": get_flight_delays,
        "cancellation policies": get_cancellation_policies,
        "flight promotions": get_flight_promotions,
        "frequent flyer programs": get_frequent_flyer_programs
    }

    for query_type, func in query_types.items():
        if query_type in message:
            details = func(place, query_type)
            return f"The best {query_type} in {place} are: {details}.", detect_emotion(message)
    
    # If the query doesn't match, fall back to factual question check
    normalized_message = message.lower().strip()
    if is_factual_question(normalized_message):
        gemini_response = get_response(message)
        if gemini_response.lower() != "doesn't know":
            save_to_knowledge_base(message, gemini_response)
            return gemini_response, detect_emotion(message)
    
    user_last_question["user123"] = message
    return "I'm not sure about that. Can you tell me the answer?", detect_emotion(message)

   

def get_google_places_data(place, query_type):
    # Placeholder - replace with actual API call
    return f"Data for {query_type} in {place}"






def get_openai_response(message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_google_places_data(place, query_type):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        'query': f"{query_type} in {place}",
        'key': GOOGLE_PLACES_API_KEY  # Ensure your API key is set correctly
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "OK":
            results = data.get('results', [])
            places = [result['name'] for result in results]
            return ", ".join(places) if places else "No places found"
        else:
            return "No places found"
    else:
        return "Information not available"




def validate_iata_code(code):
    # List of known IATA codes (this list should be more comprehensive in a real scenario)
    known_iata_codes = ["NYC", "LAX", "SYD", "LON", "CDG", "BER", "SFO", "ORD", "JFK", "ATL", "DXB", "HND", "HKG", "SIN", "BOM", "DEL"]
    return code in known_iata_codes


# Example usage
def get_weather_data(place, query_type):
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': place,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        weather = data['weather'][0]['description']
        return f"The current weather in {place} is {weather}."
    return "Weather information not available."

def get_time_data(place, query_type):
    url = f"http://worldtimeapi.org/api/timezone/{place}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        time = data['datetime']
        return f"The current time in {place} is {time}."
    return "Time information not available."

def get_date_data(place, query_type):
    url = f"http://worldtimeapi.org/api/timezone/{place}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        date = data['datetime']
        return f"Today's date in {place} is {date}."
    return "Date information not available."

def get_transport_data(place, query_type):
    details = [
        "By air: Available through major airports.",
        "By train: Well-connected railway stations.",
        "By road: Bus and taxi services.",
        "By sea: Available through major ports."
    ]
    return f"Here are some transport options in {place}: " + ", ".join(details)

def get_safety_tips(place, query_type):
    tips = [
        "Keep your belongings safe.",
        "Avoid less crowded areas at night.",
        "Stay aware of your surroundings.",
        "Carry a copy of your identification."
    ]
    return f"Here are some safety tips for {place}: " + ", ".join(tips)

def get_country_info(place, query_type):
    url = f"https://restcountries.com/v3.1/name/{place}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()[0]
        if query_type == "local language":
            languages = data['languages']
            return f"The local languages in {place} are: " + ", ".join(languages.values())
        elif query_type == "currency":
            currencies = data['currencies']
            return f"The currencies used in {place} are: " + ", ".join(currencies.keys())
        elif query_type == "visa requirements":
            return "Please check the official government website for visa requirements specific to your nationality."
        elif query_type == "local laws":
            return "Important local laws include regulations on behavior, dress code, and public conduct."
    return "Country information not available."

def get_emergency_contacts(place, query_type):
    contacts = [
        "Police: 100",
        "Ambulance: 101",
        "Fire: 102"
    ]
    return f"Here are some emergency contacts for {place}: " + ", ".join(contacts)

def get_public_transport_info(place, query_type):
    details = [
        "Buses",
        "Trains",
        "Trams",
        "Subways"
    ]
    return f"Here are some public transport options in {place}: " + ", ".join(details)

def get_internet_availability(place, query_type):
    details = "Wi-Fi is widely available in hotels, cafes, and public areas."
    return f"Internet availability in {place}: " + details

def get_local_cuisine(place, query_type):
    details = "The local cuisine includes various traditional dishes specific to the region."
    return f"Here are some examples of local cuisine in {place}: " + details

def get_cultural_events(place, query_type):
    details = "Cultural events include festivals, parades, and local celebrations."
    return f"Here are some cultural events in {place}: " + details

def get_hospital_locations(place, query_type):
    details = "Hospitals are located in major cities and towns."
    return f"Here are some hospital locations in {place}: " + details

def get_festivals(place, query_type):
    details = "Festivals celebrated include cultural, religious, and national events."
    return f"Here are some festivals celebrated in {place}: " + details

def get_language_phrases(place, query_type):
    phrases = [
        "Hello: (Local language greeting)",
        "Thank you: (Local language thanks)",
        "Please: (Local language please)"
    ]
    return f"Here are some useful language phrases for {place}: " + ", ".join(phrases)

def get_public_holidays(place, query_type):
    details = "Public holidays include national and regional celebrations."
    return f"Here are some public holidays in {place}: " + details

def get_tips_for_tourists(place, query_type):
    tips = [
        "Learn a few words of the local language.",
        "Respect local customs and traditions.",
        "Stay hydrated and take care of your health."
    ]
    return f"Here are some tips for tourists in {place}: " + ", ".join(tips)

def get_transport_options(place, query_type):
    options = [
        "Car rentals",
        "Bicycle rentals",
        "Scooter rentals",
        "Ride-sharing services"
    ]
    return f"Here are some transport options in {place}: " + ", ".join(options)

def get_history(place, query_type):
    url = f"{WIKIPEDIA_API_URL}{place}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return f"Here's some history about {place}: " + data.get('extract', 'History information not available.')
    return "History information not available."

def get_president_info(place, query_type):
    url = f"https://restcountries.com/v3.1/name/{place}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()[0]
        return f"The current president of {place} is: " + data.get('government', {}).get('leader', 'President information not available.')
    return "President information not available."

def get_flights(from_location, to_location, departure_date):
    try:
        response = amadeus.shopping.flight_offers_search.get(
            origin=from_location,
            destination=to_location,
            departureDate=departure_date
        )
        flights = [
            f"Flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} "
            f"to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} at "
            f"{offer['itineraries'][0]['segments'][0]['departure']['at']}"
            for offer in response.data
        ]
        return ", ".join(flights) if flights else "No flights found"
    except ResponseError as error:
        return str(error)



def map_location_to_iata(location):
    city_to_iata = {
        "sydney": "SYD",
        "colombo": "CMB",
        "kandy": "KDZ",
        "galle": "GLL",
        # Add more mappings as needed
    }
    return city_to_iata.get(location.lower())


def get_flight_schedule(place, query_type):
    try:
        response = amadeus.schedule.flights.get(origin="SYD", destination=place, departureDate="2023-06-01")
        data = response.data
        flights = [f"Flight {flight['carrierCode']}{flight['flightNumber']} departs at {flight['departure']['at']} and arrives at {flight['arrival']['at']}." for flight in data]
        return f"Here are some flight schedules for {place}: " + ", ".join(flights)
    except ResponseError as error:
        return str(error)

def get_flight_status(place, query_type):
    try:
        response = amadeus.schedule.flights.get(origin="SYD", destination=place, departureDate="2023-06-01")
        data = response.data
        statuses = [f"Flight {flight['carrierCode']}{flight['flightNumber']} is currently {flight['flightStatus']}." for flight in data]
        return f"Here are some flight statuses for {place}: " + ", ".join(statuses)
    except ResponseError as error:
        return str(error)



def get_most_popular_flights(place, query_type):
    try:
        response = amadeus.analytics.itinerary_price_metrics.get(origin="SYD", destination=place, departureDate="2023-06-01", include="most-popular")
        data = response.data
        flights = [f"Most popular flight: {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} on {offer['itineraries'][0]['segments'][0]['departure']['at']} for ${offer['price']['total']}." for offer in data]
        return f"Here are the most popular flights to {place}: " + ", ".join(flights)
    except ResponseError as error:
        return str(error)

def get_cheapest_flights(place, query_type):
    url = "https://sky-scanner3.p.rapidapi.com/flights/search-one-way"
    querystring = {
        "fromEntityId": "NYC",
        "toEntityId": place,
        "cabinClass": "economy",
        "currency": "USD",
        "countryCode": "US",
        "market": "en-US"
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "sky-scanner3.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code == 200:
        data = response.json()
        flights = [
            f"Flight from {flight['legs'][0]['origin']['displayCode']} to {flight['legs'][0]['destination']['displayCode']} on {flight['legs'][0]['departure']} for ${flight['price']['amount']}."
            for flight in data['itineraries']['buckets'][0]['items']
        ]
        return ", ".join(flights)
    else:
        return "Error retrieving flights"

def get_available_flights(place, query_type):
    url = "https://sky-scanner3.p.rapidapi.com/flights/search-one-way"
    querystring = {
        "fromEntityId": "NYC",
        "toEntityId": place,
        "cabinClass": "economy",
        "currency": "USD",
        "countryCode": "US",
        "market": "en-US"
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "sky-scanner3.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code == 200:
        data = response.json()
        flights = [
            f"Flight from {flight['legs'][0]['origin']['displayCode']} to {flight['legs'][0]['destination']['displayCode']} on {flight['legs'][0]['departure']} for ${flight['price']['amount']}."
            for flight in data['itineraries']['buckets'][0]['items']
        ]
        return ", ".join(flights)
    else:
        return "Error retrieving flights"

def get_airports(city, query_type):
    url = "https://sky-scanner3.p.rapidapi.com/flights/airports"
    headers = {
        "x-rapidapi-key": "eaf2d094admsh6a12bb1ea9b72a6p1245f9jsn9d1d124fd42b",
        "x-rapidapi-host": "sky-scanner3.p.rapidapi.com"
    }

    params = {"query": city}

    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        airports = [
            f"Airport: {airport['name']} ({airport['iata_code']}) in {airport['city_name']}, {airport['country_code']}."
            for airport in data.get('airports', [])
        ]
        return ", ".join(airports) if airports else "No airports found."
    else:
        return f"Error: {response.status_code} - {response.reason}"
def get_flight_prices(place, query_type):
    url = "https://sky-scanner3.p.rapidapi.com/flights/search-one-way"
    querystring = {
        "fromEntityId": "NYC",
        "toEntityId": place,
        "cabinClass": "economy",
        "currency": "USD",
        "countryCode": "US",
        "market": "en-US"
    }
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "sky-scanner3.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code == 200:
        data = response.json()
        prices = [
            f"Flight from {flight['legs'][0]['origin']['displayCode']} to {flight['legs'][0]['destination']['displayCode']} on {flight['legs'][0]['departure']} for ${flight['price']['amount']}."
            for flight in data['itineraries']['buckets'][0]['items']
        ]
        return ", ".join(prices)
    else:
        return "Error retrieving flights"

def get_direct_flights(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1, nonStop="true")
        data = response.data
        flights = [f"Direct flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} on {offer['itineraries'][0]['segments'][0]['departure']['at']} for ${offer['price']['total']}." for offer in data]
        return f"Here are some direct flights to {place}: " + ", ".join(flights)
    except ResponseError as error:
        return str(error)

def get_non_stop_flights(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1, nonStop="true")
        data = response.data
        flights = [f"Non-stop flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} on {offer['itineraries'][0]['segments'][0]['departure']['at']} for ${offer['price']['total']}." for offer in data]
        return f"Here are some non-stop flights to {place}: " + ", ".join(flights)
    except ResponseError as error:
        return str(error)

def get_connecting_flights(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        flights = [f"Connecting flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} via {offer['itineraries'][0]['segments'][1]['departure']['iataCode']} on {offer['itineraries'][0]['segments'][0]['departure']['at']} for ${offer['price']['total']}." for offer in data if len(offer['itineraries'][0]['segments']) > 1]
        return f"Here are some connecting flights to {place}: " + ", ".join(flights)
    except ResponseError as error:
        return str(error)

def get_layover_information(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        layovers = [f"Layover in {offer['itineraries'][0]['segments'][1]['departure']['iataCode']} for {offer['itineraries'][0]['segments'][1]['duration']}." for offer in data if len(offer['itineraries'][0]['segments']) > 1]
        return f"Here is some layover information for flights to {place}: " + ", ".join(layovers)
    except ResponseError as error:
        return str(error)

def get_flight_duration(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        durations = [f"Flight duration from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} is {offer['itineraries'][0]['duration']}." for offer in data]
        return f"Here are the flight durations to {place}: " + ", ".join(durations)
    except ResponseError as error:
        return str(error)

def get_arrival_time(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        arrival_times = [f"Flight arrives at {offer['itineraries'][0]['segments'][0]['arrival']['at']} at {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']}." for offer in data]
        return f"Here are the arrival times for flights to {place}: " + ", ".join(arrival_times)
    except ResponseError as error:
        return str(error)

def get_departure_time(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        departure_times = [f"Flight departs at {offer['itineraries'][0]['segments'][0]['departure']['at']} from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']}." for offer in data]
        return f"Here are the departure times for flights to {place}: " + ", ".join(departure_times)
    except ResponseError as error:
        return str(error)

def get_airline_information(place, query_type):
    try:
        response = amadeus.reference_data.airlines.get(airlineCodes="BA")
        data = response.data
        airlines = [f"Airline: {airline['commonName']} ({airline['iataCode']})." for airline in data]
        return f"Here is some airline information for flights to {place}: " + ", ".join(airlines)
    except ResponseError as error:
        return str(error)

def get_airline_ratings(place, query_type):
    try:
        response = amadeus.e_reputation.airline_ratings.get(airlineCodes="BA")
        data = response.data
        ratings = [f"Airline {rating['iataCode']} has a rating of {rating['rating']}." for rating in data]
        return f"Here are the airline ratings for flights to {place}: " + ", ".join(ratings)
    except ResponseError as error:
        return str(error)

def get_baggage_policies(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        policies = [f"Baggage policy for flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} includes {offer['travelerPricings'][0]['baggage']['pieces']} pieces of baggage." for offer in data]
        return f"Here are the baggage policies for flights to {place}: " + ", ".join(policies)
    except ResponseError as error:
        return str(error)

def get_seat_availability(place, query_type):
    try:
        response = amadeus.shopping.seatmaps.get(flightOffers=[{"type": "flight-offer", "id": "1"}])
        data = response.data
        availability = [f"Seat availability: {seat['availability']} seats available in {seat['class']} class." for seat in data[0]['segments'][0]['cabins'][0]['seats']]
        return f"Here is the seat availability for flights to {place}: " + ", ".join(availability)
    except ResponseError as error:
        return str(error)

def get_in_flight_services(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        services = [f"In-flight services for flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} include {offer['services']}." for offer in data]
        return f"Here are the in-flight services for flights to {place}: " + ", ".join(services)
    except ResponseError as error:
        return str(error)

def get_airport_information(place, query_type):
    try:
        response = amadeus.reference_data.locations.airports.get(keyword=place)
        data = response.data
        airports = [f"Airport: {airport['name']} ({airport['iataCode']})." for airport in data]
        return f"Here is some airport information for flights to {place}: " + ", ".join(airports)
    except ResponseError as error:
        return str(error)

def get_airport_lounges(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        lounges = [f"Airport lounges for flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} include {offer['services']}." for offer in data]
        return f"Here are some airport lounges for flights to {place}: " + ", ".join(lounges)
    except ResponseError as error:
        return str(error)

def get_flight_amenities(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        amenities = [f"Flight amenities for flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} include {offer['services']}." for offer in data]
        return f"Here are some flight amenities for flights to {place}: " + ", ".join(amenities)
    except ResponseError as error:
        return str(error)

def get_flight_delays(place, query_type):
    try:
        response = amadeus.schedule.flights.get(origin="SYD", destination=place, departureDate="2023-06-01")
        data = response.data
        delays = [f"Flight {flight['carrierCode']}{flight['flightNumber']} is delayed by {flight['delay']} minutes." for flight in data]
        return ", ".join(delays)
    except ResponseError as error:
        return str(error)

def get_cancellation_policies(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        policies = [f"Cancellation policy for flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} is {offer['cancellation']}." for offer in data]
        return ", ".join(policies)
    except ResponseError as error:
        return str(error)

def get_flight_promotions(place, query_type):
    try:
        response = amadeus.shopping.flight_offers_search.get(origin="SYD", destination=place, departureDate="2023-06-01", adults=1)
        data = response.data
        promotions = [f"Flight promotions for flight from {offer['itineraries'][0]['segments'][0]['departure']['iataCode']} to {offer['itineraries'][0]['segments'][0]['arrival']['iataCode']} include {offer['promotions']}." for offer in data]
        return ", ".join(promotions)
    except ResponseError as error:
        return str(error)

def get_frequent_flyer_programs(place, query_type):
    try:
        response = amadeus.reference_data.airlines.get(airlineCodes="BA")
        data = response.data
        programs = [f"Frequent flyer programs for airline {airline['iataCode']} include {airline['frequentFlyerPrograms']}." for airline in data]
        return ", ".join(programs)
    except ResponseError as error:
        return str(error)

if __name__ == '__main__':
    app.run(debug=True)
