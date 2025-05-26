import os
import sys
import cv2
import re
import json
import time
import pygame
import tempfile
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile
import face_recognition
import speech_recognition as sr
from datetime import datetime
from langdetect import detect
from gtts import gTTS
from groq import Groq

# === Configuration ===
GROQ_API_KEY = "YOUR_API_KEY_HERE"
DATA_DIR = os.path.join(os.path.dirname(__file__), "Face_detection", "faces_data")
CLEANUP_DAYS = 30  # Days after which unused face data is deleted

# === Globals ===
engine = pyttsx3.init()
greeted_names = set()
chat_history = []
greeted = False

# === Ensure DATA_DIR exists ===
os.makedirs(DATA_DIR, exist_ok=True)

# === Face Recognition Functions ===
def greet(name):
    if name not in greeted_names:
        if len(greeted_names) == 0:
            message = f"Hello {name}! How can I help you today?"
        else:
            message = f"Hello {name}! Nice to see you now. Is there anything I can assist you with?"
        engine.say(message)
        engine.runAndWait()
        print(message)
        greeted_names.add(name)

def encode_face(image):
    face_enc = face_recognition.face_encodings(image)
    return face_enc[0] if face_enc else None

def save_face(name, encoding):
    filepath = os.path.join(DATA_DIR, f"{name}.json")
    if not os.path.exists(filepath):
        data = {"encoding": encoding.tolist(), "last_seen": datetime.now().isoformat()}
        with open(filepath, "w") as f:
            json.dump(data, f)
        print(f"{name} has been registered!")
        greet(name)
    else:
        print(f"{name} already registered!")

def load_all_faces():
    known_faces = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                name = filename.replace(".json", "")
                known_faces[name] = data["encoding"]
                if "last_seen" not in data:
                    data["last_seen"] = datetime.now().isoformat()
                    with open(filepath, "w") as f:
                        json.dump(data, f)
    return known_faces

def detect_and_recognize(frame, known_faces):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    for encoding, (top, right, bottom, left) in zip(encodings, locations):
        match = False
        for name, known_enc in known_faces.items():
            if face_recognition.compare_faces([known_enc], encoding)[0]:
                cv2.putText(frame, f"Hello {name}!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                greet(name)
                match = True
                filepath = os.path.join(DATA_DIR, f"{name}.json")
                with open(filepath, "r") as f:
                    data = json.load(f)
                data["last_seen"] = datetime.now().isoformat()
                with open(filepath, "w") as f:
                    json.dump(data, f)
                break

        if not match:
            cv2.putText(frame, "Unknown Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            name = input("New face detected! What's your name? ")
            save_face(name, encoding)
            known_faces[name] = encoding
            print(f"{name} added successfully!")

    return frame

def cleanup_old_faces():
    now = datetime.now()
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                if "timestamp" in data:
                    last_used = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if (now - last_used).days > CLEANUP_DAYS:
                        os.remove(filepath)
                        print(f"Removed {filename} due to 30+ days of inactivity.")
                else:
                    print(f"No timestamp found in {filename}. Skipping...")
            except Exception as e:
                print(f"Error with {filename}: {e}")

# === Audio Functions ===
def play_audio(file_path, delete_after=True):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
    pygame.mixer.quit()
    if delete_after:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete {file_path}. Reason: {e}")


import pyaudio
import numpy as np
import tempfile
import scipy.io.wavfile
from gtts import gTTS

def record_audio_pyaudio(duration=5, samplerate=16000, channels=1):
    chunk = 1024
    format = pyaudio.paInt16
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=samplerate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording...")
    frames = []

    for _ in range(int(samplerate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_bytes = b''.join(frames)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).reshape(-1, channels)
    return audio_np

def record_and_transcribe_whisper(duration=5, samplerate=16000):
    print("Recording Prompt...")
    tts = gTTS("Please speak something...")
    tts.save("temp_speech.mp3")
    play_audio("temp_speech.mp3")

    # Record using PyAudio
    recording = record_audio_pyaudio(duration=duration, samplerate=samplerate, channels=1)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        scipy.io.wavfile.write(f.name, samplerate, recording)
        audio_path = f.name

    try:
        client = Groq(api_key=GROQ_API_KEY)
        whisper_response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=open(audio_path, "rb")
        )
        text = whisper_response.text
        print("You said:", text)
        return text
    except Exception as e:
        print("Whisper API Error:", e)
        tts = gTTS("Sorry, I couldn't understand you.")
        tts.save("temp_error.mp3")
        play_audio("temp_error.mp3")
        return ""

'''
def record_and_transcribe_whisper(duration=5, samplerate=16000):
    print("Recording...")
    tts = gTTS("Please speak something...")
    tts.save("temp_speech.mp3")
    play_audio("temp_speech.mp3")

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        scipy.io.wavfile.write(f.name, samplerate, recording)
        audio_path = f.name

    try:
        client = Groq(api_key=GROQ_API_KEY)
        whisper_response = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=open(audio_path, "rb")
        )
        text = whisper_response.text
        print("You said:", text)
        return text
    except Exception as e:
        print("Whisper API Error:", e)
        tts = gTTS("Sorry, I couldn't understand you.")
        tts.save("temp_error.mp3")
        play_audio("temp_error.mp3")
        return ""

'''

import pyttsx3
from gtts import gTTS
import os
import time

engine = pyttsx3.init()

def speak_in_language(text, lang='en'):
    if lang == 'en':
        # Use pyttsx3 with Microsoft Zira
        voices = engine.getProperty('voices')
        selected_voice = None
        for voice in voices:
            if "zira" in voice.name.lower():
                selected_voice = voice.id
                break

        if selected_voice:
            engine.setProperty('voice', selected_voice)
        else:
            print("Microsoft Zira not found. Using default English voice.")
        
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()

    elif lang == 'ta':
        # Use gTTS for Tamil
        try:
            tts = gTTS(text=text, lang='ta')
            filename = "temp_tamil_audio.mp3"
            tts.save(filename)
            play_audio(filename) 
            #os.remove(filename)
        except Exception as e:
            print(f"Error using gTTS for Tamil: {e}")

    else:
        print(f"Unsupported language: {lang}")



# === Main Execution ===
if __name__ == "__main__":
    # Start face recognition loop
    cap = cv2.VideoCapture(0)
    known_faces = load_all_faces()
    cleanup_old_faces()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_and_recognize(frame, known_faces)
        cv2.imshow("Face Recognition", frame)

        if greeted_names:
            greeted = True
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Start Doctor AI Interaction
    if greeted:
        client = Groq(api_key=GROQ_API_KEY)
        while True:
            text = record_and_transcribe_whisper()
            meaningless_inputs = {"", ".", "um", "uh", "..."}
            if text.strip().lower() in meaningless_inputs:
                speak_in_language("No valid input received. I'm listening again...")
                continue

            try:
                detected_language = detect(text)
                print(f"Detected Language: {detected_language}")
                if detected_language not in ["en", "ta"]:
                    speak_in_language("Language not supported. Please speak in Tamil or English.", "en")
                    continue
            except Exception:
                detected_language = 'en'
                print("Could not detect language. Defaulting to English.")


            if re.search(r"\b(bye|exit|quit|stop|terminate)\b", text, re.IGNORECASE):
                speak_in_language("Thank you for contacting me! Please feel free to talk to me in the future if needed! Bye. Take care.", detected_language)
                break

            if text:
                model = "llama3-70b-8192"
                user_query = text

                messages = [{"role": "system", "content": """
                    You are Dr. Sophia, an expert doctor with high intelligence, precision, and empathy.
                    Generate the text in what language the user speaks, and don't use any other languages.
                    You provide accurate medical insights in a concise, easy-to-understand manner (within 70 words).
                    - Avoid irrelevant information and unnecessary details.
                    - Use medical terms only when necessary, based on the user’s mood and mindset.
                    - Maintain a professional, yet warm and supportive tone.
                    - If the user is distressed, be empathetic and reassuring.
                    - If the question is non-medical, answer wisely but stay professional.
                    - Don't say you are working with patients or at a hospital. You are just an advisor and not a real doctor.
                """}]
                messages.extend(chat_history)
                messages.append({"role": "user", "content": user_query})

                try:
                    response = client.chat.completions.create(model=model, messages=messages)
                    model_rep = response.choices[0].message.content
                    print("Model Response:\n", model_rep)

                    chat_history.append({"role": "user", "content": user_query})
                    chat_history.append({"role": "assistant", "content": model_rep})

                    if len(chat_history) > 10:
                        chat_history = chat_history[-10:]

                except Exception as e:
                    print("Error:", e)
                    model_rep = "Error processing your request."

                speak_in_language(model_rep, detected_language)

            else:
                speak_in_language("No valid input received. I'm Listening again...", detected_language)
