This model is an intelligent, voice-driven health advisor that combines real-time face recognition, speech transcription, language detection, and LLM-powered medical interaction. It uses Python and integrates the Groq API (Whisper and LLaMA-3), delivering concise and empathetic responses to user health queries in English and Tamil.



**Features:**


Real-time face detection and recognition using webcam
Automatic user registration with facial encoding
Voice interaction with transcription using Whisper v3 via Groq API
Language detection and response in either English or Tamil
Natural conversation flow powered by LLaMA-3 (Groq)
Personalized greetings based on recognized users
Automatic cleanup of unused face data after 30 days
Supports both gTTS and pyttsx3 for speech output
Simple fallback mechanisms for unsupported languages or errors



**How It Works:**


The webcam detects and identifies a face.
If the face is new, the system prompts for a name and stores the facial encoding.
Once identified, the assistant greets the user and begins voice-based interaction.
The user speaks a query, which is recorded and transcribed using Whisper.
The transcribed text is analyzed using Groq’s LLaMA-3 API for a health-focused response.
The response is spoken back to the user in the detected language.



**Tech Stack:**


Languages: Python
AI/ML: face_recognition, Groq Whisper and LLaMA-3
Speech: gTTS, pyttsx3, pyaudio, pygame
Image Processing: OpenCV
NLP: langdetect, regular expressions
I/O: sounddevice, scipy.io.wavfile, tempfile



**Requirements:**


Python 3.7+
Webcam and Microphone
Groq API Key (for Whisper and LLaMA-3 access)



**Python Libraries:**


Install the dependencies using pip:
pip install opencv-python face_recognition gtts pyttsx3 pygame sounddevice scipy langdetect speechrecognition pyaudio groq



**Directory Structure:**


project/

│

├── main.py

├── Face_detection/

│   └── faces_data/           # Stores JSON-encoded face data

├── temp_speech.mp3           # Temporary TTS files (runtime)

└── README.md



**Setup and Usage:**


Clone the repository.
Add your Groq API key in the GROQ_API_KEY variable.
Run the script
Look at the camera. If unrecognized, enter your name when prompted.
Wait for the assistant to greet you.
Speak a medical question. The assistant will reply with a concise and helpful response.
Say "bye", "exit", or "stop" to end the session.



**Notes:**


The application supports only English and Tamil for now.
Audio files are deleted automatically after playback unless an error occurs.
Face data older than 30 days is automatically purged to save space.
