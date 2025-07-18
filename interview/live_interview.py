import cv2
import torch
import pyttsx3
import time
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from PIL import Image
import threading
import speech_recognition as sr
from torchvision import models, transforms
import psycopg2
import queue

# ✅ Emotion labels
labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# ✅ Load emotion model
model_path = "emotion_model.pt"
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ✅ Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ✅ Speak using pyttsx3 in a thread
def speak(text):
    def _speak():
        try:
            tts = pyttsx3.init()
            tts.setProperty('rate', 145)
            tts.say(text)
            tts.runAndWait()
            tts.stop()
        except Exception as e:
            print(f"[TTS Error] {e}")
    thread = threading.Thread(target=_speak)
    thread.start()
    thread.join()

# ✅ Record audio
def record_audio(filename="temp.wav", duration=20, fs=44100):
    print(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, (audio * 32767).astype('int16'))

# ✅ Transcribe audio
def transcribe_audio(filename="temp.wav"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except:
            return "Could not understand audio."

# ✅ Questions
questions = [
    "Tell me about yourself.",
    "What are your strengths?",
    "Describe a challenge you faced.",
    "Why should we hire you?",
    "What are your career goals?"
]

emotion_log = []
answer_log = []
emotion = "neutral"
question_index = 0
ask_next = True
frame_interval = 3
last_emotion_time = 0
result_queue = queue.Queue()

# ✅ Start webcam
cap = cv2.VideoCapture(0)
print("\n🎤 AI Interview starting...")

# ✅ Ask and record answers
def handle_question(q_idx, question, result_queue):
    speak(question)
    record_audio("temp.wav", duration=20)
    answer_text = transcribe_audio("temp.wav")
    result_queue.put((question, answer_text))

# ✅ Interview loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Ask next question
    if question_index < len(questions) and ask_next and result_queue.empty():
        threading.Thread(target=handle_question, args=(question_index, questions[question_index], result_queue)).start()
        ask_next = False

    # Retrieve answer
    if not result_queue.empty():
        q, a = result_queue.get()
        answer_log.append({"question": q, "answer": a})
        question_index += 1
        ask_next = True

    # Emotion detection every few seconds
    if current_time - last_emotion_time >= frame_interval:
        last_emotion_time = current_time
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            emotion = labels[pred.item()]
        emotion_log.append({"timestamp": time.strftime("%H:%M:%S"), "emotion": emotion})

    # Show current emotion on video
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # If all questions done, show message
    if question_index >= len(questions):
        cv2.putText(frame, "✅ Interview Complete!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("🎥 AI Interview", frame)
        cv2.waitKey(2000)  # Show final message for 2 seconds
        break

    # Show video
    cv2.imshow("🎥 AI Interview", frame)

    # Allow manual quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Close video feed
cap.release()
cv2.destroyAllWindows()

# ✅ Final safety: wait for last answer to be collected
while not result_queue.empty():
    q, a = result_queue.get()
    answer_log.append({"question": q, "answer": a})

# ✅ Save to PostgreSQL
username = os.getenv("LOGGED_IN_USER", "unknown")
conn = psycopg2.connect(
    dbname='behavioural',
    user='postgres',
    password='shivanirao1710',
    host='localhost',
    port='5432'
)
cur = conn.cursor()

# Clear previous logs
cur.execute("DELETE FROM emotion_logs WHERE username = %s", (username,))
cur.execute("DELETE FROM interview_answers WHERE username = %s", (username,))

# Insert new logs
for e in emotion_log:
    cur.execute("INSERT INTO emotion_logs (timestamp, emotion, username) VALUES (%s, %s, %s)",
                (e['timestamp'], e['emotion'], username))

for a in answer_log:
    cur.execute("INSERT INTO interview_answers (question, answer, username) VALUES (%s, %s, %s)",
                (a['question'], a['answer'], username))

conn.commit()
conn.close()

print("\n✅ Interview Complete! Logs saved to PostgreSQL.")
