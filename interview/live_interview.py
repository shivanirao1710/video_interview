import cv2
import torch
import time
import os
import signal
import sys
from PIL import Image
from torchvision import models, transforms
import psycopg2

# Global flag for controlling the loop
running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("🚦 Received shutdown signal")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

# Load emotion labels and model
labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
model_path = "emotion_model.pt"
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def clear_previous_emotions(username):
    """Clear all emotion logs for a specific user"""
    try:
        conn = psycopg2.connect(
            dbname='behavioural',
            user='postgres',
            password='shivanirao1710',
            host='localhost',
            port='5432'
        )
        cur = conn.cursor()
        cur.execute("DELETE FROM emotion_logs WHERE username = %s", (username,))
        conn.commit()
        conn.close()
        print(f"🧹 Cleared previous emotion logs for {username}")
    except Exception as e:
        print(f"❌ Failed to clear previous emotions: {e}")

def save_emotions_to_db(emotion_log, username):
    """Save emotions to database (without deleting previous entries)"""
    try:
        conn = psycopg2.connect(
            dbname='behavioural',
            user='postgres',
            password='shivanirao1710',
            host='localhost',
            port='5432'
        )
        cur = conn.cursor()
        for e in emotion_log:
            cur.execute("INSERT INTO emotion_logs (timestamp, emotion, username) VALUES (%s, %s, %s)",
                       (e['timestamp'], e['emotion'], username))
        conn.commit()
        conn.close()
        print(f"✅ Saved {len(emotion_log)} emotion logs to DB")
    except Exception as e:
        print(f"❌ Failed to save emotions to DB: {e}")

def main():
    username = os.getenv("LOGGED_IN_USER", "unknown")
    emotion_log = []
    frame_interval = 3  # seconds between emotion checks
    last_emotion_time = 0
    last_save_time = time.time()
    save_interval = 10  # seconds between DB saves

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    print("📸 Camera started. Emotion tracking running...")
    emotion = "neutral"  # Default emotion

    while running:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Could not read frame from camera")
            break

        current_time = time.time()

        # Emotion detection logic
        if current_time - last_emotion_time >= frame_interval:
            last_emotion_time = current_time

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img)
                _, pred = torch.max(outputs, 1)
                emotion = labels[pred.item()]

            timestamp = time.strftime("%H:%M:%S")
            emotion_log.append({"timestamp": timestamp, "emotion": emotion})
            print(f"[{timestamp}] Emotion: {emotion}")

        # Periodic save to DB
        if current_time - last_save_time >= save_interval and emotion_log:
            save_emotions_to_db(emotion_log, username)
            emotion_log = []  # Clear saved logs
            last_save_time = current_time

        # Display frame with emotion
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Live Interview Emotion Tracker", frame)

        # Break loop if 'q' is pressed (but keep process running)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            # Don't break here - just hide the window

    # Final save before exiting
    if emotion_log:
        save_emotions_to_db(emotion_log, username)
    
    cap.release()
    print("🛑 Emotion tracking stopped")

if __name__ == "__main__":
    main()