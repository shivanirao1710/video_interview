import cv2
import torch
import time
import os
from PIL import Image
from torchvision import models, transforms
import psycopg2

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

# Emotion detection loop
username = os.getenv("LOGGED_IN_USER", "unknown")
emotion_log = []
frame_interval = 3
last_emotion_time = 0

cap = cv2.VideoCapture(0)
print("📸 Camera started. Emotion tracking running...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

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

    # Show webcam with overlay
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Live Interview Emotion Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save emotions to DB
conn = psycopg2.connect(
    dbname='behavioural',
    user='postgres',
    password='shivanirao1710',
    host='localhost',
    port='5432'
)
cur = conn.cursor()
cur.execute("DELETE FROM emotion_logs WHERE username = %s", (username,))
for e in emotion_log:
    cur.execute("INSERT INTO emotion_logs (timestamp, emotion, username) VALUES (%s, %s, %s)",
                (e['timestamp'], e['emotion'], username))
conn.commit()
conn.close()
print("✅ Emotion log saved.")
