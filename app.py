import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime
from insightface.app import FaceAnalysis

# Initialize database
def init_db():
    conn = sqlite3.connect("attendance.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        entry_time TEXT
    )''')
    conn.commit(); conn.close()

# Log attendance
def mark_attendance(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect("attendance.db")
    conn.execute("INSERT INTO attendance (name, entry_time) VALUES (?, ?)", (name, now))
    conn.commit(); conn.close()

# Load face model
@st.cache_resource
def load_model():
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app

# Load known face encodings
@st.cache_resource
def load_known_faces(model):
    encodings = []
    names = []
    path = "known_faces"
    if not os.path.exists(path):
        os.makedirs(path)
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        if img is None:
            continue
        faces = model.get(img)
        if faces:
            encodings.append(faces[0].embedding)
            names.append(os.path.splitext(file)[0])
    return encodings, names

# ------------------------ Streamlit UI ------------------------
st.set_page_config(page_title="Face Attendance", layout="centered")
st.title("📸 Face Recognition Attendance System")
st.markdown("Upload your image or use the camera to mark attendance.")

init_db()
model = load_model()
known_encs, known_names = load_known_faces(model)

# Upload or capture photo
source = st.radio("Choose input method:", ["Upload Image", "Use Camera"])
image = None

if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif source == "Use Camera":
    captured = st.camera_input("Take a picture")
    if captured:
        image = Image.open(captured)

# Process image
if image:
    st.image(image, caption="Captured Image", use_column_width=True)
    img_np = np.array(image.convert("RGB"))
    faces = model.get(img_np)

    if not faces:
        st.warning("No face detected. Try again.")
    else:
        matched = False
        for face in faces:
            emb = face.embedding
            for i, known in enumerate(known_encs):
                sim = cosine_similarity([emb], [known])[0][0]
                if sim > 0.6:
                    name = known_names[i]
                    mark_attendance(name)
                    st.success(f"✔ Attendance marked for {name}")
                    matched = True
                    break
            if not matched:
                st.error("❌ Face not recognized.")

# View attendance
if st.button("📋 Show Attendance Log"):
    conn = sqlite3.connect("attendance.db")
    rows = conn.execute("SELECT * FROM attendance ORDER BY entry_time DESC").fetchall()
    conn.close()

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["ID", "Name", "Time"])
        st.dataframe(df)
    else:
        st.info("No attendance logged yet.")
