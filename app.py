import streamlit as st
import cv2, os, sqlite3
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import pandas as pd

# ----------------------- DATABASE -----------------------
def init_db():
    conn = sqlite3.connect("attendance.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        entry_time TEXT,
        exit_time TEXT,
        duration TEXT,
        exit_count INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()

def mark_entry(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect("attendance.db")
    conn.execute("INSERT INTO attendance (name, entry_time, exit_count) VALUES (?, ?, 0)", (name, now))
    conn.commit()
    conn.close()

def mark_exit(name):
    now = datetime.now()
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT id, entry_time, exit_count FROM attendance WHERE name=? AND exit_time IS NULL ORDER BY id DESC LIMIT 1", (name,))
    row = c.fetchone()
    if row:
        entry_time = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
        duration = str(now - entry_time)
        conn.execute("UPDATE attendance SET exit_time=?, duration=?, exit_count=? WHERE id=?",
                     (now.strftime('%Y-%m-%d %H:%M:%S'), duration, row[2]+1, row[0]))
        conn.commit()
    conn.close()

# ------------------- FACE RECOGNITION SETUP -------------------
@st.cache_resource
def load_model():
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app

@st.cache_resource
def load_known_faces():
    model = load_model()
    encodings, names = [], []
    known_faces_dir = "known_faces"
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)
    for file in os.listdir(known_faces_dir):
        path = os.path.join(known_faces_dir, file)
        img = cv2.imread(path)
        if img is not None:
            faces = model.get(img)
            if faces:
                encodings.append(faces[0].embedding)
                names.append(os.path.splitext(file)[0])
    return encodings, names

# -------------------------- MAIN APP --------------------------
def main():
    st.set_page_config(page_title="Smart Attendance System", layout="wide")
    st.title("🎓 Smart Face Recognition Attendance System (InsightFace)")

    init_db()
    model = load_model()
    known_encs, known_names = load_known_faces()
    seen = {}

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam")
            break

        faces = model.get(frame)
        for face in faces:
            emb = face.embedding
            x1, y1, x2, y2 = map(int, face.bbox)
            match = False
            for i, known in enumerate(known_encs):
                sim = cosine_similarity([emb], [known])[0][0]
                if sim > 0.6:
                    name = known_names[i]
                    now = datetime.now()
                    if name not in seen:
                        seen[name] = now
                        mark_entry(name)
                        st.success(f"✔ Entry marked for {name}")
                    elif (now - seen[name]).seconds > 5400:
                        mark_exit(name)
                        del seen[name]
                        st.info(f"❌ Exit marked for {name}")

                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    match = True
                    break

            if not match:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

    cap.release()

    if st.button("Show Attendance Log"):
        conn = sqlite3.connect("attendance.db")
        try:
            df = conn.execute("SELECT * FROM attendance").fetchall()
        except:
            st.warning("No attendance records found.")
            df = None
        conn.close()

        if df:
            df = pd.DataFrame(df, columns=["ID", "Name", "Entry Time", "Exit Time", "Duration", "Exit Count"])
            st.dataframe(df)

if __name__ == "__main__":
    main()
