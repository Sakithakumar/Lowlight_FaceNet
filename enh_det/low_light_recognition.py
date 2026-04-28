import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# 1. CLAHE Enhancer
# ===========================
class CLAHEEnhancer:
    def enhance(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced

# ===========================
# 2. Initialize FaceAnalysis
# ===========================
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))
print("✅ FaceAnalysis model loaded (buffalo_l).")

# ===========================
# 3. Load Known Faces
# ===========================
def load_known_faces(folder="known_faces"):
    known_faces = {}
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"📁 Created '{folder}' folder. Add images like 's.jpg' etc.")
        return known_faces

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️  Could not read {filename}")
                continue

            faces = app.get(img)
            if len(faces) > 0:
                face = faces[0]
                if hasattr(face, 'embedding') and face.embedding is not None:
                    known_faces[name] = face.embedding.flatten()
                    print(f"✅ Loaded face: {name} | Embedding shape: {face.embedding.shape}")
                else:
                    print(f"⚠️  No embedding for {filename}")
            else:
                print(f"⚠️  No face detected in {filename}")
    return known_faces

known_faces = load_known_faces()

# ===========================
# 4. Recognize Faces
# ===========================
def recognize_faces(frame, known_faces, threshold=0.4):
    enhancer = CLAHEEnhancer()
    enhanced_frame = enhancer.enhance(frame)
    faces = app.get(enhanced_frame)
    results = []

    for face in faces:
        bbox = face.bbox.astype(int)
        if not hasattr(face, 'embedding') or face.embedding is None:
            continue

        embedding = face.embedding.flatten()
        best_match = "Unknown"
        best_score = 0.0

        for name, known_emb in known_faces.items():
            sim = cosine_similarity([embedding], [known_emb])[0][0]
            if sim > best_score:
                best_score = sim
                if sim > threshold:
                    best_match = name

        results.append({
            'bbox': bbox,
            'name': best_match,
            'confidence': best_score
        })

    return results, enhanced_frame

# ===========================
# 5. Live Webcam with Matplotlib Display
# ===========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("🎥 Press Ctrl+C to quit | Frames displayed via Matplotlib")

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results, enhanced_frame = recognize_faces(frame, known_faces, threshold=0.4)

            # Draw results
            display_frame = enhanced_frame.copy()
            for res in results:
                x1, y1, x2, y2 = res['bbox']
                name = res['name']
                conf = res['confidence']
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({conf:.2f})" if name != "Unknown" else "Unknown"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display with matplotlib
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.imshow(rgb)
            ax.axis('off')
            ax.set_title(f"Low-Light Face Recognition | FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
            plt.draw()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")

    cap.release()
    plt.close()
    print("✅ Done.")

if __name__ == "__main__":
    main()