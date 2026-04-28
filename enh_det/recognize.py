import cv2
import pickle
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Load CLAHEEnhancer class from train_dataset.py if you want
from train_dataset import CLAHEEnhancer  

# Load pre-trained face analysis model
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    known_faces = pickle.load(f)
print(f"✅ Loaded {len(known_faces)} embeddings from embeddings.pkl")

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

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("🎥 Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results, enhanced_frame = recognize_faces(frame, known_faces, threshold=0.4)

        display_frame = enhanced_frame.copy()
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            name = res['name']
            conf = res['confidence']
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({conf:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done.")

if __name__ == "__main__":
    main()
