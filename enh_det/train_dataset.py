import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from low_light_recognition import CLAHEEnhancer

# ===========================
# Initialize FaceAnalysis
# ===========================
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))
enhancer = CLAHEEnhancer()

# ===========================
# Get all image paths recursively
# ===========================
def get_image_paths(dataset_dir):
    image_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# ===========================
# Train Dataset
# ===========================
def train_dataset(dataset_dir="dataset", output_file="trained_faces.pkl"):
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset folder '{dataset_dir}' does not exist. Please create it.")
        return None

    embeddings = []
    labels = []

    image_paths = get_image_paths(dataset_dir)
    if not image_paths:
        print(f"❌ No images found in '{dataset_dir}'. Please add images.")
        return None

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        # Enhance image for low light
        img = enhancer.enhance(img)

        # Detect face
        faces = app.get(img)
        if len(faces) == 0:
            print(f"⚠️ No face detected in {img_path}")
            continue

        # Get embedding
        face = faces[0]
        if hasattr(face, 'embedding') and face.embedding is not None:
            embeddings.append(face.embedding.flatten())

            # Use parent folder as label
            label = os.path.basename(os.path.dirname(img_path))
            labels.append(label)
            print(f"✅ Processed {label}/{os.path.basename(img_path)}")
        else:
            print(f"⚠️ No embedding for {img_path}")

    if len(embeddings) == 0:
        print("❌ No faces found. Training failed.")
        return None

    # Save embeddings + labels
    data = {"embeddings": np.array(embeddings), "labels": labels}
    with open(output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"\n✅ Training complete. Saved {len(labels)} samples to {output_file}")
    return data

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    dataset_dir = "dataset"  # change if your dataset is in another folder
    output_file = "trained_faces.pkl"
    train_dataset(dataset_dir, output_file)
