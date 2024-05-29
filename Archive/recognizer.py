import os
import pickle
import numpy as np
from PIL import Image
import cv2
from scipy.spatial.distance import cosine
from generate_embeddings import ImageEmbeddingGenerator

class VectorStorage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vectors = {}

    def add_vector(self, identifier, vector, meta_info):
        self.vectors[identifier] = {'vector': np.array(vector).flatten(), 'meta_info': meta_info}
        self.save_vectors()

    def update_vector(self, identifier, new_vector=None, new_meta_info=None):
        if identifier in self.vectors:
            if new_vector is not None:
                self.vectors[identifier]['vector'] = np.array(new_vector).flatten()
            if new_meta_info is not None:
                self.vectors[identifier]['meta_info'] = new_meta_info
            self.save_vectors()
        else:
            print(f"Vector with identifier {identifier} not found.")

    def save_vectors(self):
        with open(self.file_path, 'wb') as file:
            pickle.dump(self.vectors, file)

    def load_vectors(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'rb') as file:
                self.vectors = pickle.load(file)
        else:
            print(f"File {self.file_path} not found. Starting with an empty storage.")

    def find_closest_vector(self, vector, top_n=1):
        vector = np.array(vector).flatten()
        similarities = []
        for identifier, data in self.vectors.items():
            stored_vector = data['vector']
            similarity = 1 - cosine(vector, stored_vector)
            similarities.append((identifier, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def get_vector_info(self, identifier):
        return self.vectors.get(identifier, None)

def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                try:
                    Image.open(os.path.join(root, file))
                    image_files.append(os.path.join(root, file))
                except (OSError, IOError):
                    pass
    return image_files

def trainer(user_data, embedding_generator, storage):
    image_files = get_image_files("user_data")

    for path in image_files:
        embedding = embedding_generator.generate_embedding(path)
        storage.add_vector(path, embedding, {'path': path})

    print(f"Number of embeddings stored: {len(storage.vectors)}")
    return storage

def recognition(target_image_path, face_cascade, embedding_generator, storage, top_n):
    frame = cv2.imread(target_image_path)
    if frame is None:
        raise ValueError(f"Failed to load image from path: {target_image_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_image = Image.fromarray(face_region)
        face_embedding = embedding_generator.generate_embedding(face_image)

        closest = storage.find_closest_vector(face_embedding, top_n=top_n)
        print("Closest vector (Cosine Similarity):", closest)
        if closest:
            closest_id = closest[0][0]
            closest_meta = storage.get_vector_info(closest_id)
            return {'ID':closest_id, 'Meta': 'closest_meta'}

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

    storage = VectorStorage('vectors.pkl')
    storage.load_vectors()

    embedding_generator = ImageEmbeddingGenerator()

    # Training...
    # storage = trainer("user_data", embedding_generator, storage)

    # Recognition...
    target_image_path = "t2.jpg"
    match = recognition(target_image_path, face_cascade, embedding_generator, storage, top_n=1)
    print(match["ID"])


