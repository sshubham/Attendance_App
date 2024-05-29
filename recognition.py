import cv2
from PIL import Image

def recognition(target_image_path, face_cascade, embedding_generator, storage, top_n=1):
    frame = cv2.imread(target_image_path)
    if frame is None:
        raise ValueError(f"Failed to load image from path: {target_image_path}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    for (x, y, w, h) in faces:
        print("Face Found...")
        face_region = gray[y:y+h, x:x+w]
        face_image = Image.fromarray(face_region)
        face_embedding = embedding_generator.generate_embedding(face_image)

        closest = storage.find_closest_vector(face_embedding, top_n=top_n)
        # print("Closest vector (Cosine Similarity):", closest)
        if closest:
            closest_id = closest[0][0]
            closest_meta = storage.get_vector_info(closest_id)
            return {'ID': closest_id, 'Meta': closest_meta, "Distance": closest[0][1]}
    return None
