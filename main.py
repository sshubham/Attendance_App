import cv2
import os
import time
from vector_storage import VectorStorage
from generate_embeddings import ImageEmbeddingGenerator
from trainer import trainer
from recognition import recognition

def extract_id_from_path(file_path):
    normalized_path = os.path.normpath(file_path)
    path_parts = normalized_path.split(os.sep)
    if len(path_parts) >= 3:
        return path_parts[-2]
    return None

def detect_and_save_face():
    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    start_time = time.time()
    countdown = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        elapsed_time = time.time() - start_time
        remaining_time = countdown - int(elapsed_time)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Taking picture in {remaining_time} sec', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Detection', frame)

        margin = 60

        if remaining_time <= 0 and len(faces) > 0:
            x, y, w, h = faces[0]
            
            # Expand the region of interest (ROI) by adding margins
            x -= margin
            y -= margin
            w += 2 * margin
            h += 2 * margin
            
            # Ensure that the ROI stays within the image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(gray.shape[1] - x, w)
            h = min(gray.shape[0] - y, h)
            
            face_img = gray[y:y+h, x:x+w] 

            temp_dir = "face_capture/"

            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            image_path = os.path.join(temp_dir, 'captured_face.jpg')

            # Check if the file exists and remove it if it does
            if os.path.exists(image_path):
                os.remove(image_path)

            # Save the new image
            cv2.imwrite(image_path, face_img)
            print(f"Image saved at: {image_path}")

            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    detect_and_save_face()

    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    storage = VectorStorage('vectors.pkl')
    embedding_generator = ImageEmbeddingGenerator()

    target_image_path = "face_capture/captured_face.jpg"

    match = recognition(target_image_path, face_cascade, embedding_generator, storage, top_n=1)
    print(match)
    if match and 'Distance' in match and match['Distance'] < 80:
        print(f"User Match: {extract_id_from_path(match['ID'])} -> Distance: {match['Distance']}")
    else:
        print("Unrecognized...")

if __name__ == "__main__":
    main()
