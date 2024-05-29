from image_utils import get_image_files
import cv2
from vector_storage import VectorStorage
from generate_embeddings import ImageEmbeddingGenerator


def trainer(user_data, embedding_generator, storage):
    image_files = get_image_files(user_data)

    for path in image_files:
        embedding = embedding_generator.generate_embedding(path)
        storage.add_vector(path, embedding, {'path': path})

    print(f"Number of embeddings stored: {len(storage.vectors)}")
    return storage


if __name__=='__main__':

    face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
    storage = VectorStorage('vectors.pkl')
    embedding_generator = ImageEmbeddingGenerator()

    # Training...
    storage = trainer("user_data", embedding_generator, storage)