from PIL import Image
from io import BytesIO
from imgbeddings import imgbeddings

class ImageEmbeddingGenerator:
    def __init__(self):
        self.ibed = imgbeddings()

    def generate_embedding(self, image_path):
        embedding = self.ibed.to_embeddings(image_path)
        return embedding

# Example usage:
# if __name__ == "__main__":
#     path = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     generator = ImageEmbeddingGenerator()
#     embedding = generator.generate_embedding(path)
    
