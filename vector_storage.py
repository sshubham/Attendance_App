import os
import pickle
import numpy as np
import faiss

class VectorStorage:
    def __init__(self, file_path):
        self.file_path = file_path
        self.vectors = {}
        self.load_vectors()
        self.index = None
        self.build_index()

    def add_vector(self, identifier, vector, meta_info):
        self.vectors[identifier] = {'vector': np.array(vector).flatten(), 'meta_info': meta_info}
        self.save_vectors()
        self.build_index()

    def update_vector(self, identifier, new_vector=None, new_meta_info=None):
        if identifier in self.vectors:
            if new_vector is not None:
                self.vectors[identifier]['vector'] = np.array(new_vector).flatten()
            if new_meta_info is not None:
                self.vectors[identifier]['meta_info'] = new_meta_info
            self.save_vectors()
            self.build_index()
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

    def build_index(self):
        if self.vectors:
            vector_list = np.array([data['vector'] for data in self.vectors.values()]).astype('float32')
            self.index = faiss.IndexFlatL2(vector_list.shape[1])
            self.index.add(vector_list)
        else:
            self.index = None

    def find_closest_vector(self, vector, top_n=1):
        if self.index is None:
            print("Index is empty. No vectors available.")
            return []

        vector = np.array(vector).flatten().astype('float32').reshape(1, -1)
        
        if self.index.ntotal == 0:
            print("Index is empty. No vectors available.")
            return []

        distances, indices = self.index.search(vector, top_n)

        results = [
            (list(self.vectors.keys())[index], distances[0][i])
            for i, index in enumerate(indices[0])
        ]
        if not results:
            print("No match found.")
        return results

    def get_vector_info(self, identifier):
        return self.vectors.get(identifier, None)

# Example usage:
if __name__ == "__main__":
    storage = VectorStorage('vector_storage.pkl')

    # Adding vectors
    storage.add_vector('vec1', [1, 2, 3], {'info': 'first vector'})
    storage.add_vector('vec2', [4, 5, 6], {'info': 'second vector'})
    storage.add_vector('vec3', [7, 8, 9], {'info': 'third vector'})

    # Finding closest vectors
    query_vector = [1, 2, 3]
    closest_vectors = storage.find_closest_vector(query_vector, top_n=2)
    print(closest_vectors)

    # Testing with an unknown vector
    unknown_vector = [10, 10, 10]
    closest_vectors = storage.find_closest_vector(unknown_vector, top_n=2)
    print(closest_vectors)

    # Get vector info
    print(storage.get_vector_info('vec1'))
