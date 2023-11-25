import numpy as np

class LSH:
    def __init__(self, num_layers, num_hashes, input_dim):
        self.num_layers = num_layers
        self.num_hashes = num_hashes
        self.input_dim = input_dim
        self.hash_tables = [{} for _ in range(num_layers)]
        self.random_planes = [np.random.randn(num_hashes, input_dim) for _ in range(num_layers)]

    def hash_vector(self, vector, planes):
        return ''.join(['1' if np.dot(vector, plane) > 0 else '0' for plane in planes])

    def index_vector(self, vector, index):
        for i in range(self.num_layers):
            hash_value = self.hash_vector(vector, self.random_planes[i])
            hash_key = tuple(vector)  # Convert NumPy array to tuple
            if hash_value not in self.hash_tables[i]:
                self.hash_tables[i][hash_value] = []
            self.hash_tables[i][hash_value].append((hash_key, index))

    def index_vectors(self, vectors):
        for i, vector in enumerate(vectors):
            self.index_vector(vector, i)
            
    def display_buckets(self):
        for i in range(self.num_layers):
            print(f"Layer {i+1} Buckets:")
            for key, bucket in self.hash_tables[i].items():
                print(f"  Hash: {key}, Size: {len(bucket)}")
                # To display the contents of the bucket
                # Uncomment the following lines
                print("    Contents:")
                for item in bucket:
                    print(f"      {item}")
            print()

# Example usage:
num_layers = 5
num_hashes = 10
input_dim = 3

# Generate random vectors for demonstration
np.random.seed(42)
data = np.random.rand(100, input_dim)

lsh = LSH(num_layers, num_hashes, input_dim)
lsh.index_vectors(data)

# Display the buckets
lsh.display_buckets()

