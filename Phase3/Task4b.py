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

    def hash_query_vector(self, query_vector):
        hashed_buckets = []
        for i in range(self.num_layers):
            hash_value = self.hash_vector(query_vector, self.random_planes[i])
            if hash_value in self.hash_tables[i]:
                bucket = self.hash_tables[i][hash_value]
                hashed_buckets.append((i+1, hash_value, bucket))
            else:
                hashed_buckets.append((i+1, hash_value, []))
        return hashed_buckets

    def calculate_distances(self, query_vector, hashed_buckets):
        distances = []
        for layer, _, bucket in hashed_buckets:
            layer_distances = []
            for item in bucket:
                vector = np.array(item[1])  # Retrieve the vector from the bucket
                distance = np.linalg.norm(query_vector - vector)  # Calculate Euclidean distance
                layer_distances.append((item[0], distance))  # Store index and distance
            distances.append((layer, layer_distances))
        return distances



import scipy.io
from google.colab import drive
drive.mount('/content/drive')
mat = scipy.io.loadmat('/content/drive/MyDrive/arrays.mat')

sel = "avgpool_features"

# Example usage:
num_layers = 5
num_hashes = 10
input_dim = len(mat[sel][0])

# Generate random vectors for demonstration
# np.random.seed(42)
# data = np.random.rand(100, input_dim)

lsh = LSH(num_layers, num_hashes, input_dim)
lsh.index_vectors(mat[sel])

# Display the buckets
#lsh.display_buckets()

index = int(input("The index of the image you want to search (will be going through only the even) : "))

query = mat[sel][index]
hashed_buckets = lsh.hash_query_vector(query)

for layer, hash_value, bucket in hashed_buckets:
    print(f"Layer {layer} - Hash: {hash_value}, Bucket Size: {len(bucket)}")

# Calculate distances from query to items in hashed buckets
distances = lsh.calculate_distances(query, hashed_buckets)

# After calculating distances
k = 20  # Set the value of k for top k indices

# Flatten distances list and sort based on distance
all_distances = [(layer, index, dist) for layer, layer_distances in distances for index, dist in layer_distances]
sorted_distances = sorted(all_distances, key=lambda x: x[2])[:k]

# Extract indices from sorted distances
top_k_indices = [(layer, index) for layer, index, _ in sorted_distances]
print(top_k_indices)


for j in range(k):
  hello = np.array(top_k_indices[j][1])
  for i,rod in enumerate(mat[sel]):
    arrays_equal = np.array_equal(hello, rod)
    #print(hello)
    #print(rod)
    #print(arrays_equal)
    if arrays_equal == 1:
      print(i)

