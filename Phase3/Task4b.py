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

    def display_buckets(self, layer):
        if layer <= 0 or layer > self.num_layers:
            print("Layer index out of range.")
            return

        print(f"Layer {layer} Buckets:")
        for key, value in self.hash_tables[layer - 1].items():
            print(f"Hash Value: {key}")
            print("Items:")
            for item in value:
                print(f"   Index: {item[1]}, Vector: {item[0]}")
            print("=" * 20)



import scipy.io
#from google.colab import drive
#drive.mount('/content/drive')
#mat = scipy.io.loadmat('/content/drive/MyDrive/arrays.mat')
#above three for googlecolab

mat = scipy.io.loadmat('arrays.mat')

sel = "avgpool_features"

print(mat[sel])
print(len(mat[sel]))
print(len(mat[sel][0]))

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
#for i in range(num_layers):
#  lsh.display_buckets(i)

#lsh.display_buckets(1)

index = int(input("The index of the image you want to search (will be going through only the even) : "))

query = mat[sel][index]
hashed_buckets = lsh.hash_query_vector(query)

for layer, hash_value, bucket in hashed_buckets:
    print(f"Layer {layer} - Hash: {hash_value}, Bucket Size: {len(bucket)}")


unique_indices = set()

for layer, _, bucket in hashed_buckets:
    print(f"Layer {layer} - Bucket Size: {len(bucket)}")
    for item in bucket:
        unique_indices.add(item[1])  # Collecting unique indices

print("Unique Vector Indices:")
print(unique_indices)

print(len(unique_indices))

from heapq import nsmallest

k = int(input("Enter the value of k for nearest neighbors: "))

distances = []  # To store calculated distances
for _, _, bucket in hashed_buckets:
    for item in bucket:
        vector = np.array(item[0])  # Retrieve the vector from the bucket
        distance = np.linalg.norm(query - vector)  # Calculate Euclidean distance
        distances.append((item[1], distance))  # Store index and distance

# Sort distances and retrieve k indices with smallest distances
nearest_indices = [index for index, _ in nsmallest(k, distances, key=lambda x: x[1])]

print(f"Indices of {k} Nearest Neighbors:")
print(nearest_indices)

#To remove duplicates
distances_unique = set(distances)

# Sort distances and retrieve k indices with smallest distances
nearest_indis = [indi for indi, _ in nsmallest(k, distances_unique, key=lambda x: x[1])]

print(f"Indices of {k} Nearest Neighbors:")
print(nearest_indis)

