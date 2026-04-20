import numpy as np
import time
from fastembed import TextEmbedding
from llm import load_movies

print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("Loading movie dataset...")
movies = list(load_movies())

print(f"Preparing {len(movies)} documents for embedding...")
documents = []

for movie in movies:
    # Build a rich narrative string that captures the semantic essence of the movie
    genres_text = ", ".join(movie.genres)
    cast_text = ", ".join(movie.cast[:3])
    keywords_text = ", ".join(movie.keywords)
    
    doc = (
        f"Movie Title: {movie.title}. "
        f"Genres: {genres_text}. "
        f"Director: {movie.director}. "
        f"Cast: {cast_text}. "
        f"Keywords: {keywords_text}. "
        f"Story Overview: {movie.overview}"
    )
    documents.append(doc)

print("Computing dense math vectors (this will take about 10-30 seconds depending on CPU)...")
start_time = time.time()
# fastembed yields a generator, we cast to list and then stack to a 2D numpy array
embeddings_list = list(embedder.embed(documents))
embeddings_matrix = np.vstack(embeddings_list)
elapsed = time.time() - start_time

print(f"Finished computing vectors! Matrix shape: {embeddings_matrix.shape}. Took {elapsed:.2f} seconds.")

# Save to disk
output_file = "movie_embeddings.npy"
np.save(output_file, embeddings_matrix)
print(f"Successfully saved dense vectors to {output_file}!")
