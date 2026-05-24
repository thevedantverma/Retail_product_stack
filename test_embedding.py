import cv2

from embeddings.embedder import get_embedding

image = cv2.imread("test.jpg")

embedding = get_embedding(image)

print(embedding.shape)