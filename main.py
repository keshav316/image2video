import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load and resize image
img = cv2.imread("image.png")
img = cv2.resize(img, (256, 256))
height, width, channels = img.shape

# Flatten image for PCA
flat_img = img.reshape(-1, 3)

# Normalize pixel values
scaler = StandardScaler()
flat_img_scaled = scaler.fit_transform(flat_img)

# Apply PCA with stronger distortion (1 component)
pca = PCA(n_components=1)
flat_img_pca = pca.fit_transform(flat_img_scaled)

# Reconstruct image from PCA output
reconstructed = scaler.inverse_transform(pca.inverse_transform(flat_img_pca))
reconstructed_img = reconstructed.reshape(height, width, 3).astype(np.uint8)

# Create video writer
video_writer = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))

# Generate animated frames with motion and transformation
for i in range(30):  # 3 seconds at 10 FPS
    alpha = i / 30

    # Simulate motion: zoom + horizontal shift
    zoom_factor = 1 + (i * 0.01)  # gradual zoom
    dx = int(i * 2)               # horizontal shift

    # Resize image to simulate zoom
    zoomed = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    zh, zw, _ = zoomed.shape

    # Crop center to original size
    start_x = (zw - width) // 2
    start_y = (zh - height) // 2
    cropped = zoomed[start_y:start_y+height, start_x:start_x+width]

    # Shift horizontally
    shifted = np.roll(cropped, dx, axis=1)

    # Blend with PCA-transformed image
    frame = cv2.addWeighted(shifted, 1 - alpha, reconstructed_img, alpha, 0)
    video_writer.write(frame)

video_writer.release()
print("✅ Video saved as output_video.avi")
