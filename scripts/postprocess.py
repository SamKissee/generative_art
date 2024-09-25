import cv2
import numpy as np
import os

# Apply glitch effect to the generated images
def glitch_effect(image_path, output_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # Randomly shuffle some rows to create glitch effect
    for i in range(10):
        start_row = np.random.randint(0, h)
        end_row = start_row + np.random.randint(5, 20)
        img[start_row:end_row] = np.roll(img[start_row:end_row], np.random.randint(0, w), axis=1)

    cv2.imwrite(output_path, img)

# Apply post-processing to all generated images
def process_images():
    input_dir = 'outputs/'
    output_dir = 'outputs/glitch/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            glitch_effect(os.path.join(input_dir, filename), os.path.join(output_dir, 'glitch_' + filename))

process_images()
