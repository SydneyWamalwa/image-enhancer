from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def color_correction(image):
    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Apply color correction to reduce magenta cast
    # Convert image to LAB color space
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)

    # Split into channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Reshape L channel to match the shapes of A and B channels
    l = l.reshape(a.shape)

    # Merge channels back
    lab = cv2.merge((l, a, b))

    # Convert back to BGR color space
    image_cv_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Convert back to PIL format
    image_corrected = Image.fromarray(cv2.cvtColor(image_cv_corrected, cv2.COLOR_BGR2RGB))
    return image_corrected


def remove_grid_pattern(image):
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Convert image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply FFT to the image
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Create a mask to suppress the grid frequencies
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius of the grid frequency suppression
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    # Apply mask and inverse FFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Convert back to uint8
    img_back = np.uint8(img_back)

    # Convert back to PIL format
    image_denoised = Image.fromarray(img_back)
    return image_denoised

def enhance_image(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode

    # Resize the image to the expected dimensions
    input_shape = input_details[0]['shape'][1:3]
    image = image.resize(input_shape, Image.LANCZOS)

    # Normalize image to [-1, 1]
    image_array = np.array(image, dtype=np.float32) / 127.5 - 1.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Check if input shape matches the expected shape
    # Check if input shape matches the expected shape
    expected_shape = input_details[0]['shape'][1:3]
    if not np.all(image_array.shape[1:3] == expected_shape):
        raise ValueError(f"Input image shape {image_array.shape[1:3]} doesn't match expected shape {expected_shape}")


    # Set the tensor to point to the input data to be processed
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()

    # Get the enhanced image array
    enhanced_image_array = interpreter.get_tensor(output_details[0]['index'])

    # Remove the extra dimension
    enhanced_image_array = np.squeeze(enhanced_image_array, axis=0)

    # Denormalize the image array from [-1, 1] back to [0, 255]
    enhanced_image_array = (enhanced_image_array + 1.0) * 127.5
    enhanced_image_array = np.clip(enhanced_image_array, 0, 255).astype(np.uint8)

    # Convert back to PIL format
    enhanced_image_pil = Image.fromarray(enhanced_image_array)

    # Save the final enhanced image
    enhanced_image_path = image_path.replace('uploads', 'enhanced')

    # Create the enhanced folder if it doesn't exist
    if not os.path.exists('enhanced/'):
        os.makedirs('enhanced/')

    enhanced_image_pil.save(enhanced_image_path)  # Save the enhanced image
    return enhanced_image_path



@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML form for uploading images

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)  # Secure the filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  # Save the uploaded file
        enhanced_image_path = enhance_image(file_path)  # Enhance the image
        return send_file(enhanced_image_path, as_attachment=True)  # Send the enhanced image to the user

if __name__ == '__main__':
    app.run(debug=True)  # Run the app in debug mode
