# Image Enhancer

This project uses a TensorFlow Lite model to enhance images. It provides a simple Flask web interface where users can upload an image and download the enhanced version of the image.

## Features

- Image enhancement using a TensorFlow Lite model
- Simple Flask web interface for uploading and downloading images
- Color correction to remove color casts
- Grid pattern removal
- Image upscaling for better quality

## Installation

1. Clone this repository:
    ```
    [git clone https://github.com/yourusername/image-enhancer.git](https://github.com/SydneyWamalwa/image-enhancer.git)
    ```
2. Navigate to the project directory:
    ```
    cd image-enhancer
    ```
3. Install the required Python packages:
    ```
    pip install -r requirements.txt
    ```
4. Download the TensorFlow Lite model and place it in the project directory.

## Usage

1. Start the Flask app:
    ```
    python app.py
    ```
2. Open a web browser and navigate to `http://localhost:5000`.
3. Upload an image using the web interface.
4. Download the enhanced image.

## Contributing

Contributions are welcome! Please read the contributing guidelines first.

## License

This project is licensed under the terms of the MIT license. See the LICENSE file for details.
