from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import io
import base64
import cv2
import os

app = Flask(__name__)


class HOGDescriptor:
    def __init__(self):
        self.cell_size = (8, 8)
        self.block_size = (2, 2)
        self.n_bins = 9
        self.scaler = StandardScaler()
        self.clf = LinearDiscriminantAnalysis()

    def compute_gradients(self, img):
        img = self.resize_image(img, (64, 128))
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        dx = np.zeros_like(gray)
        dy = np.zeros_like(gray)
        dx[:, 1:-1] = gray[:, :-2] - gray[:, 2:]
        dy[1:-1, :] = gray[:-2, :] - gray[2:, :]
        dx[:, 0] = dx[:, 1]
        dx[:, -1] = dx[:, -2]
        dy[0, :] = dy[1, :]
        dy[-1, :] = dy[-2, :]

        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx) * (180 / np.pi) % 180
        return magnitude, angle

    def resize_image(self, img, size):
        h, w = img.shape[:2]
        target_w, target_h = size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = np.zeros((target_h, target_w, img.shape[2]) if len(img.shape) == 3 else (target_h, target_w))
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        for i in range(new_h):
            for j in range(new_w):
                src_x = int(j / scale)
                src_y = int(i / scale)
                if len(img.shape) == 3:
                    resized[start_y + i, start_x + j, :] = img[src_y, src_x, :]
                else:
                    resized[start_y + i, start_x + j] = img[src_y, src_x]
        return resized

    def create_histograms(self, magnitude, angle):
        rows, cols = magnitude.shape
        cell_rows = rows // self.cell_size[0]
        cell_cols = cols // self.cell_size[1]
        histograms = np.zeros((cell_rows, cell_cols, self.n_bins))
        bin_width = 180 / self.n_bins

        for i in range(cell_rows):
            for j in range(cell_cols):
                y_start = i * self.cell_size[0]
                y_end = y_start + self.cell_size[0]
                x_start = j * self.cell_size[1]
                x_end = x_start + self.cell_size[1]

                mag_cell = magnitude[y_start:y_end, x_start:x_end]
                ang_cell = angle[y_start:y_end, x_start:x_end]

                for mag, ang in zip(mag_cell.ravel(), ang_cell.ravel()):
                    ang %= 180
                    bin_idx = int(ang / bin_width)
                    next_bin = (bin_idx + 1) % self.n_bins
                    bin_start_angle = bin_idx * bin_width
                    ratio = (ang - bin_start_angle) / bin_width
                    histograms[i, j, bin_idx] += mag * (1 - ratio)
                    histograms[i, j, next_bin] += mag * ratio
        return histograms

    def block_normalization(self, histograms):
        cell_rows, cell_cols, _ = histograms.shape
        blocks = []
        for i in range(cell_rows - 1):
            for j in range(cell_cols - 1):
                block = histograms[i:i + 2, j:j + 2, :].ravel()
                norm = np.sqrt(np.sum(block ** 2) + 1e-6)
                normalized_block = block / norm
                blocks.append(normalized_block)
        return np.concatenate(blocks)

    def extract(self, img):
        magnitude, angle = self.compute_gradients(img)
        histograms = self.create_histograms(magnitude, angle)
        hog_features = self.block_normalization(histograms)
        return hog_features

    def train(self, X_train, y_train, X_test, y_test):
        X_train = self.scaler.fit_transform(X_train)
        self.clf.fit(X_train, y_train)
        X_test = self.scaler.transform(X_test)
        y_pred = self.clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.clf.predict(X)


def load_dataset(data_path, class_folders, samples_per_class=50):
    hog = HOGDescriptor()
    features = []
    labels = []
    for label, folder in enumerate(class_folders):
        folder_path = os.path.join(data_path, folder)
        count = 0
        for filename in os.listdir(folder_path):
            if count >= samples_per_class:
                break
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    hog_features = hog.extract(img)
                    features.append(hog_features)
                    labels.append(label)
                    count += 1
    X = np.array(features)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def initialize_classifier():
    hog = HOGDescriptor()
    DATA_PATH = "FMD/image"  # Adjust to your FMD dataset path
    CLASSES = ['fabric', 'wood', 'glass']
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset path {DATA_PATH} not found. Please ensure the FMD dataset is available.")
    X_train, X_test, y_train, y_test = load_dataset(DATA_PATH, CLASSES, samples_per_class=50)
    accuracy = hog.train(X_train, y_train, X_test, y_test)
    return hog, accuracy


def classify_image(image_data):
    hog, _ = initialize_classifier()
    img = np.array(image_data, dtype=np.uint8)
    features = hog.extract(img)
    prediction = hog.predict(features.reshape(1, -1))[0]
    classes = ['Fabric', 'Wood', 'Glass']
    return classes[prediction]


# Store accuracy globally for rendering in template
hog, model_accuracy = initialize_classifier()


@app.route('/')
def index():
    return render_template('index.html', accuracy=model_accuracy * 100)


@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(file).convert('RGB')
        img_array = np.array(img)

        # Classify the image
        result = classify_image(img_array)

        # Convert image to base64 for display
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'class': result,
            'image': f'data:image/png;base64,{img_str}',
            'accuracy': model_accuracy * 100
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)