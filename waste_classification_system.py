import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from flask import Flask, request, render_template_string
from markupsafe import Markup
import base64
from io import BytesIO
from PIL import Image
import imgaug.augmenters as iaa

app = Flask(__name__)


def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_glcm(image, distance=1, angle=0):
    rows, cols = image.shape
    max_gray = 256
    glcm = np.zeros((max_gray, max_gray), dtype=np.float64)

    if angle == 0:
        dx, dy = distance, 0
    elif angle == 90:
        dx, dy = 0, -distance
    elif angle == 180:
        dx, dy = -distance, 0
    elif angle == 270:
        dx, dy = 0, distance
    else:
        raise ValueError("Only angles 0, 90, 180, 270 are supported.")

    for i in range(rows):
        for j in range(cols):
            x, y = j + dx, i + dy
            if 0 <= x < cols and 0 <= y < rows:
                row_val = image[i, j]
                col_val = image[y, x]
                glcm[row_val, col_val] += 1

    glcm_sum = glcm.sum()
    if glcm_sum != 0:
        glcm /= glcm_sum

    return glcm


def extract_features_from_glcm(glcm):
    levels = glcm.shape[0]
    contrast = dissimilarity = homogeneity = energy = correlation = asm = 0.0

    mean_i = np.sum([i * np.sum(glcm[i, :]) for i in range(levels)])
    mean_j = np.sum([j * np.sum(glcm[:, j]) for j in range(levels)])
    std_i = np.sqrt(np.sum([(i - mean_i) ** 2 * np.sum(glcm[i, :]) for i in range(levels)]))
    std_j = np.sqrt(np.sum([(j - mean_j) ** 2 * np.sum(glcm[:, j]) for j in range(levels)]))

    for i in range(levels):
        for j in range(levels):
            val = glcm[i, j]
            diff = i - j
            contrast += diff ** 2 * val
            dissimilarity += abs(diff) * val
            homogeneity += val / (1 + diff ** 2)
            energy += val ** 2
            asm += val ** 2
            if std_i > 0 and std_j > 0:
                correlation += ((i - mean_i) * (j - mean_j) * val)

    if std_i > 0 and std_j > 0:
        correlation /= (std_i * std_j)
    else:
        correlation = 0.0

    return [energy, contrast, dissimilarity, homogeneity, correlation, asm]


def extract_glcm_features(data):
    features = []
    for img in data:
        glcm = compute_glcm(img, distance=1, angle=0)
        feat = extract_features_from_glcm(glcm)
        features.append(feat)
    return np.array(features)


def load_data(data_dir):
    images = []
    labels = []
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    for category in categories:
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = preprocess_image(img)
            images.append(img)
            labels.append(category)

    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
    ])
    augmented_images = augmentation(images=images)
    images += list(augmented_images)
    labels += labels

    return np.array(images), np.array(labels)


def knn_predict(test_feat, train_feats, train_labels, k=1):
    distances = [np.linalg.norm(test_feat - feat) for feat in train_feats]
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = [train_labels[i] for i in nearest_indices]
    return max(set(nearest_labels), key=nearest_labels.count)


def train_model(X_train, y_train):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    features = extract_glcm_features(X_train)
    return features, y_train_encoded, le


def evaluate_model(X_test, y_test, train_feats, train_labels, le):
    test_feats = extract_glcm_features(X_test)
    y_test_encoded = le.transform(y_test)
    predictions = [knn_predict(test_feat, train_feats, train_labels) for test_feat in test_feats]
    accuracy = accuracy_score(y_test_encoded, predictions)
    return accuracy


def classify_image(image, train_feats, train_labels, le):
    image = preprocess_image(image)
    glcm_features = extract_glcm_features([image])[0]
    prediction = knn_predict(glcm_features, train_feats, train_labels)
    return le.inverse_transform([prediction])[0]


data_dir = 'archive/Garbage classification/Garbage classification'  # Adjusted to match TrashNet structure
X, y = load_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_feats, train_labels, le = train_model(X_train, y_train)
accuracy = evaluate_model(X_test, y_test, train_feats, train_labels, le)

# HTML template

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trash Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h2, h3, h4 { text-align: center; }
        form { text-align: center; margin: 20px 0; }
        input[type="file"], input[type="submit"] { margin: 10px; }
        .result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; text-align: center; }
        img { max-width: 300px; margin: 10px auto; display: block; }
    </style>
</head>
<body>
    <h2>Upload an Image to Classify Trash</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><br>
        <input type="submit" value="Classify Image">
    </form>
    {% if prediction %}
    <div class="result">
        <h3>Predicted Class: {{ prediction }}</h3>
        <h4>Model Accuracy: {{ accuracy }}%</h4>
        {{ image_data | safe }}
        <script>
            function speak(text) {
                if ('speechSynthesis' in window) {
                    const synth = window.speechSynthesis;
                    const utterance = new SpeechSynthesisUtterance(text);
                    synth.speak(utterance);
                } else {
                    console.log("Speech Synthesis not supported.");
                }
            }
            speak("تم تصنيف الصورة على أنها: {{ prediction }}");
        </script>
    </div>
    {% endif %}
</body>
</html>
"""

# Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_data = None
    template_accuracy = round(accuracy * 100, 2)
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            try:
                img = Image.open(file.stream)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                prediction = classify_image(img, train_feats, train_labels, le)
                _, buffer = cv2.imencode('.jpg', img)
                img_str = base64.b64encode(buffer).decode('utf-8')
                image_data = Markup(f'<img src="data:image/jpeg;base64,{img_str}" width="300">')
            except Exception as e:
                prediction = f"Error: {str(e)}"
    return render_template_string(HTML_TEMPLATE, prediction=prediction, image_data=image_data, accuracy=template_accuracy)

if __name__ == '__main__':
    app.run(debug=False)