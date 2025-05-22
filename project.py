import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
from PIL import Image
import base64

# إعداد Flask
app = Flask(__name__)

# إعداد أسماء الفئات
CLASSES = ['glass', 'paper', 'plastic', 'metal']
DATA_PATH = 'archive/Garbage classification/Garbage classification'  # مسار مجلد البيانات

# مصنف الميكنة
class RecyclingClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SVC(kernel='rbf', probability=True)

    def extract_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor((128,128),(16,16),(8,8),(8,8),9)
        hog_feat = hog.compute(cv2.resize(gray, (128,128))).flatten()

        lbp = local_binary_pattern(gray, 24, 3, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0,26))
        lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-6)

        return np.concatenate([hog_feat, lbp_hist])

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, img):
        features = self.extract_features(img)
        features_scaled = self.scaler.transform([features])
        pred = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        return pred, proba


# تحميل البيانات وتدريب النموذج
def load_data():
    features, labels = [], []
    clf = RecyclingClassifier()
    for idx, class_name in enumerate(CLASSES):
        folder = os.path.join(DATA_PATH, class_name)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(img, (128, 128))
                feat = clf.extract_features(img)
                features.append(feat)
                labels.append(idx)
    return np.array(features), np.array(labels)


print("[INFO] Loading and training classifier...")
X, y = load_data()
classifier = RecyclingClassifier()
classifier.train(X, y)
print("[INFO] Training completed.")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_data = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file.stream).convert('RGB')
            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.resize(img_bgr, (128, 128))

            pred_class, proba = classifier.predict(img_bgr)
            prediction = CLASSES[pred_class]
            confidence = f"{np.max(proba)*100:.2f}%"

            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            image_data = base64.b64encode(buffer).decode('utf-8')

    return render_template('index.html', prediction=prediction,
                           confidence=confidence, image_data=image_data)


if __name__ == '__main__':
    app.run(debug=True)
