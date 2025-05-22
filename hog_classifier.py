import cv2
import numpy as np
import os
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from flask import Flask, request, render_template
import base64
from io import BytesIO
from PIL import Image
from skimage.feature import local_binary_pattern  # إضافة ميزات LBP للمساعدة في التمييز

app = Flask(__name__)

# إعدادات التطبيق
CLASSES = ['glass', 'paper', 'plastic', 'metal', 'trash']  # تأكد من تطابقها مع مجلداتك
DATA_PATH =  "archive/Garbage classification/Garbage classification"


class AdvancedMaterialClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

    def extract_features(self, img):
        # الحفاظ على الألوان الأصلية للمساعدة في التمييز
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_hist = self.compute_color_histogram(img)
        else:
            gray = img
            color_hist = np.array([])

        # تحسين معالجة الصورة
        gray = cv2.equalizeHist(gray)

        # استخراج ميزات متعددة
        hog_features = self.compute_hog(gray)
        lbp_features = self.compute_lbp(gray)
        edge_features = self.compute_edge_stats(gray)

        # دمج جميع الميزات
        features = np.concatenate([
            hog_features,
            lbp_features,
            edge_features,
            color_hist
        ])

        return features

    def compute_hog(self, gray_img):
        gray_img = cv2.resize(gray_img, (128, 128))
        hog = cv2.HOGDescriptor(
            (128, 128),  # winSize
            (16, 16),  # blockSize
            (8, 8),  # blockStride
            (8, 8),  # cellSize
            9  # nbins
        )
        return hog.compute(gray_img).flatten()

    def compute_lbp(self, gray_img, radius=3, n_points=24):
        gray_img = cv2.resize(gray_img, (128, 128))
        lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
        return hist.astype(np.float32)

    def compute_edge_stats(self, gray_img):
        gray_img = cv2.resize(gray_img, (128, 128))
        edges = cv2.Canny(gray_img, 100, 200)
        return np.array([np.mean(edges), np.std(edges)])

    def compute_color_histogram(self, color_img, bins=8):
        if len(color_img.shape) == 2:
            return np.array([])

        color_img = cv2.resize(color_img, (128, 128))
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

        # حساب الهيستوجرام لكل قناة
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()

        # تطبيع الهيستوجرام
        h_hist = h_hist / (h_hist.sum() + 1e-6)
        s_hist = s_hist / (s_hist.sum() + 1e-6)
        v_hist = v_hist / (v_hist.sum() + 1e-6)

        return np.concatenate([h_hist, s_hist, v_hist])

    def train(self, X, y):
        X = self.scaler.fit_transform(X)
        self.clf.fit(X, y)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        return self.clf.predict_proba(X)


def load_balanced_dataset(data_path, class_folders, samples_per_class=200):
    classifier = AdvancedMaterialClassifier()
    features = []
    labels = []

    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(data_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Folder {class_path} does not exist!")
            continue

        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = image_files[:samples_per_class]

        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img_features = classifier.extract_features(img)
                features.append(img_features)
                labels.append(class_idx)

    return np.array(features), np.array(labels)


# تحميل البيانات وتدريب النموذج
print("Loading dataset...")
X, y = load_balanced_dataset(DATA_PATH, CLASSES)

if len(X) == 0:
    print("Error: No data loaded! Check your DATA_PATH and folder structure.")
    exit()

print(f"Loaded {len(X)} samples with {X.shape[1]} features each")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

classifier = AdvancedMaterialClassifier()
print("\nTraining classifier...")
classifier.train(X_train, y_train)

# تقييم النموذج
y_pred = classifier.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded", classes=CLASSES)

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No selected file", classes=CLASSES)

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # قراءة الصورة مع الحفاظ على الألوان
                img = Image.open(file.stream)
                img = np.array(img)

                # تصنيف الصورة
                features = classifier.extract_features(img)
                prediction = classifier.predict([features])[0]
                proba = classifier.predict_proba([features])[0]
                pred_class = CLASSES[prediction]

                # تحضير الصورة للعرض مع الحفاظ على الألوان
                display_img = cv2.resize(img, (256, 256))
                if len(display_img.shape) == 3:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                    _, buffer = cv2.imencode('.jpg', display_img)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                else:
                    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                    _, buffer = cv2.imencode('.jpg', display_img)
                    img_str = base64.b64encode(buffer).decode('utf-8')

                # تحضير نتائج الاحتمالات
                proba_results = {CLASSES[i]: f"{prob * 100:.1f}%" for i, prob in enumerate(proba)}

                return render_template('index.html',
                                       prediction=pred_class,
                                       probabilities=proba_results,
                                       image_data=img_str,
                                       classes=CLASSES)

            except Exception as e:
                print(f"Error: {str(e)}")
                return render_template('index.html',
                                       error="Error processing image",
                                       classes=CLASSES)

    return render_template('index.html', classes=CLASSES)


if __name__ == "__main__":
    app.run(debug=True)