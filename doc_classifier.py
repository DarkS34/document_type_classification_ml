import argparse
import os
import cv2
import numpy as np
from joblib import load, dump
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from scanner import DocumentScanner

# === CONFIGURATION ===
IMG_SIZE = (400, 300)
CLASS_MAP = {
    'comics': 0,
    'libros': 1,
    'manuscrito': 2,
    'mecanografiado': 3,
    'tickets': 4
}
CLASS_MAP_INV = {v: k for k, v in CLASS_MAP.items()}

BEST_CLASSIFIER = 1
TRAIN_DIR = "./data/Learning"
TEST_DIR = "./data/Test"

# Classifier configurations
CLASSIFIER_CONFIGS = {
    'C1': {'use_lda': False, 'use_scanner': False, 'to_gray': False},
    'C2': {'use_lda': True, 'use_scanner': False, 'to_gray': False},
    'C3': {'use_lda': False, 'use_scanner': True, 'to_gray': True},
    'C4': {'use_lda': True, 'use_scanner': True, 'to_gray': True}
}


class DocumentClassifier:
    def __init__(self, model_path=None, use_lda=False, lda_path=None, use_scanner=False, scaler_path=None):
        self.model = load(model_path) if model_path else None
        self.use_lda = use_lda
        self.lda = load(lda_path) if use_lda and lda_path else None
        self.use_scanner = use_scanner
        self.scanner = DocumentScanner() if use_scanner else None
        self.scaler = load(scaler_path) if scaler_path else StandardScaler()

    def preprocess_image(self, image_path):
        """Preprocesses an image for prediction"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        if self.use_scanner and self.scanner:
            img = self.scanner.transform_image(image_path)
            if img is None:
                raise ValueError("Could not rectify image")

        img = self._apply_image_transforms(img, self.use_scanner)
        features = self._extract_features(img)
        return features

    def _apply_image_transforms(self, img, to_gray=False):
        """Applies image transformations based on configuration"""
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE)
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.resize(img, IMG_SIZE)
        return img

    def _extract_features(self, img):
        """Extracts features from an image"""
        features = img.reshape(1, -1).astype(np.float32)
        features = self.scaler.transform(features)
        if self.use_lda and self.lda:
            features = self.lda.transform(features)
        return features

    def predict(self, image_path):
        """Predicts the class of an image"""
        features = self.preprocess_image(image_path)
        label = self.model.predict(features)[0]
        return CLASS_MAP_INV.get(label, "Unknown class")

    def train(self, images, labels, n_components=4):
        """Trains the classifier"""
        X = self._prepare_training_data(images, labels, n_components)
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X, labels)
        return self.model, self.lda if self.use_lda else self.model

    def _prepare_training_data(self, images, labels, n_components):
        """Prepares training data"""
        X = images.reshape((images.shape[0], -1)).astype(np.float32)
        X = self.scaler.fit_transform(X)
        if self.use_lda:
            self.lda = LinearDiscriminantAnalysis(n_components=n_components)
            X = self.lda.fit_transform(X, labels)
        return X

    def evaluate(self, images, labels):
        """Evaluates classifier performance"""
        X = self._prepare_evaluation_data(images)
        preds = self.model.predict(X)
        acc = accuracy_score(labels, preds)
        print("Accuracy:", acc)
        print(classification_report(labels, preds))
        return acc

    def _prepare_evaluation_data(self, images):
        """Prepares evaluation data"""
        X = images.reshape((images.shape[0], -1)).astype(np.float32)
        X = self.scaler.transform(X)
        if self.use_lda and self.lda:
            X = self.lda.transform(X)
        return X


def load_images_from_folder(folder):
    """Loads images from a folder organized by classes"""
    features, labels = [], []
    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        
        if class_name.lower() not in CLASS_MAP:
            print(f"Warning: Class '{class_name}' not recognized, skipping.")
            continue
            
        label = CLASS_MAP[class_name.lower()]
        for filename in os.listdir(class_folder):
            path = os.path.join(class_folder, filename)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                features.append(img)
                labels.append(label)
    return np.array(features), np.array(labels)


def process_images(images, labels, use_scanner=False, to_gray=False):
    """Processes images applying scanner and/or grayscale conversion"""
    if not use_scanner and not to_gray:
        return images, labels
    
    processed = []
    filtered_labels = []
    scanner = DocumentScanner() if use_scanner else None
    
    for image, label in zip(images, labels):
        processed_image = image
        
        if use_scanner and scanner:
            processed_image = scanner.transform_image(image)
            if processed_image is None:
                continue
        
        if to_gray:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            processed_image = np.expand_dims(processed_image, axis=-1)
        
        processed_image = cv2.resize(processed_image, IMG_SIZE)
        processed.append(processed_image)
        filtered_labels.append(label)
    
    return np.array(processed), np.array(filtered_labels)


def create_classifier(config, model_path=None, lda_path=None, scaler_path=None):
    """Creates a classifier based on specified configuration"""
    return DocumentClassifier(
        model_path=model_path,
        use_lda=config['use_lda'],
        lda_path=lda_path,
        use_scanner=config['use_scanner'],
        scaler_path=scaler_path
    )


def train_all_classifiers():
    """Trains all defined classifiers"""
    if not os.path.exists("./models/"):
        os.makedirs("./models/")
    training_images, training_labels = load_images_from_folder(TRAIN_DIR)
    
    for classifier_name, config in CLASSIFIER_CONFIGS.items():
        print(f"\033[92m[TRAINING ...]\033[0m Classifier {classifier_name}", flush=True, end="")
        
        # Process images according to configuration
        processed_images, processed_labels = process_images(
            training_images, training_labels, 
            config['use_scanner'], config['to_gray']
        )
        
        # Create and train classifier
        classifier = create_classifier(config)
        classifier.train(processed_images, processed_labels)
        
        # Save models
        model_path = f"./models/model_{classifier_name}.joblib"
        scaler_path = f"./models/scaler_{classifier_name}.joblib"
        dump(classifier.model, model_path)
        dump(classifier.scaler, scaler_path)
        
        if config['use_lda']:
            lda_path = f"./models/model_{classifier_name}_lda.joblib"
            dump(classifier.lda, lda_path)
        
        print(f"\r\033[92m[TRAINING COMPLETED]\033[0m Classifier {classifier_name}", flush=True)


def evaluate_all_classifiers():
    """Evaluates all defined classifiers"""
    test_images, test_labels = load_images_from_folder(TEST_DIR)
    results = {}
    
    for classifier_name, config in CLASSIFIER_CONFIGS.items():
        print(f"\033[92m[EVALUATING]\033[0m Classifier {classifier_name}")
        
        # Process test images according to configuration
        processed_images, processed_labels = process_images(
            test_images, test_labels,
            config['use_scanner'], config['to_gray']
        )
        
        # Load and evaluate classifier
        model_path = f"./models/model_{classifier_name}.joblib"
        scaler_path = f"./models/scaler_{classifier_name}.joblib"
        lda_path = f"./models/model_{classifier_name}_lda.joblib" if config['use_lda'] else None
        
        classifier = create_classifier(config, model_path, lda_path, scaler_path)
        accuracy = classifier.evaluate(processed_images, processed_labels)
        
        results[classifier_name] = accuracy
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document classification script")
    parser.add_argument('--train', dest="train_mode",
                        action="store_true", help='Training mode')
    parser.add_argument('--test', dest="test_mode",
                        action="store_true", help='Evaluation mode')
    parser.add_argument('path', nargs='?', help='Path to image for classification')
    args = parser.parse_args()

    if args.train_mode:
        train_all_classifiers()
    elif args.test_mode:
        results = evaluate_all_classifiers()
        print("\n=== RESULTS SUMMARY ===")
        for classifier_name, accuracy in results.items():
            print(f"{classifier_name}: {accuracy:.4f}")
    elif args.path:
        if os.path.exists(args.path):
            try:
                best_config = CLASSIFIER_CONFIGS[f'C{BEST_CLASSIFIER}']
                model_path = f"./models/model_C{BEST_CLASSIFIER}.joblib"
                scaler_path = f"./models/scaler_C{BEST_CLASSIFIER}.joblib"
                lda_path = f"./models/model_C{BEST_CLASSIFIER}_lda.joblib" if best_config['use_lda'] else None
                
                best_classifier = create_classifier(best_config, model_path, lda_path, scaler_path)
                
                print(f"Prediction: {best_classifier.predict(args.path)}")
            except Exception as e:
                print(f"Error processing image: {e}")
        else:
            print("The specified path does not exist.")
    else:
        parser.print_help()