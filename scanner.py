import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class DocumentScanner:
    def __init__(self, reference_descriptors:str="ref_descriptors.npy"):
        self.reference_descriptors = np.load(reference_descriptors)

    def transform_image(self, image: str | cv2.typing.MatLike):
        
        if isinstance(image, str):
            original_image = cv2.imread(image)
            gray_image = cv2.imread(image, 0)
        elif isinstance(image, cv2.typing.MatLike):
            original_image = image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        preprocessed_image = self.preprocess_image(gray_image)

        keypoints, descriptors = self.detect_harris_keypoints(
            preprocessed_image)
        corner_points = self.get_corner_points(
            preprocessed_image.shape, keypoints, descriptors)

        if len(corner_points) == 4:
            return self._apply_perspective_transform(original_image, corner_points)
        else:
            return None

    def preprocess_image(self, image):
        thresholded = self._threshold_image(image)
        morphed = self._apply_morphology(thresholded)
        return morphed

    def _threshold_image(self, image):
        h, w = image.shape[:2]
        scale = min(h, w) / 1000

        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
        smooth_sigma = max(1, int(5 * scale))
        smoothed_hist = gaussian_filter1d(hist, sigma=smooth_sigma)

        peak_distance = int(20 * scale)
        peaks, _ = find_peaks(smoothed_hist, height=0.1 *
                              np.max(smoothed_hist), distance=peak_distance)

        if len(peaks) >= 2:
            peak1, peak2 = sorted(
                peaks, key=lambda p: smoothed_hist[p], reverse=True)[:2]
            start, end = min(peak1, peak2), max(peak1, peak2)
            threshold = start + np.argmin(smoothed_hist[start:end + 1])
            _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _apply_morphology(self, image):
        h, w = image.shape[:2]
        scale = min(h, w) / 1000

        large_kernel_size = max(5, int(30 * scale))
        small_kernel_size = max(3, int(7 * scale))

        large_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (large_kernel_size, large_kernel_size))
        small_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (small_kernel_size, small_kernel_size))

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, small_kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, large_kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, large_kernel)

        return image

    def detect_harris_keypoints(self, image):
        gray = np.float32(image)
        harris_response = cv2.cornerHarris(gray, 20, 5, 0.06)
        harris_response = cv2.dilate(harris_response, None)
        threshold = 0.05 * harris_response.max()
        coords = np.argwhere(harris_response > threshold)

        response_values = [harris_response[y, x] for y, x in coords]
        sorted_indices = np.argsort(response_values)[::-1]

        keypoints = []
        selected_points = []
        window_size = 60
        for idx in sorted_indices:
            y, x = coords[idx]
            if not any(abs(x - sx) < window_size and abs(y - sy) < window_size for sx, sy in selected_points):
                selected_points.append((x, y))
                keypoints.append(cv2.KeyPoint(float(x), float(y), 20))

        sift = cv2.SIFT_create()
        return sift.compute(image, keypoints)

    def get_corner_points(self, image_shape, keypoints, descriptors):
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        h, w = image_shape
        corners = {"tl": None, "tr": None, "bl": None, "br": None}
        best_dists = {q: float('inf') for q in corners}

        for kp, desc in zip(keypoints, descriptors):
            x, y = kp.pt
            matches = matcher.knnMatch(desc.reshape(
                1, -1), self.reference_descriptors, k=1)
            if not matches or not matches[0]:
                continue
            dist = matches[0][0].distance

            if x < w/2 and y < h/2 and dist < best_dists["tl"]:
                corners["tl"], best_dists["tl"] = kp, dist
            elif x >= w/2 and y < h/2 and dist < best_dists["tr"]:
                corners["tr"], best_dists["tr"] = kp, dist
            elif x < w/2 and y >= h/2 and dist < best_dists["bl"]:
                corners["bl"], best_dists["bl"] = kp, dist
            elif x >= w/2 and y >= h/2 and dist < best_dists["br"]:
                corners["br"], best_dists["br"] = kp, dist

        return [p.pt for p in corners.values() if p is not None]

    def _apply_perspective_transform(self, image, points):
        points = np.array(points, dtype="float32")
        tl, tr, bl, br = points
        width = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))
        height = max(int(np.linalg.norm(bl - tl)),
                     int(np.linalg.norm(br - tr)))
        dst = np.array([[0, 0], [width - 1, 0], [0, height - 1],
                       [width - 1, height - 1]], dtype="float32")
        warped = cv2.warpPerspective(
            image, cv2.getPerspectiveTransform(points, dst), (width, height))
        return warped
    
