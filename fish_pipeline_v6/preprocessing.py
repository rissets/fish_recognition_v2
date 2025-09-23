import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
import logging
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import detection & segmentation models
from models.detection.inference import YOLOInference
from models.segmentation.inference import Inference as SegmentationInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FishPreprocessor:
    def __init__(self):
        self.detection_model = YOLOInference(
            model_path=os.path.join('..','models','detection','model.ts'),
            imsz=(640,640), conf_threshold=0.05, nms_threshold=0.3, yolo_ver='v10')
        self.segmentation_model = SegmentationInference(
            model_path=os.path.join('..','models','segmentation','model.ts'),
            image_size=416, threshold=0.5)
        self.augment = A.Compose([
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.Rotate(limit=15),
            A.GaussianBlur(),
            A.ColorJitter(),
        ])

    def process_image(self, image_path, output_dir):
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f'Cannot read {image_path}')
            return []
        # Detection
        images = [image]
        pred, _ = self.detection_model.preprocess(images)
        with torch.no_grad():
            results = self.detection_model.model(pred)
        boxes = self.detection_model.v10postprocess(results[0])
        if len(boxes) > 0:
            x1, y1, x2, y2, conf = boxes[0]
            crop = image[int(y1):int(y2), int(x1):int(x2)]
        else:
            crop = image
        # Segmentation
        seg_poly = self.segmentation_model.predict(crop)[0]
        # Augment 10x
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for i in range(10):
            aug = self.augment(image=crop)['image']
            out_path = os.path.join(output_dir, f'{Path(image_path).stem}_aug_{i+1}.jpg')
            cv2.imwrite(out_path, aug)
            results.append(out_path)
        return results

if __name__ == '__main__':
    # Example usage
    input_folder = '../images'
    output_folder = 'output/preprocessed'
    os.makedirs(output_folder, exist_ok=True)
    preprocessor = FishPreprocessor()
    for img_file in Path(input_folder).glob('*.jpg'):
        logger.info(f'Processing {img_file}')
        preprocessor.process_image(str(img_file), output_folder)
