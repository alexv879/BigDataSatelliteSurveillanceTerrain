"""
Advanced Data Validation and Quality Checks
Comprehensive data validation pipeline for satellite imagery
"""
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
from loguru import logger
import json
from dataclasses import dataclass, asdict
from PIL import Image
import hashlib


@dataclass
class DataQualityReport:
    """Data quality report"""
    total_images: int
    valid_images: int
    corrupt_images: int
    duplicate_images: int
    size_issues: int
    format_issues: int
    channel_issues: int
    class_distribution: Dict[str, int]
    mean_resolution: Tuple[float, float]
    mean_file_size: float
    issues: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DataValidator:
    """Comprehensive data validation"""

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (64, 64),
        max_resolution: Tuple[int, int] = (4096, 4096),
        allowed_formats: List[str] = ['.jpg', '.jpeg', '.png', '.tif', '.tiff'],
        expected_channels: int = 3
    ):
        """
        Initialize data validator

        Args:
            min_resolution: Minimum image resolution
            max_resolution: Maximum image resolution
            allowed_formats: Allowed image formats
            expected_channels: Expected number of channels
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.allowed_formats = allowed_formats
        self.expected_channels = expected_channels

        logger.info("Data validator initialized")

    def validate_image(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate single image

        Args:
            image_path: Path to image

        Returns:
            (is_valid, error_message)
        """
        try:
            # Check file exists
            if not image_path.exists():
                return False, "File does not exist"

            # Check format
            if image_path.suffix.lower() not in self.allowed_formats:
                return False, f"Invalid format: {image_path.suffix}"

            # Try to load image
            image = Image.open(image_path)
            image_array = np.array(image)

            # Check resolution
            height, width = image_array.shape[:2]
            if height < self.min_resolution[0] or width < self.min_resolution[1]:
                return False, f"Resolution too low: {width}x{height}"

            if height > self.max_resolution[0] or width > self.max_resolution[1]:
                return False, f"Resolution too high: {width}x{height}"

            # Check channels
            if len(image_array.shape) == 2:
                channels = 1
            else:
                channels = image_array.shape[2]

            if channels != self.expected_channels and channels not in [1, 3, 4]:
                return False, f"Invalid channels: {channels}"

            # Check for corruption
            if image_array.size == 0:
                return False, "Empty image"

            # Check for extreme values
            if np.all(image_array == 0) or np.all(image_array == 255):
                return False, "Uniform pixel values"

            return True, None

        except Exception as e:
            return False, f"Error loading image: {str(e)}"

    def validate_dataset(
        self,
        data_dir: str,
        check_duplicates: bool = True
    ) -> DataQualityReport:
        """
        Validate entire dataset

        Args:
            data_dir: Data directory
            check_duplicates: Check for duplicate images

        Returns:
            Data quality report
        """
        logger.info(f"Validating dataset: {data_dir}")

        data_path = Path(data_dir)
        image_paths = []

        # Collect all image paths
        for ext in self.allowed_formats:
            image_paths.extend(data_path.rglob(f"*{ext}"))

        total_images = len(image_paths)
        valid_images = 0
        corrupt_images = 0
        size_issues = 0
        format_issues = 0
        channel_issues = 0
        issues = []
        class_distribution = {}
        resolutions = []
        file_sizes = []

        # Track duplicates
        image_hashes = set() if check_duplicates else None
        duplicate_images = 0

        # Validate each image
        for image_path in image_paths:
            is_valid, error = self.validate_image(image_path)

            if is_valid:
                valid_images += 1

                # Track class distribution
                class_name = image_path.parent.name
                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1

                # Track resolution
                try:
                    img = Image.open(image_path)
                    resolutions.append(img.size)
                    file_sizes.append(image_path.stat().st_size / 1024)  # KB

                    # Check for duplicates
                    if check_duplicates:
                        img_hash = self._compute_image_hash(image_path)
                        if img_hash in image_hashes:
                            duplicate_images += 1
                            issues.append(f"Duplicate: {image_path}")
                        else:
                            image_hashes.add(img_hash)

                except Exception as e:
                    logger.warning(f"Could not analyze {image_path}: {e}")

            else:
                corrupt_images += 1
                issues.append(f"{image_path}: {error}")

                if "resolution" in error.lower():
                    size_issues += 1
                elif "format" in error.lower():
                    format_issues += 1
                elif "channel" in error.lower():
                    channel_issues += 1

        # Compute statistics
        mean_resolution = (
            np.mean([r[0] for r in resolutions]),
            np.mean([r[1] for r in resolutions])
        ) if resolutions else (0, 0)

        mean_file_size = np.mean(file_sizes) if file_sizes else 0

        # Create report
        report = DataQualityReport(
            total_images=total_images,
            valid_images=valid_images,
            corrupt_images=corrupt_images,
            duplicate_images=duplicate_images,
            size_issues=size_issues,
            format_issues=format_issues,
            channel_issues=channel_issues,
            class_distribution=class_distribution,
            mean_resolution=mean_resolution,
            mean_file_size=mean_file_size,
            issues=issues[:100]  # Limit to first 100 issues
        )

        logger.info(f"Validation complete: {valid_images}/{total_images} valid")
        return report

    def _compute_image_hash(self, image_path: Path) -> str:
        """Compute perceptual hash of image"""
        try:
            img = cv2.imread(str(image_path))
            img = cv2.resize(img, (8, 8))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg = img.mean()
            diff = img > avg
            return hashlib.md5(diff.tobytes()).hexdigest()
        except:
            return hashlib.md5(str(image_path).encode()).hexdigest()


class DataAnomalyDetector:
    """Detect anomalies in dataset"""

    def __init__(self):
        self.statistics = {}

    def fit(self, data_dir: str):
        """
        Fit on dataset to learn statistics

        Args:
            data_dir: Data directory
        """
        logger.info("Learning dataset statistics...")

        image_paths = list(Path(data_dir).rglob("*.jpg"))
        image_paths.extend(Path(data_dir).rglob("*.png"))

        # Collect statistics
        brightness_values = []
        contrast_values = []
        sizes = []

        for path in image_paths[:1000]:  # Sample first 1000
            try:
                img = cv2.imread(str(path))
                if img is not None:
                    brightness = img.mean()
                    contrast = img.std()
                    brightness_values.append(brightness)
                    contrast_values.append(contrast)
                    sizes.append(img.shape[:2])
            except:
                continue

        self.statistics = {
            'brightness_mean': np.mean(brightness_values),
            'brightness_std': np.std(brightness_values),
            'contrast_mean': np.mean(contrast_values),
            'contrast_std': np.std(contrast_values),
            'size_mean': np.mean(sizes, axis=0) if sizes else [0, 0]
        }

        logger.info("Statistics learned")

    def detect_anomalies(
        self,
        image_path: str,
        threshold: float = 3.0
    ) -> Tuple[bool, List[str]]:
        """
        Detect if image is anomalous

        Args:
            image_path: Path to image
            threshold: Z-score threshold for anomaly

        Returns:
            (is_anomaly, reasons)
        """
        reasons = []

        try:
            img = cv2.imread(image_path)
            if img is None:
                return True, ["Could not load image"]

            # Check brightness
            brightness = img.mean()
            z_brightness = abs(
                (brightness - self.statistics['brightness_mean']) /
                (self.statistics['brightness_std'] + 1e-6)
            )

            if z_brightness > threshold:
                reasons.append(f"Unusual brightness (z={z_brightness:.2f})")

            # Check contrast
            contrast = img.std()
            z_contrast = abs(
                (contrast - self.statistics['contrast_mean']) /
                (self.statistics['contrast_std'] + 1e-6)
            )

            if z_contrast > threshold:
                reasons.append(f"Unusual contrast (z={z_contrast:.2f})")

            return len(reasons) > 0, reasons

        except Exception as e:
            return True, [f"Error: {str(e)}"]


class ClassBalanceAnalyzer:
    """Analyze class balance in dataset"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.class_counts = {}

    def analyze(self) -> Dict[str, any]:
        """
        Analyze class distribution

        Returns:
            Analysis results
        """
        data_path = Path(self.data_dir)

        # Count images per class
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                image_count = len(list(class_dir.glob("*.jpg")))
                image_count += len(list(class_dir.glob("*.png")))
                self.class_counts[class_dir.name] = image_count

        if not self.class_counts:
            return {"error": "No classes found"}

        # Compute statistics
        counts = list(self.class_counts.values())
        total = sum(counts)

        analysis = {
            'class_counts': self.class_counts,
            'total_images': total,
            'num_classes': len(self.class_counts),
            'min_samples': min(counts),
            'max_samples': max(counts),
            'mean_samples': np.mean(counts),
            'std_samples': np.std(counts),
            'imbalance_ratio': max(counts) / (min(counts) + 1),
            'is_balanced': max(counts) / (min(counts) + 1) < 3.0
        }

        logger.info(f"Class balance analysis: {analysis['imbalance_ratio']:.2f}x imbalance")
        return analysis

    def suggest_weights(self) -> Dict[str, float]:
        """Suggest class weights for balanced training"""
        if not self.class_counts:
            return {}

        total = sum(self.class_counts.values())
        num_classes = len(self.class_counts)

        weights = {}
        for class_name, count in self.class_counts.items():
            weight = total / (num_classes * count)
            weights[class_name] = weight

        logger.info("Class weights computed")
        return weights


def validate_and_report(
    data_dir: str,
    output_path: str = "validation/reports/data_quality.json"
):
    """
    Run complete validation and generate report

    Args:
        data_dir: Data directory
        output_path: Output path for report
    """
    # Data validation
    validator = DataValidator()
    quality_report = validator.validate_dataset(data_dir)

    # Class balance analysis
    balance_analyzer = ClassBalanceAnalyzer(data_dir)
    balance_report = balance_analyzer.analyze()

    # Combined report
    full_report = {
        'data_quality': quality_report.to_dict(),
        'class_balance': balance_report,
        'recommendations': []
    }

    # Add recommendations
    if quality_report.corrupt_images > 0:
        full_report['recommendations'].append(
            f"Remove or fix {quality_report.corrupt_images} corrupt images"
        )

    if quality_report.duplicate_images > 0:
        full_report['recommendations'].append(
            f"Remove {quality_report.duplicate_images} duplicate images"
        )

    if balance_report.get('imbalance_ratio', 0) > 3.0:
        full_report['recommendations'].append(
            "Dataset is imbalanced - consider using class weights or oversampling"
        )

    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Validation report saved to {output_path}")
    return full_report


if __name__ == "__main__":
    print("Data Validation Tools Ready")
    print("Features:")
    print("- Image integrity validation")
    print("- Duplicate detection")
    print("- Anomaly detection")
    print("- Class balance analysis")
    print("- Comprehensive quality reports")
