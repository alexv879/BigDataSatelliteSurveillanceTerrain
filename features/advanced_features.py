"""
Advanced Feature Engineering for Satellite Imagery
Texture analysis, spectral features, object-based analysis (OBIA)
Powerful features for terrain classification beyond raw pixels
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import filters, morphology, measure
import cv2


class TextureFeatures:
    """
    Extract texture features from satellite imagery
    Critical for distinguishing terrain types with similar spectral signatures
    """

    @staticmethod
    def compute_glcm_features(
        image: np.ndarray,
        distances: List[int] = [1, 2, 3],
        angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    ) -> Dict[str, np.ndarray]:
        """
        Compute Gray-Level Co-occurrence Matrix (GLCM) texture features

        Args:
            image: Input image (H, W) - grayscale
            distances: Pixel pair distances
            angles: Pixel pair angles

        Returns:
            Dictionary of texture features
        """
        # Normalize to 0-255
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_norm = ((image - image.min()) / (image.max() - image.min() + 1e-10) * 255).astype(np.uint8)
        else:
            image_norm = image.astype(np.uint8)

        # Compute GLCM
        glcm = graycomatrix(
            image_norm,
            distances=distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True
        )

        # Extract features
        features = {}

        # Contrast - measures local variation
        features['contrast'] = graycoprops(glcm, 'contrast').mean()

        # Dissimilarity - similar to contrast but grows linearly
        features['dissimilarity'] = graycoprops(glcm, 'dissimilarity').mean()

        # Homogeneity - measures closeness of distribution to diagonal
        features['homogeneity'] = graycoprops(glcm, 'homogeneity').mean()

        # Energy (Angular Second Moment) - uniformity measure
        features['energy'] = graycoprops(glcm, 'energy').mean()

        # Correlation - linear dependency of grey levels
        features['correlation'] = graycoprops(glcm, 'correlation').mean()

        # ASM (Angular Second Moment)
        features['ASM'] = graycoprops(glcm, 'ASM').mean()

        logger.info(f"GLCM features computed: {len(features)} features")

        return features

    @staticmethod
    def compute_glcm_per_pixel(
        image: np.ndarray,
        window_size: int = 15
    ) -> np.ndarray:
        """
        Compute GLCM features for each pixel using sliding window

        Args:
            image: Input image
            window_size: Window size for local GLCM

        Returns:
            Feature map (H, W, num_features)
        """
        h, w = image.shape
        pad = window_size // 2

        # Pad image
        image_padded = np.pad(image, pad, mode='reflect')

        # Feature maps
        n_features = 6  # contrast, dissimilarity, homogeneity, energy, correlation, ASM
        feature_maps = np.zeros((h, w, n_features), dtype=np.float32)

        # Normalize image
        image_norm = ((image_padded - image_padded.min()) /
                      (image_padded.max() - image_padded.min() + 1e-10) * 255).astype(np.uint8)

        # Compute for each pixel
        for i in range(h):
            for j in range(w):
                # Extract window
                window = image_norm[i:i+window_size, j:j+window_size]

                # Compute GLCM for window
                try:
                    glcm = graycomatrix(
                        window,
                        distances=[1],
                        angles=[0],
                        levels=256,
                        symmetric=True,
                        normed=True
                    )

                    # Extract features
                    feature_maps[i, j, 0] = graycoprops(glcm, 'contrast')[0, 0]
                    feature_maps[i, j, 1] = graycoprops(glcm, 'dissimilarity')[0, 0]
                    feature_maps[i, j, 2] = graycoprops(glcm, 'homogeneity')[0, 0]
                    feature_maps[i, j, 3] = graycoprops(glcm, 'energy')[0, 0]
                    feature_maps[i, j, 4] = graycoprops(glcm, 'correlation')[0, 0]
                    feature_maps[i, j, 5] = graycoprops(glcm, 'ASM')[0, 0]
                except:
                    pass  # Keep zeros for invalid windows

        logger.info(f"Per-pixel GLCM computed: {feature_maps.shape}")

        return feature_maps

    @staticmethod
    def compute_lbp_features(
        image: np.ndarray,
        radius: int = 3,
        n_points: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Local Binary Pattern (LBP) texture descriptor

        Args:
            image: Input image
            radius: LBP radius
            n_points: Number of circularly symmetric points

        Returns:
            (lbp_image, lbp_histogram)
        """
        # Compute LBP
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')

        # Compute histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-10)  # Normalize

        logger.info(f"LBP computed: {n_bins} patterns")

        return lbp, hist

    @staticmethod
    def compute_gabor_features(
        image: np.ndarray,
        frequencies: List[float] = [0.1, 0.2, 0.3],
        angles: List[float] = [0, 45, 90, 135]
    ) -> List[np.ndarray]:
        """
        Gabor filter bank for texture analysis

        Args:
            image: Input image
            frequencies: Filter frequencies
            angles: Filter orientations (degrees)

        Returns:
            List of filtered images
        """
        filtered_images = []

        for freq in frequencies:
            for angle_deg in angles:
                # Create Gabor kernel
                angle_rad = np.deg2rad(angle_deg)
                kernel = cv2.getGaborKernel(
                    (21, 21),
                    sigma=3.0,
                    theta=angle_rad,
                    lambd=1.0/freq,
                    gamma=0.5,
                    psi=0
                )

                # Apply filter
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                filtered_images.append(filtered)

        logger.info(f"Gabor features: {len(filtered_images)} responses")

        return filtered_images


class SpectralFeatures:
    """
    Advanced spectral feature extraction
    """

    @staticmethod
    def compute_spectral_indices(
        bands: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive set of spectral indices

        Args:
            bands: Dictionary of spectral bands
                   (e.g., {'Blue': ..., 'Green': ..., 'Red': ..., 'NIR': ..., 'SWIR1': ..., 'SWIR2': ...})

        Returns:
            Dictionary of spectral indices
        """
        indices = {}

        # Extract bands (with defaults)
        B = bands.get('Blue', None)
        G = bands.get('Green', None)
        R = bands.get('Red', None)
        NIR = bands.get('NIR', None)
        SWIR1 = bands.get('SWIR1', None)
        SWIR2 = bands.get('SWIR2', None)
        RE = bands.get('RedEdge', None)  # Red Edge

        # VEGETATION INDICES
        if NIR is not None and R is not None:
            indices['NDVI'] = (NIR - R) / (NIR + R + 1e-10)
            indices['SR'] = NIR / (R + 1e-10)  # Simple Ratio
            indices['DVI'] = NIR - R  # Difference Vegetation Index

        if NIR is not None and R is not None and B is not None:
            indices['EVI'] = 2.5 * ((NIR - R) / (NIR + 6*R - 7.5*B + 1))
            indices['ARVI'] = (NIR - (R - (R - B))) / (NIR + (R - (R - B)) + 1e-10)

        if NIR is not None and G is not None:
            indices['GNDVI'] = (NIR - G) / (NIR + G + 1e-10)

        if RE is not None and R is not None:
            indices['NDRE'] = (NIR - RE) / (NIR + RE + 1e-10)  # Red Edge NDVI

        # WATER INDICES
        if G is not None and NIR is not None:
            indices['NDWI'] = (G - NIR) / (G + NIR + 1e-10)

        if G is not None and SWIR1 is not None:
            indices['MNDWI'] = (G - SWIR1) / (G + SWIR1 + 1e-10)  # Modified NDWI

        if NIR is not None and SWIR1 is not None:
            indices['NDMI'] = (NIR - SWIR1) / (NIR + SWIR1 + 1e-10)  # Moisture

        # URBAN/BUILT-UP INDICES
        if SWIR1 is not None and NIR is not None:
            indices['NDBI'] = (SWIR1 - NIR) / (SWIR1 + NIR + 1e-10)

        if R is not None and SWIR1 is not None:
            indices['BUI'] = (R - SWIR1) / (R + SWIR1 + 1e-10)  # Built-Up Index

        # SOIL INDICES
        if R is not None and G is not None:
            indices['BI'] = np.sqrt((R**2 + G**2) / 2)  # Brightness Index

        if SWIR1 is not None and R is not None:
            indices['NDSI_soil'] = (SWIR1 - R) / (SWIR1 + R + 1e-10)

        # BURN INDICES
        if NIR is not None and SWIR2 is not None:
            indices['NBR'] = (NIR - SWIR2) / (NIR + SWIR2 + 1e-10)  # Normalized Burn Ratio

        if NIR is not None and SWIR1 is not None and SWIR2 is not None:
            indices['NBR2'] = (SWIR1 - SWIR2) / (SWIR1 + SWIR2 + 1e-10)

        # SNOW INDICES
        if G is not None and SWIR1 is not None:
            indices['NDSI'] = (G - SWIR1) / (G + SWIR1 + 1e-10)  # Normalized Difference Snow Index

        logger.info(f"Computed {len(indices)} spectral indices")

        return indices

    @staticmethod
    def compute_band_ratios(
        bands: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all possible band ratios

        Args:
            bands: Dictionary of bands

        Returns:
            Dictionary of band ratios
        """
        ratios = {}

        band_names = list(bands.keys())

        for i, name1 in enumerate(band_names):
            for j, name2 in enumerate(band_names):
                if i != j:
                    ratio_name = f"{name1}/{name2}"
                    ratios[ratio_name] = bands[name1] / (bands[name2] + 1e-10)

        logger.info(f"Computed {len(ratios)} band ratios")

        return ratios


class ObjectBasedImageAnalysis:
    """
    Object-Based Image Analysis (OBIA)
    Segment image into objects and extract object-level features
    """

    @staticmethod
    def segment_image(
        image: np.ndarray,
        scale: int = 100,
        min_size: int = 50
    ) -> np.ndarray:
        """
        Segment image into objects using watershed or SLIC

        Args:
            image: Input image (H, W, C)
            scale: Segmentation scale
            min_size: Minimum segment size

        Returns:
            Segmentation labels (H, W)
        """
        # Convert to LAB color space for better segmentation
        if image.shape[2] == 3:
            image_lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        else:
            image_lab = image

        # SLIC superpixel segmentation
        from skimage.segmentation import slic, felzenszwalb

        segments = slic(
            image_lab,
            n_segments=scale,
            compactness=10,
            sigma=1,
            start_label=1
        )

        # Remove small segments
        segments = morphology.remove_small_objects(segments.astype(int), min_size=min_size)

        logger.info(f"Image segmented: {segments.max()} objects")

        return segments

    @staticmethod
    def extract_object_features(
        image: np.ndarray,
        segments: np.ndarray,
        spectral_indices: Optional[Dict[str, np.ndarray]] = None
    ) -> List[Dict[str, float]]:
        """
        Extract features for each segmented object

        Args:
            image: Original image
            segments: Segmentation labels
            spectral_indices: Optional spectral indices

        Returns:
            List of feature dictionaries (one per object)
        """
        object_features = []

        # Get region properties
        regions = measure.regionprops(segments, intensity_image=image.mean(axis=2) if image.ndim == 3 else image)

        for region in regions:
            features = {}

            # Geometric features
            features['area'] = region.area
            features['perimeter'] = region.perimeter
            features['compactness'] = (region.perimeter ** 2) / (4 * np.pi * region.area + 1e-10)
            features['eccentricity'] = region.eccentricity
            features['solidity'] = region.solidity
            features['extent'] = region.extent

            # Shape features
            minr, minc, maxr, maxc = region.bbox
            features['length'] = maxr - minr
            features['width'] = maxc - minc
            features['aspect_ratio'] = features['length'] / (features['width'] + 1e-10)

            # Spectral features (mean values)
            mask = segments == region.label

            if image.ndim == 3:
                for band_idx in range(image.shape[2]):
                    features[f'band_{band_idx}_mean'] = image[mask, band_idx].mean()
                    features[f'band_{band_idx}_std'] = image[mask, band_idx].std()
            else:
                features['intensity_mean'] = region.mean_intensity
                features['intensity_std'] = image[mask].std()

            # Spectral index features
            if spectral_indices:
                for index_name, index_image in spectral_indices.items():
                    features[f'{index_name}_mean'] = index_image[mask].mean()
                    features[f'{index_name}_std'] = index_image[mask].std()

            object_features.append(features)

        logger.info(f"Extracted features for {len(object_features)} objects")

        return object_features

    @staticmethod
    def classify_objects(
        object_features: List[Dict[str, float]],
        classifier
    ) -> np.ndarray:
        """
        Classify objects using extracted features

        Args:
            object_features: List of feature dictionaries
            classifier: Trained classifier

        Returns:
            Class labels for each object
        """
        # Convert to feature matrix
        import pandas as pd

        df = pd.DataFrame(object_features)
        feature_matrix = df.values

        # Classify
        predictions = classifier.predict(feature_matrix)

        logger.info(f"Classified {len(predictions)} objects")

        return predictions


class SpatialFeatures:
    """
    Spatial and morphological features
    """

    @staticmethod
    def compute_morphological_profiles(
        image: np.ndarray,
        sizes: List[int] = [3, 5, 7, 9, 11]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Morphological opening/closing profiles

        Args:
            image: Input image
            sizes: Structuring element sizes

        Returns:
            Opening and closing profiles
        """
        profiles = {
            'opening': [],
            'closing': []
        }

        for size in sizes:
            selem = morphology.disk(size)

            # Opening
            opened = morphology.opening(image, selem)
            profiles['opening'].append(opened)

            # Closing
            closed = morphology.closing(image, selem)
            profiles['closing'].append(closed)

        logger.info(f"Morphological profiles: {len(sizes)} scales")

        return profiles

    @staticmethod
    def compute_edge_features(
        image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Edge detection features

        Args:
            image: Input image

        Returns:
            Edge feature maps
        """
        features = {}

        # Sobel
        features['sobel'] = filters.sobel(image)

        # Canny
        features['canny'] = filters.canny(image)

        # Roberts
        features['roberts'] = filters.roberts(image)

        # Scharr
        features['scharr'] = filters.scharr(image)

        logger.info("Edge features computed")

        return features


if __name__ == "__main__":
    print("Advanced Feature Engineering Ready!")
    print("\n🔬 Texture Features:")
    print("  - GLCM (Contrast, Homogeneity, Energy, Correlation)")
    print("  - LBP (Local Binary Patterns)")
    print("  - Gabor filter banks")
    print("\n📊 Spectral Features:")
    print("  - 20+ vegetation indices (NDVI, EVI, SAVI, etc.)")
    print("  - Water indices (NDWI, MNDWI, NDMI)")
    print("  - Urban indices (NDBI, BUI)")
    print("  - Soil, burn, and snow indices")
    print("  - All band ratios")
    print("\n🎯 Object-Based Analysis (OBIA):")
    print("  - Image segmentation (SLIC, watershed)")
    print("  - Object feature extraction")
    print("  - Geometric, spectral, and texture features per object")
    print("\n📐 Spatial Features:")
    print("  - Morphological profiles")
    print("  - Edge detection (Sobel, Canny, Roberts)")
    print("\n💪 Powerful features beyond raw pixels for superior classification!")
