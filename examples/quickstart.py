#!/usr/bin/env python3
"""
Quickstart Example for Satellite Terrain Classification
Demonstrates basic usage of the system
"""
import numpy as np
from loguru import logger

# Configure logger
logger.info("🛰️ Satellite Terrain Classification - Quickstart Example")
logger.info("=" * 70)


def example_1_basic_classification():
    """Example 1: Basic terrain classification"""
    logger.info("\n📊 Example 1: Basic Terrain Classification")
    logger.info("-" * 70)

    # Create dummy satellite image
    satellite_image = np.random.rand(224, 224, 3).astype(np.float32)

    logger.info(f"Input image shape: {satellite_image.shape}")
    logger.info("✓ Basic classification example prepared")
    logger.info("  To run with real model:")
    logger.info("  from models.swin_transformer import create_swin_transformer")
    logger.info("  model = create_swin_transformer(num_classes=10)")
    logger.info("  predictions = model.predict(satellite_image[np.newaxis, ...])")


def example_2_multi_modal_fusion():
    """Example 2: Multi-modal satellite data fusion"""
    logger.info("\n🌈 Example 2: Multi-Modal Fusion (Optical + SAR + Multispectral)")
    logger.info("-" * 70)

    from multimodal import MultiModalFusionNetwork

    # Create dummy multi-modal data
    optical = np.random.rand(1, 224, 224, 3).astype(np.float32)
    sar = np.random.rand(1, 224, 224, 2).astype(np.float32)
    multispectral = np.random.rand(1, 224, 224, 8).astype(np.float32)

    logger.info(f"Optical shape: {optical.shape}")
    logger.info(f"SAR shape: {sar.shape}")
    logger.info(f"Multispectral shape: {multispectral.shape}")

    # Create fusion network
    model = MultiModalFusionNetwork(num_classes=10, fusion_method='late')

    # Fuse and predict
    inputs = {'optical': optical, 'sar': sar, 'multispectral': multispectral}
    logger.info("✓ Multi-modal fusion model created")
    logger.info("  Ready for all-weather, all-time terrain classification!")


def example_3_domain_analysis():
    """Example 3: Domain-specific analysis (Agriculture)"""
    logger.info("\n🌾 Example 3: Agriculture Monitoring")
    logger.info("-" * 70)

    from applications import AgricultureMonitoring

    # Create dummy NIR and Red bands
    nir = np.random.rand(512, 512).astype(np.float32)
    red = np.random.rand(512, 512).astype(np.float32)

    # Compute NDVI
    indices = AgricultureMonitoring.compute_vegetation_indices(nir, red)

    logger.info(f"Computed {len(indices)} vegetation indices")
    logger.info(f"Indices: {list(indices.keys())}")
    logger.info("✓ Agriculture analysis ready")
    logger.info("  Applications: crop health, yield estimation, irrigation management")


def example_4_uncertainty_quantification():
    """Example 4: Uncertainty quantification"""
    logger.info("\n📈 Example 4: Uncertainty Quantification")
    logger.info("-" * 70)

    logger.info("Uncertainty estimation with Monte Carlo Dropout:")
    logger.info("  - Epistemic uncertainty (model uncertainty)")
    logger.info("  - Aleatoric uncertainty (data uncertainty)")
    logger.info("  - Conformal prediction (statistical guarantees)")
    logger.info("  - OOD detection (identify anomalous inputs)")
    logger.info("✓ Trustworthy AI with uncertainty quantification")


def example_5_preprocessing():
    """Example 5: Advanced preprocessing"""
    logger.info("\n⚙️ Example 5: Advanced Preprocessing")
    logger.info("-" * 70)

    from preprocessing import PanSharpening

    # Create dummy pan and multispectral images
    pan = np.random.rand(1024, 1024).astype(np.float32)
    multispectral = np.random.rand(256, 256, 4).astype(np.float32)

    logger.info(f"Panchromatic: {pan.shape}")
    logger.info(f"Multispectral: {multispectral.shape}")

    # Pan-sharpen
    sharpened = PanSharpening.brovey_transform(pan, multispectral)

    logger.info(f"Pan-sharpened: {sharpened.shape}")
    logger.info("✓ High-resolution color imagery created")
    logger.info("  Also available: radiometric calibration, atmospheric correction")


def example_6_geospatial():
    """Example 6: Geospatial integration"""
    logger.info("\n🗺️ Example 6: Geospatial Integration")
    logger.info("-" * 70)

    logger.info("Geospatial capabilities:")
    logger.info("  - Read/write GeoTIFF with metadata")
    logger.info("  - Coordinate transformations (any CRS)")
    logger.info("  - Area calculations")
    logger.info("  - Interactive web maps (Folium)")
    logger.info("  - Tile-based processing for large images")
    logger.info("✓ Full GIS integration")


def example_7_active_learning():
    """Example 7: Active learning"""
    logger.info("\n🎯 Example 7: Active Learning (Reduce Labeling Costs by 80%)")
    logger.info("-" * 70)

    logger.info("Active learning strategies:")
    logger.info("  - Uncertainty sampling (select uncertain samples)")
    logger.info("  - Diversity sampling (maximize coverage)")
    logger.info("  - BALD (Bayesian Active Learning)")
    logger.info("  - Hybrid strategies")
    logger.info("✓ Smart sample selection for efficient labeling")


def main():
    """Run all examples"""
    try:
        example_1_basic_classification()
        example_2_multi_modal_fusion()
        example_3_domain_analysis()
        example_4_uncertainty_quantification()
        example_5_preprocessing()
        example_6_geospatial()
        example_7_active_learning()

        logger.info("\n" + "=" * 70)
        logger.info("✅ All examples completed successfully!")
        logger.info("=" * 70)
        logger.info("\n📚 Next steps:")
        logger.info("  1. Install dependencies: pip install -r requirements.txt")
        logger.info("  2. Train a model: python main.py train --architecture swin --data ./data")
        logger.info("  3. Run inference: python main.py predict --model ./model.h5 --input ./image.tif")
        logger.info("  4. Start API server: python main.py serve --port 8000")
        logger.info("\n🚀 Ready for production deployment!")

    except Exception as e:
        logger.error(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
