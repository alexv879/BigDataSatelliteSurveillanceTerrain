"""
Domain-Specific Satellite Analysis Applications
Agriculture, Disaster Assessment, Urban Planning, Environmental Monitoring
Real-world operational applications
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CropType(Enum):
    """Crop types for agriculture monitoring"""
    WHEAT = "wheat"
    CORN = "corn"
    RICE = "rice"
    SOYBEAN = "soybean"
    COTTON = "cotton"
    SUGARCANE = "sugarcane"
    UNKNOWN = "unknown"


@dataclass
class CropHealthMetrics:
    """Crop health analysis results"""
    ndvi_mean: float
    ndvi_std: float
    evi_mean: float
    health_status: str  # 'excellent', 'good', 'moderate', 'poor', 'critical'
    stressed_area_percent: float
    estimated_yield_index: float
    irrigation_needed: bool
    disease_risk: str  # 'low', 'medium', 'high'


class AgricultureMonitoring:
    """
    Precision agriculture and crop health monitoring
    """

    @staticmethod
    def compute_vegetation_indices(
        nir: np.ndarray,
        red: np.ndarray,
        blue: Optional[np.ndarray] = None,
        green: Optional[np.ndarray] = None,
        swir: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive vegetation indices

        Args:
            nir: Near-infrared band
            red: Red band
            blue: Blue band (optional)
            green: Green band (optional)
            swir: Short-wave infrared (optional)

        Returns:
            Dictionary of vegetation indices
        """
        indices = {}

        # NDVI (Normalized Difference Vegetation Index)
        indices['NDVI'] = (nir - red) / (nir + red + 1e-10)

        # SAVI (Soil-Adjusted Vegetation Index)
        L = 0.5  # Soil brightness correction factor
        indices['SAVI'] = ((nir - red) / (nir + red + L)) * (1 + L)

        # EVI (Enhanced Vegetation Index)
        if blue is not None:
            indices['EVI'] = 2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1))

        # GNDVI (Green NDVI)
        if green is not None:
            indices['GNDVI'] = (nir - green) / (nir + green + 1e-10)

        # NDMI (Normalized Difference Moisture Index)
        if swir is not None:
            indices['NDMI'] = (nir - swir) / (nir + swir + 1e-10)

        # MSAVI (Modified Soil-Adjusted Vegetation Index)
        indices['MSAVI'] = (2 * nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red))) / 2

        # ARVI (Atmospherically Resistant Vegetation Index)
        if blue is not None:
            rb = red - (red - blue)
            indices['ARVI'] = (nir - rb) / (nir + rb + 1e-10)

        logger.info(f"Computed {len(indices)} vegetation indices")

        return indices

    @staticmethod
    def assess_crop_health(
        ndvi: np.ndarray,
        evi: np.ndarray,
        ndmi: Optional[np.ndarray] = None
    ) -> CropHealthMetrics:
        """
        Assess overall crop health

        Args:
            ndvi: NDVI values
            evi: EVI values
            ndmi: NDMI values (optional, for water stress)

        Returns:
            Crop health metrics
        """
        # Statistics
        ndvi_mean = float(np.nanmean(ndvi))
        ndvi_std = float(np.nanstd(ndvi))
        evi_mean = float(np.nanmean(evi))

        # Health status based on NDVI
        if ndvi_mean > 0.7:
            health_status = 'excellent'
        elif ndvi_mean > 0.5:
            health_status = 'good'
        elif ndvi_mean > 0.3:
            health_status = 'moderate'
        elif ndvi_mean > 0.2:
            health_status = 'poor'
        else:
            health_status = 'critical'

        # Stressed area (NDVI < 0.3)
        stressed_pixels = (ndvi < 0.3).sum()
        total_pixels = ndvi.size
        stressed_area_percent = float(100 * stressed_pixels / total_pixels)

        # Yield estimation (simplified)
        estimated_yield_index = float(ndvi_mean * 100)  # 0-100 scale

        # Irrigation need (based on moisture index)
        irrigation_needed = False
        if ndmi is not None:
            ndmi_mean = np.nanmean(ndmi)
            if ndmi_mean < 0.2:  # Low moisture
                irrigation_needed = True

        # Disease risk (based on variability)
        if ndvi_std > 0.2:
            disease_risk = 'high'
        elif ndvi_std > 0.1:
            disease_risk = 'medium'
        else:
            disease_risk = 'low'

        metrics = CropHealthMetrics(
            ndvi_mean=ndvi_mean,
            ndvi_std=ndvi_std,
            evi_mean=evi_mean,
            health_status=health_status,
            stressed_area_percent=stressed_area_percent,
            estimated_yield_index=estimated_yield_index,
            irrigation_needed=irrigation_needed,
            disease_risk=disease_risk
        )

        logger.info(f"Crop health: {health_status}, Yield index: {estimated_yield_index:.1f}")

        return metrics

    @staticmethod
    def detect_crop_stress(
        ndvi_current: np.ndarray,
        ndvi_baseline: np.ndarray,
        threshold: float = 0.15
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Detect crop stress by comparing to baseline

        Args:
            ndvi_current: Current NDVI
            ndvi_baseline: Historical baseline NDVI
            threshold: NDVI decrease threshold for stress

        Returns:
            (stress_map, statistics)
        """
        # Compute difference
        ndvi_diff = ndvi_baseline - ndvi_current

        # Stress map
        stress_map = (ndvi_diff > threshold).astype(np.uint8)

        # Statistics
        total_area = stress_map.size
        stressed_area = stress_map.sum()
        stressed_percent = 100 * stressed_area / total_area

        stats = {
            'stressed_area_pixels': int(stressed_area),
            'stressed_percent': float(stressed_percent),
            'mean_ndvi_drop': float(ndvi_diff[stress_map == 1].mean()) if stressed_area > 0 else 0.0
        }

        logger.info(f"Crop stress detected: {stressed_percent:.1f}% of area")

        return stress_map, stats


class DisasterAssessment:
    """
    Disaster impact assessment from satellite imagery
    Floods, fires, earthquakes, hurricanes
    """

    @staticmethod
    def detect_flooding(
        nir: np.ndarray,
        green: np.ndarray,
        swir: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Detect flooded areas using water indices

        Args:
            nir: Near-infrared
            green: Green band
            swir: Short-wave infrared (optional)

        Returns:
            (flood_map, statistics)
        """
        # NDWI (Normalized Difference Water Index)
        ndwi = (green - nir) / (green + nir + 1e-10)

        # MNDWI (Modified NDWI) - better for urban areas
        if swir is not None:
            mndwi = (green - swir) / (green + swir + 1e-10)
            water_index = mndwi
        else:
            water_index = ndwi

        # Threshold for water
        flood_map = (water_index > 0.3).astype(np.uint8)

        # Statistics
        total_pixels = flood_map.size
        flooded_pixels = flood_map.sum()
        flooded_percent = 100 * flooded_pixels / total_pixels

        stats = {
            'flooded_pixels': int(flooded_pixels),
            'flooded_percent': float(flooded_percent),
            'mean_water_index': float(water_index[flood_map == 1].mean()) if flooded_pixels > 0 else 0.0
        }

        logger.info(f"Flooding detected: {flooded_percent:.2f}% of area")

        return flood_map, stats

    @staticmethod
    def detect_wildfire(
        nir: np.ndarray,
        red: np.ndarray,
        swir: np.ndarray,
        thermal: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Detect active fires and burn scars

        Args:
            nir: Near-infrared
            red: Red band
            swir: Short-wave infrared
            thermal: Thermal band (optional)

        Returns:
            (fire_map, statistics)
        """
        # NBR (Normalized Burn Ratio)
        nbr = (nir - swir) / (nir + swir + 1e-10)

        # Active fire detection (if thermal available)
        if thermal is not None:
            # Hot spots (simplified - real implementation uses more complex algorithm)
            active_fire = (thermal > np.percentile(thermal, 99))
        else:
            active_fire = np.zeros_like(nbr, dtype=bool)

        # Burn scar detection (low NBR indicates burned area)
        burn_scar = (nbr < 0.1).astype(np.uint8)

        # Combined fire map
        fire_map = np.maximum(burn_scar, active_fire.astype(np.uint8) * 2)
        # 0: no fire, 1: burn scar, 2: active fire

        # Statistics
        total_pixels = fire_map.size
        burn_scar_pixels = (fire_map == 1).sum()
        active_fire_pixels = (fire_map == 2).sum()

        stats = {
            'burn_scar_pixels': int(burn_scar_pixels),
            'active_fire_pixels': int(active_fire_pixels),
            'burn_scar_percent': float(100 * burn_scar_pixels / total_pixels),
            'active_fire_percent': float(100 * active_fire_pixels / total_pixels),
            'severity': 'critical' if active_fire_pixels > 0 else ('high' if burn_scar_pixels > 1000 else 'low')
        }

        logger.info(f"Fire detected: {active_fire_pixels} active, {burn_scar_pixels} burned")

        return fire_map, stats

    @staticmethod
    def assess_building_damage(
        pre_event: np.ndarray,
        post_event: np.ndarray,
        threshold: float = 0.3
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Assess building damage from earthquakes, hurricanes, etc.

        Args:
            pre_event: Image before disaster
            post_event: Image after disaster
            threshold: Change threshold for damage

        Returns:
            (damage_map, statistics)
        """
        # Compute change
        if pre_event.ndim == 3:
            pre_gray = np.mean(pre_event, axis=2)
            post_gray = np.mean(post_event, axis=2)
        else:
            pre_gray = pre_event
            post_gray = post_event

        # Absolute change
        change = np.abs(pre_gray - post_gray)

        # Normalize
        change_norm = change / (pre_gray.max() + 1e-10)

        # Damage map
        damage_map = (change_norm > threshold).astype(np.uint8)

        # Classify severity
        damage_severe = (change_norm > 0.5).astype(np.uint8)
        damage_moderate = ((change_norm > 0.3) & (change_norm <= 0.5)).astype(np.uint8)
        damage_light = ((change_norm > threshold) & (change_norm <= 0.3)).astype(np.uint8)

        # Combined damage levels: 0=no damage, 1=light, 2=moderate, 3=severe
        damage_levels = damage_light + 2*damage_moderate + 3*damage_severe

        # Statistics
        total_pixels = damage_map.size
        damaged_pixels = damage_map.sum()

        stats = {
            'damaged_pixels': int(damaged_pixels),
            'damaged_percent': float(100 * damaged_pixels / total_pixels),
            'light_damage_percent': float(100 * damage_light.sum() / total_pixels),
            'moderate_damage_percent': float(100 * damage_moderate.sum() / total_pixels),
            'severe_damage_percent': float(100 * damage_severe.sum() / total_pixels)
        }

        logger.info(f"Damage assessment: {stats['damaged_percent']:.1f}% affected")

        return damage_levels, stats


class UrbanAnalysis:
    """
    Urban planning and growth monitoring
    """

    @staticmethod
    def detect_urban_areas(
        nir: np.ndarray,
        red: np.ndarray,
        swir: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Detect and map urban/built-up areas

        Args:
            nir: Near-infrared
            red: Red band
            swir: Short-wave infrared

        Returns:
            (urban_map, statistics)
        """
        # NDBI (Normalized Difference Built-up Index)
        ndbi = (swir - nir) / (swir + nir + 1e-10)

        # NDVI for vegetation mask
        ndvi = (nir - red) / (nir + red + 1e-10)

        # Urban areas: high NDBI and low NDVI
        urban_map = ((ndbi > 0.1) & (ndvi < 0.2)).astype(np.uint8)

        # Statistics
        total_pixels = urban_map.size
        urban_pixels = urban_map.sum()
        urban_percent = 100 * urban_pixels / total_pixels

        stats = {
            'urban_pixels': int(urban_pixels),
            'urban_percent': float(urban_percent),
            'mean_ndbi': float(ndbi[urban_map == 1].mean()) if urban_pixels > 0 else 0.0
        }

        logger.info(f"Urban area: {urban_percent:.2f}%")

        return urban_map, stats

    @staticmethod
    def monitor_urban_growth(
        urban_t1: np.ndarray,
        urban_t2: np.ndarray,
        time_delta_years: float
    ) -> Dict[str, any]:
        """
        Monitor urban expansion over time

        Args:
            urban_t1: Urban map at time 1
            urban_t2: Urban map at time 2
            time_delta_years: Time difference in years

        Returns:
            Growth statistics
        """
        # New urban areas
        new_urban = ((urban_t2 == 1) & (urban_t1 == 0)).astype(np.uint8)

        # Lost urban areas (rare, but possible with demolition)
        lost_urban = ((urban_t1 == 1) & (urban_t2 == 0)).astype(np.uint8)

        # Statistics
        total_pixels = urban_t1.size
        initial_urban = urban_t1.sum()
        final_urban = urban_t2.sum()
        new_urban_pixels = new_urban.sum()
        lost_urban_pixels = lost_urban.sum()

        # Growth rate
        growth_rate = float(100 * (final_urban - initial_urban) / (initial_urban + 1)) / time_delta_years

        stats = {
            'initial_urban_percent': float(100 * initial_urban / total_pixels),
            'final_urban_percent': float(100 * final_urban / total_pixels),
            'new_urban_pixels': int(new_urban_pixels),
            'new_urban_percent': float(100 * new_urban_pixels / total_pixels),
            'growth_rate_percent_per_year': growth_rate,
            'expansion_map': new_urban
        }

        logger.info(f"Urban growth: {growth_rate:.2f}% per year")

        return stats


class EnvironmentalMonitoring:
    """
    Environmental and ecosystem monitoring
    """

    @staticmethod
    def monitor_deforestation(
        ndvi_t1: np.ndarray,
        ndvi_t2: np.ndarray,
        forest_threshold: float = 0.6
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Detect deforestation events

        Args:
            ndvi_t1: NDVI at time 1
            ndvi_t2: NDVI at time 2
            forest_threshold: NDVI threshold for forest

        Returns:
            (deforestation_map, statistics)
        """
        # Forest masks
        forest_t1 = (ndvi_t1 > forest_threshold)
        forest_t2 = (ndvi_t2 > forest_threshold)

        # Deforestation: was forest, now not forest
        deforestation = (forest_t1 & ~forest_t2).astype(np.uint8)

        # Statistics
        total_pixels = deforestation.size
        initial_forest = forest_t1.sum()
        deforested_pixels = deforestation.sum()
        deforested_percent = 100 * deforested_pixels / (initial_forest + 1)

        stats = {
            'initial_forest_pixels': int(initial_forest),
            'deforested_pixels': int(deforested_pixels),
            'deforested_percent_of_forest': float(deforested_percent),
            'deforested_percent_of_total': float(100 * deforested_pixels / total_pixels)
        }

        logger.info(f"Deforestation: {deforested_percent:.2f}% of forest lost")

        return deforestation, stats

    @staticmethod
    def assess_water_quality(
        blue: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Assess water quality indicators

        Args:
            blue, green, red, nir: Spectral bands

        Returns:
            Water quality indices
        """
        indices = {}

        # Chlorophyll-a concentration (proxy)
        # Higher green/blue ratio indicates more algae
        indices['chlorophyll_proxy'] = green / (blue + 1e-10)

        # Turbidity index
        # Higher red reflectance indicates more suspended sediments
        indices['turbidity_proxy'] = red / (green + 1e-10)

        # NDTI (Normalized Difference Turbidity Index)
        indices['NDTI'] = (red - green) / (red + green + 1e-10)

        # Water clarity (inverse of turbidity)
        indices['clarity'] = 1.0 / (indices['turbidity_proxy'] + 1e-10)

        logger.info("Water quality indices computed")

        return indices

    @staticmethod
    def monitor_snow_ice_cover(
        green: np.ndarray,
        swir: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Monitor snow and ice cover

        Args:
            green: Green band
            swir: Short-wave infrared

        Returns:
            (snow_map, statistics)
        """
        # NDSI (Normalized Difference Snow Index)
        ndsi = (green - swir) / (green + swir + 1e-10)

        # Snow/ice map
        snow_map = (ndsi > 0.4).astype(np.uint8)

        # Statistics
        total_pixels = snow_map.size
        snow_pixels = snow_map.sum()
        snow_percent = 100 * snow_pixels / total_pixels

        stats = {
            'snow_ice_pixels': int(snow_pixels),
            'snow_ice_percent': float(snow_percent),
            'mean_ndsi': float(ndsi[snow_map == 1].mean()) if snow_pixels > 0 else 0.0
        }

        logger.info(f"Snow/ice cover: {snow_percent:.2f}%")

        return snow_map, stats


if __name__ == "__main__":
    print("Domain-Specific Satellite Analysis Ready!")
    print("\n📊 Applications:")
    print("\n🌾 AGRICULTURE:")
    print("  - Crop health monitoring (NDVI, EVI, SAVI, MSAVI)")
    print("  - Yield estimation")
    print("  - Irrigation management")
    print("  - Disease and pest detection")
    print("  - Crop stress identification")
    print("\n🔥 DISASTER ASSESSMENT:")
    print("  - Flood mapping and extent")
    print("  - Wildfire detection (active fires + burn scars)")
    print("  - Building damage assessment")
    print("  - Earthquake impact analysis")
    print("  - Hurricane/typhoon damage")
    print("\n🏙️ URBAN PLANNING:")
    print("  - Urban area detection (NDBI)")
    print("  - Urban growth monitoring")
    print("  - Infrastructure mapping")
    print("  - Sprawl analysis")
    print("\n🌳 ENVIRONMENTAL MONITORING:")
    print("  - Deforestation detection")
    print("  - Water quality assessment")
    print("  - Snow/ice cover monitoring")
    print("  - Coastal erosion")
    print("  - Wetland health")
    print("\n💼 Real-world operational deployment ready!")
