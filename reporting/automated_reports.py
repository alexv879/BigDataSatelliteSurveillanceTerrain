"""
Automated Report Generation for Satellite Terrain Analysis
Generate comprehensive reports with maps, statistics, and visualizations
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from loguru import logger
import json


class TerrainReport:
    """
    Generate comprehensive terrain analysis reports
    """

    def __init__(
        self,
        output_dir: str = "reports",
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize report generator

        Args:
            output_dir: Output directory
            class_names: List of terrain class names
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = class_names or [
            "Forest", "Barren", "Water", "Urban",
            "Agriculture", "Grass", "Snow", "Wetland",
            "Desert", "Shrubland"
        ]

        logger.info(f"Report generator initialized: {output_dir}")

    def generate_report(
        self,
        predictions: np.ndarray,
        metadata: Dict,
        statistics: Dict,
        maps: Dict[str, str],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Generate complete HTML report

        Args:
            predictions: Classification predictions
            metadata: Image metadata
            statistics: Analysis statistics
            maps: Dictionary of map file paths
            timestamp: Report timestamp

        Returns:
            Path to generated report
        """
        if timestamp is None:
            timestamp = datetime.now()

        report_id = timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"report_{report_id}.html"

        # Generate HTML
        html = self._generate_html(
            predictions,
            metadata,
            statistics,
            maps,
            timestamp
        )

        # Write report
        with open(report_path, 'w') as f:
            f.write(html)

        logger.info(f"Report generated: {report_path}")

        return str(report_path)

    def _generate_html(
        self,
        predictions: np.ndarray,
        metadata: Dict,
        statistics: Dict,
        maps: Dict[str, str],
        timestamp: datetime
    ) -> str:
        """Generate HTML content"""
        # Compute class distribution
        unique, counts = np.unique(predictions, return_counts=True)
        total_pixels = predictions.size

        class_percentages = []
        for cls, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            class_percentages.append((cls, count, percentage))

        # HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Satellite Terrain Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .section {{
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .map-container {{
            margin: 20px 0;
        }}
        .map-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .stat-box {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            min-width: 200px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .progress-bar {{
            background-color: #ecf0f1;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }}
        .progress-fill {{
            background-color: #3498db;
            height: 100%;
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ Satellite Terrain Analysis Report</h1>
        <p>Generated: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="section">
        <h2>📊 Overview</h2>
        <div class="stat-box">
            <div class="stat-label">Total Area</div>
            <div class="stat-value">{statistics.get('total_area_km2', 0):.2f} km²</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Image Resolution</div>
            <div class="stat-value">{metadata.get('resolution', 'N/A')}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Classes Detected</div>
            <div class="stat-value">{len(unique)}</div>
        </div>
    </div>

    <div class="section">
        <h2>🗺️ Classification Maps</h2>
        """

        # Add maps
        for map_name, map_path in maps.items():
            html += f"""
        <div class="map-container">
            <h3>{map_name}</h3>
            <img src="{map_path}" alt="{map_name}">
        </div>
            """

        html += """
    </div>

    <div class="section">
        <h2>📈 Land Cover Distribution</h2>
        <table>
            <tr>
                <th>Terrain Type</th>
                <th>Pixels</th>
                <th>Percentage</th>
                <th>Area (km²)</th>
                <th>Distribution</th>
            </tr>
        """

        # Add class rows
        for cls, count, percentage in sorted(class_percentages, key=lambda x: x[2], reverse=True):
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class {cls}"
            area_km2 = statistics.get('total_area_km2', 0) * (percentage / 100)

            html += f"""
            <tr>
                <td><strong>{class_name}</strong></td>
                <td>{count:,}</td>
                <td>{percentage:.2f}%</td>
                <td>{area_km2:.2f}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {percentage}%"></div>
                    </div>
                </td>
            </tr>
            """

        html += """
        </table>
    </div>

    <div class="section">
        <h2>📍 Metadata</h2>
        <table>
        """

        # Add metadata rows
        for key, value in metadata.items():
            html += f"""
            <tr>
                <td><strong>{key}</strong></td>
                <td>{value}</td>
            </tr>
            """

        html += """
        </table>
    </div>

    <div class="section">
        <h2>ℹ️ Analysis Details</h2>
        <p><strong>Model:</strong> {model}</p>
        <p><strong>Confidence:</strong> {confidence:.2f}%</p>
        <p><strong>Processing Time:</strong> {processing_time:.2f}s</p>
    </div>

    <div class="section" style="background-color: #ecf0f1; font-size: 12px;">
        <p>This report was automatically generated by the Satellite Terrain Classification System.</p>
        <p>Report ID: {report_id}</p>
    </div>
</body>
</html>
        """.format(
            model=statistics.get('model_name', 'Unknown'),
            confidence=statistics.get('avg_confidence', 0) * 100,
            processing_time=statistics.get('processing_time', 0),
            report_id=report_id
        )

        return html

    def generate_change_detection_report(
        self,
        t1_predictions: np.ndarray,
        t2_predictions: np.ndarray,
        change_map: np.ndarray,
        metadata: Dict,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Generate change detection report

        Args:
            t1_predictions: Time 1 predictions
            t2_predictions: Time 2 predictions
            change_map: Change detection map
            metadata: Metadata
            timestamp: Report timestamp

        Returns:
            Path to report
        """
        if timestamp is None:
            timestamp = datetime.now()

        report_id = timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"change_report_{report_id}.html"

        # Compute changes
        changed_pixels = (change_map == 1).sum()
        total_pixels = change_map.size
        change_percentage = (changed_pixels / total_pixels) * 100

        # Classify change types
        change_types = {}
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j:
                    mask = (t1_predictions == i) & (t2_predictions == j)
                    count = mask.sum()
                    if count > 0:
                        change_types[f"{self.class_names[i]} → {self.class_names[j]}"] = count

        # Sort by count
        sorted_changes = sorted(change_types.items(), key=lambda x: x[1], reverse=True)

        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Change Detection Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #e74c3c;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .section {{
            background-color: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .alert {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #e74c3c;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Change Detection Report</h1>
        <p>Generated: {timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="alert">
            <strong>⚠️ {change_percentage:.2f}% of area has changed</strong>
            <p>{changed_pixels:,} pixels out of {total_pixels:,} total pixels</p>
        </div>
    </div>

    <div class="section">
        <h2>Change Types</h2>
        <table>
            <tr>
                <th>Change Type</th>
                <th>Pixels</th>
                <th>Percentage</th>
            </tr>
        """

        for change_type, count in sorted_changes[:10]:  # Top 10
            percentage = (count / changed_pixels) * 100 if changed_pixels > 0 else 0
            html += f"""
            <tr>
                <td>{change_type}</td>
                <td>{count:,}</td>
                <td>{percentage:.2f}%</td>
            </tr>
            """

        html += f"""
        </table>
    </div>

    <div class="section">
        <h2>Metadata</h2>
        <p><strong>Time 1:</strong> {metadata.get('time1', 'N/A')}</p>
        <p><strong>Time 2:</strong> {metadata.get('time2', 'N/A')}</p>
        <p><strong>Location:</strong> {metadata.get('location', 'N/A')}</p>
    </div>
</body>
</html>
        """

        with open(report_path, 'w') as f:
            f.write(html)

        logger.info(f"Change detection report generated: {report_path}")

        return str(report_path)

    def export_statistics(
        self,
        predictions: np.ndarray,
        metadata: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export statistics as JSON

        Args:
            predictions: Predictions
            metadata: Metadata
            output_path: Output path (optional)

        Returns:
            Path to JSON file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"statistics_{timestamp}.json"

        # Compute statistics
        unique, counts = np.unique(predictions, return_counts=True)
        total_pixels = predictions.size

        class_stats = {}
        for cls, count in zip(unique, counts):
            class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
            class_stats[class_name] = {
                'pixels': int(count),
                'percentage': float(count / total_pixels * 100)
            }

        stats = {
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'total_pixels': int(total_pixels),
            'unique_classes': int(len(unique)),
            'class_distribution': class_stats
        }

        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics exported: {output_path}")

        return str(output_path)


if __name__ == "__main__":
    print("Automated Report Generation Ready!")
    print("\nFeatures:")
    print("- HTML reports with maps and statistics")
    print("- Change detection reports")
    print("- JSON statistics export")
    print("- Professional formatting")
    print("- Interactive visualizations")
    print("\n📊 Comprehensive terrain analysis reports!")
