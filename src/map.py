import os
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point as ShapelyPoint
from sklearn.cluster import DBSCAN

from src.config import load_configuration
from src.model import runs_2_loc
from src.model.mapping.mapping import make_folium_map


def mapping_function(configuration_path):
    map_configuration = load_configuration(configuration_path).map
    print("Arguments: ", map_configuration)
    print('-----------------------------------------------------------------------------------------------------------')

    # Define the run_path
    run_name = f"{datetime.now().strftime('%y%m%d-%H%M%S')}"
    run_path = Path(f'runs/map/{run_name}')
    os.makedirs(run_path, exist_ok=True)
    print(f'Results saved at name: {run_path}')

    framerate = map_configuration.video_framerate

    # Add colony point to GeoDataFrame data
    points = []
    hive_coordinates = map_configuration.hive_coordinates
    points.append({
        'id': 'colony',
        'latitude': hive_coordinates[0],
        'longitude': hive_coordinates[1],
        'geometry': ShapelyPoint(hive_coordinates[1], hive_coordinates[0])
    })

    detection_path = map_configuration.detection_path
    duration_measurement_method = map_configuration.duration_measurement_method

    # Get all CSV files in the detection path
    csv_files = glob.glob(f"{detection_path}/*.csv")

    # Process each detection file
    for csv_file in csv_files:
        # Read the CSV file
        detections = pd.read_csv(csv_file)
        print(f"Processing file: {csv_file}")

        # Extract filename without extension and parse components
        file_name = Path(csv_file).stem
        file_parts = file_name.split("_")

        video_name, year, month, day, hour, minute, second = file_parts[:7]
        year, month, day = int(year), int(month), int(day)
        hour, minute, second = int(hour), int(minute), int(second)

        grouped = detections.groupby('run_id')

        for run_id, run_detections in grouped:

            # Compute dance duration
            if duration_measurement_method == 'range':
                run_duration = (run_detections['frame_id'].max() - run_detections['frame_id'].min()) - 7

            elif duration_measurement_method == 'count':
                run_duration = len(run_detections) - 7

            # Threshold dance duration
            if run_duration < map_configuration.duration_threshold:
                continue

            # Convert dance duration from frame to seconds
            run_duration = run_duration / framerate

            # Get run angle from the first row
            run_angle = run_detections['angle'].iloc[0]

            # Create datetime objects for video start and first frame
            video_start_time = datetime(year, month, day, hour, minute, second)
            first_frame_id = run_detections['frame_id'].iloc[0]
            first_frame_time = first_frame_id / framerate

            # Calculate target location
            target = runs_2_loc(
                run_id=run_id,
                run_duration=run_duration,
                run_angle=run_angle,
                hive_coordinates=hive_coordinates,
                video_start_time=video_start_time,
                first_frame_time=first_frame_time
            )

            target['video_name'] = video_name

            points.append(target)

    # Cluster with DBSCAN, epsilon = 100 m, min points = 10
    gdf = gpd.GeoDataFrame(points, geometry='geometry', crs="EPSG:4326")
    # gdf_proj = gdf.to_crs(epsg=3095)
    # coords = list(zip(gdf_proj.geometry.x, gdf_proj.geometry.y))
    # db = DBSCAN(eps=100, min_samples=10, metric='euclidean')
    # gdf_proj['cluster_id'] = db.fit_predict(coords)
    # gdf['cluster_id'] = gdf_proj['cluster_id']

    # save geopackage to run_path and create folium map
    make_folium_map(points, run_path)
