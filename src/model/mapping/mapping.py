import json
import math
import os
from datetime import datetime, timedelta

import cv2
import geopandas as gpd
import pandas as pd
import pytz
import folium
import numpy as np
import geopandas as gpd
import folium
from sklearn.cluster import DBSCAN
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


from timezonefinder import TimezoneFinder
from pysolar.solar import get_azimuth
from geopy.distance import geodesic
from geopy.point import Point
from shapely.geometry import Point as ShapelyPoint


def distance_from_duration(duration):
    # if args['translation_method'] == 'schurch':
    #     a, b = 0.00138, 0.17
    # else:
    #     raise ValueError('Translation method is not valid.')
    a, b = 0.00138, 0.17
    distance = (duration - b) / a
    return distance


def runs_2_loc(path_tubes, path_geospatial, path_videos):
    # Load geospatial data
    with open(path_geospatial, 'r') as geo_file:
        geospatial_data = json.load(geo_file)

    # Initialize the TimezoneFinder
    tf = TimezoneFinder()

    # Collect data for GeoDataFrame
    points_data = []

    for file in os.listdir(path_tubes):
        if file.endswith('.csv'):
            video_name = file.replace('.csv', '')
            csv_path = os.path.join(path_tubes, file)
            data = pd.read_csv(csv_path)

            # Group by run_id and extract relevant information
            grouped = data.groupby('run_id')

            geo_entry = next((entry for entry in geospatial_data if entry['video'] == video_name), None)
            if geo_entry is None:
                continue

            latitude = geo_entry['latitude']
            longitude = geo_entry['longitude']

            # Get the timezone using latitude and longitude
            timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
            if timezone_str is None:
                print(f"Could not determine timezone for location: ({latitude}, {longitude})")
                continue
            timezone = pytz.timezone(timezone_str)

            # Convert the start time of the video to the appropriate timezone
            video_start_time = datetime.strptime(geo_entry['time'], "%Y-%m-%d_%H:%M")
            video_start_time = timezone.localize(video_start_time)
            print(video_start_time)

            # Extract the frame rate of the video
            video_path = os.path.join(path_videos, video_name + '.MOV')
            if not os.path.exists(video_path):
                print(f"Video file {video_path} not found.")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video file {video_path}.")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if video_fps == 0:
                print(f"Could not determine frame rate for video {video_path}.")
                continue

            # Add colony point to GeoDataFrame data
            points_data.append({
                'video_name': video_name,
                'run_id': 'colony',
                'latitude': latitude,
                'longitude': longitude,
                'angle': None,
                'distance': 0,
                'geometry': ShapelyPoint(longitude, latitude)
            })

            for run_id, group in grouped:
                first_row = group.iloc[0]
                frame_id = first_row['frame_id']
                angle = first_row['angle']
                duration = len(group)-14

                # Calculate actual group start time
                frame_offset = frame_id / video_fps
                group_start_time = video_start_time + timedelta(seconds=frame_offset)

                # Get solar azimuth (replace with real logic)
                solar_azimuth = get_azimuth(latitude, longitude, group_start_time)

                # Calculate distance from duration (replace with real logic)
                distance = distance_from_duration(duration / video_fps)

                # Calculate the target coordinates
                origin = Point(latitude, longitude)
                target_point = geodesic(meters=distance).destination(origin, solar_azimuth+math.degrees(angle))

                points_data.append({
                    'video_name': video_name,
                    'run_id': run_id,
                    'latitude': target_point.latitude,
                    'longitude': target_point.longitude,
                    'angle': solar_azimuth+math.degrees(angle),
                    'distance': distance,
                    'geometry': ShapelyPoint(target_point.longitude, target_point.latitude)
                })

    return points_data


def save_folium_map(points_data, output_html_map, eps=10, min_samples=2):
    # Create a GeoDataFrame from points_data
    gdf = gpd.GeoDataFrame(points_data, crs="EPSG:4326")

    # Filter out the hive point (colony)
    hive_point = gdf[gdf['run_id'] == 'colony'].iloc[0]
    hive_coords = (hive_point['latitude'], hive_point['longitude'])

    # Filter non-hive points for clustering
    non_hive_points = gdf[gdf['run_id'] != 'colony']

    # Prepare data for clustering: angle only
    clustering_data = non_hive_points[['angle']].to_numpy()

    # Apply DBSCAN clustering based on angle
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(clustering_data)
    non_hive_points['cluster'] = db.labels_

    # Generate distinct colors for each cluster using the Tab10 colormap
    cluster_ids = set(db.labels_)
    tab10 = plt.get_cmap('tab10')
    cluster_colors = {cluster_id: mcolors.to_hex(tab10(i % 10)) for i, cluster_id in enumerate(cluster_ids)}

    # Initialize the Folium map with Esri World Imagery tiles
    folium_map = folium.Map(location=hive_coords, zoom_start=15, tiles="Esri.WorldImagery")

    # Add the hive point as a marker
    folium.Marker(
        location=[hive_coords[0], hive_coords[1]],
        popup='Hive',
        icon=folium.Icon(color='red', icon='home')
    ).add_to(folium_map)

    # Process each cluster
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            # Skip noise points
            continue

        # Get points in the current cluster
        cluster_points = non_hive_points[non_hive_points['cluster'] == cluster_id]

        # Calculate the median angle and median distance
        median_angle = np.median(cluster_points['angle'])
        median_distance = np.median(cluster_points['distance'])

        # Compute the target point based on median distance and angle
        origin = (hive_coords[0], hive_coords[1])
        target_point = geodesic(meters=median_distance).destination(origin, median_angle)

        # Plot original points with transparency (non-filled)
        for _, row in cluster_points.iterrows():
            color = cluster_colors[cluster_id]
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=color,
                fill=False,
                opacity=0.5
            ).add_to(folium_map)

        # Plot cluster point at median distance (filled circle)
        folium.CircleMarker(
            location=[target_point.latitude, target_point.longitude],
            radius=10,
            color=cluster_colors[cluster_id],
            fill=True,
            fill_color=cluster_colors[cluster_id],
            fill_opacity=0.7
        ).add_to(folium_map)

    # Save the map to an HTML file
    folium_map.save(output_html_map)


def save_folium_map_old(points_data, output_html_map):
    # Create a GeoDataFrame from points_data
    gdf = gpd.GeoDataFrame(points_data, crs="EPSG:4326")

    # Initialize Folium map centered at the first colony point (assuming it's the first entry)
    colony_coords = (gdf.iloc[0]['latitude'], gdf.iloc[0]['longitude'])
    folium_map = folium.Map(location=colony_coords, zoom_start=15)

    # Add markers and lines to the Folium map
    for _, row in gdf.iterrows():
        if row['run_id'] == 'colony':
            # Add a marker for the colony
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup='Colony',
                icon=folium.Icon(color='red')
            ).add_to(folium_map)
        else:
            # Add a marker for the target point
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Run ID: {row['run_id']}\nAngle: {row['angle']}°\nDistance: {row['distance']:.2f} m",
                icon=folium.Icon(color='blue')
            ).add_to(folium_map)

            # Draw a line between the colony and the target point
            folium.PolyLine(
                locations=[colony_coords, [row['latitude'], row['longitude']]],
                color='blue'
            ).add_to(folium_map)

    # Save the map to an HTML file
    folium_map.save(output_html_map)
