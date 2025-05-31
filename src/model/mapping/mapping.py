import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytz
import geopandas as gpd
import folium
import matplotlib.colors as mcolors

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


def runs_2_loc(detections, video_start_time, hive_coordinates, video_framerate, duration_measurement_method):

    # Initialize the TimezoneFinder
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=hive_coordinates[0], lng=hive_coordinates[1])
    timezone = pytz.timezone(timezone_str)

    # Convert the start time of the video to the appropriate timezone
    video_start_time = timezone.localize(video_start_time)

    # Add colony point to GeoDataFrame data
    points = []
    points.append({
        'id': 'colony',
        'latitude': hive_coordinates[0],
        'longitude': hive_coordinates[1],
        'geometry': ShapelyPoint(hive_coordinates[1], hive_coordinates[0])
    })

    grouped_by_run = detections.groupby('run_id')

    for run_id, run_detections in grouped_by_run:
        first_row = run_detections.iloc[0]
        frame_id = first_row['frame_id']
        angle = first_row['angle']
        if duration_measurement_method == 'range':
            frame_duration = run_detections['frame_id'].max() - run_detections['frame_id'].min()
        elif duration_measurement_method == 'count':
            frame_duration = len(run_detections)

        # Calculate actual group start time
        frame_offset = frame_id / video_framerate
        group_start_time = video_start_time + timedelta(seconds=frame_offset)

        # Get solar
        solar_azimuth = get_azimuth(hive_coordinates[0], hive_coordinates[1], group_start_time)

        # Calculate distance from duration
        distance = distance_from_duration(frame_duration / video_framerate)

        # Calculate the target coordinates
        origin = Point(hive_coordinates[0], hive_coordinates[1])
        target_point = geodesic(meters=distance).destination(origin, solar_azimuth+math.degrees(angle))

        points.append({
            'id': run_id,
            'dance_angle': math.degrees(angle),
            'solar_azimuth': solar_azimuth,
            'dance_time': group_start_time,
            'latitude': target_point.latitude,
            'longitude': target_point.longitude,
            'target_distance': distance,
            'target_angle': solar_azimuth+math.degrees(angle),
            'geometry': ShapelyPoint(target_point.longitude, target_point.latitude)
        })

    return points


def make_folium_map(points, save_folder):
    # Create a GeoDataFrame from points
    gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")

    # Save the GeoDataFrame as a GeoPackage
    geopackage_path = Path(save_folder) / "runs.gpkg"
    gdf.to_file(geopackage_path, driver="GPKG")

    # Create a Folium map centered around the hive using Google Satellite imagery
    hive_location = (points[0]['latitude'], points[0]['longitude'])
    m = folium.Map(
        location=hive_location,
        tiles='http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        zoom_start=15
    )

    # Add the hive as a big yellow point with a black circle
    folium.CircleMarker(
        location=hive_location,
        radius=10,
        color='black',
        fill=True,
        fill_color='yellow',
        fill_opacity=1,
        popup='Hive'
    ).add_to(m)

    # Add points for run targets with random colors from the HSV colormap
    for point in points[1:]:
        # Generate a random hue value between 0 and 1
        h = random.random()
        # Use full saturation and value for a vivid color
        rgb = mcolors.hsv_to_rgb((h, 1.0, 1.0))
        hex_color = mcolors.rgb2hex(rgb)

        folium.CircleMarker(
            location=(point['latitude'], point['longitude']),
            radius=5,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.7,
            popup=f"Run ID: {point['id']}, Distance: {point['target_distance']:.2f}m"
        ).add_to(m)

    # Save the map as an HTML file
    map_path = Path(save_folder) / "runs_map.html"
    m.save(map_path)


def map_runs(detections, save_folder, video_name, framerate, duration_measurement_method):
    parts = str(video_name.stem).split('_')
    hive_y, hive_x = float(parts[1]), float(parts[2])
    date_str = "_".join(parts[3:6])
    time_str = "_".join(parts[6:])
    datetime_str = f"{date_str}_{time_str}"
    video_start_time = datetime.strptime(datetime_str, "%Y_%m_%d_%H_%M_%S")

    points = runs_2_loc(detections, video_start_time, (hive_y, hive_x), framerate, duration_measurement_method)

    make_folium_map(points, save_folder)
