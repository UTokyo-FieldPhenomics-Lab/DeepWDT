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


def runs_2_loc(run_id,
               run_duration,
               run_angle,
               hive_coordinates,
               video_start_time,
               first_frame_time):

    # Initialize the TimezoneFinder
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=hive_coordinates[0], lng=hive_coordinates[1])
    timezone = pytz.timezone(timezone_str)

    # Convert the start time of the video to the appropriate timezone
    video_start_time = timezone.localize(video_start_time)

    # Get solar
    solar_azimuth = get_azimuth(hive_coordinates[0], hive_coordinates[1], video_start_time + timedelta(seconds=first_frame_time))

    # Calculate distance from duration
    distance = distance_from_duration(run_duration)

    # Calculate the target coordinates
    origin = Point(hive_coordinates[0], hive_coordinates[1])
    target_point = geodesic(meters=distance).destination(origin, solar_azimuth+math.degrees(run_angle))

    point = {
        'id': run_id,
        'dance_angle': math.degrees(run_angle),
        'solar_azimuth': solar_azimuth,
        'dance_time': video_start_time + timedelta(seconds=first_frame_time),
        'latitude': target_point.latitude,
        'longitude': target_point.longitude,
        'target_distance': distance,
        'target_angle': solar_azimuth+math.degrees(run_angle),
        'geometry': ShapelyPoint(target_point.longitude, target_point.latitude)}

    return point


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

