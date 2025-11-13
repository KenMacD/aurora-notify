#!/usr/bin/env python3
"""
Script to find the 4 points that make up the box around given decimal coordinates
from the ovation_aurora_latest.json file.
"""

import json
import sys
from typing import List, Tuple, Optional
import urllib.request
from datetime import datetime, timezone
import os
import urllib.parse
import math


def check_and_update_data_file(filename: str, url: str) -> None:
    """
    Check if the data file is outdated and download a new version if needed.

    Args:
        filename: Path to the local data file
        url: URL to download the latest data from
    """
    try:
        # Get current time
        # Try to use the modern timezone-aware method first
        current_time = datetime.now(timezone.utc)
        parse_format = "%Y-%m-%dT%H:%M:%S%z"

        # Check if the file exists
        if not os.path.exists(filename):
            print(f"File {filename} not found. Downloading from {url}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully.")
            return

        # Load the existing file to get its forecast time
        with open(filename, "r") as file:
            data = json.load(file)

        forecast_time_str = data.get("Forecast Time", "")
        if not forecast_time_str:
            print(f"Forecast Time not found in {filename}. Downloading new data...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully.")
            return

        # Parse the forecast time from the file
        try:
            # Format is typically "2025-11-13T02:09:00Z"
            if parse_format.endswith("%z"):
                # For timezone-aware format, we need to handle the 'Z' suffix
                forecast_time_str_tz = forecast_time_str.replace("Z", "+00:00")
                forecast_time = datetime.strptime(forecast_time_str_tz, parse_format)
            else:
                forecast_time = datetime.strptime(forecast_time_str, parse_format)
        except ValueError:
            print(
                f"Could not parse Forecast Time '{forecast_time_str}' in {filename}. Downloading new data..."
            )
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully.")
            return

        # Convert forecast time to local time and format without date
        try:
            # Add UTC timezone info to the parsed time if it doesn't have it
            if forecast_time.tzinfo is None:
                forecast_time = forecast_time.replace(tzinfo=timezone.utc)
            # Convert to local time
            forecast_local_time = forecast_time.astimezone()
            forecast_time_only = forecast_local_time.strftime("%H:%M %Z")
            if forecast_time_only.endswith(
                " "
            ):  # if timezone is not available, strftime might add empty space
                forecast_time_only = forecast_local_time.strftime("%H:%M")
        except (ValueError, AttributeError):
            # Fallback to original format string if conversion fails
            time_part = (
                forecast_time_str.split("T")[1].split("Z")[0]
                if "T" in forecast_time_str
                else forecast_time_str
            )
            # Extract only hours and minutes
            forecast_time_only = (
                ":".join(time_part.split(":")[:2]) if ":" in time_part else time_part
            )

        # Check if the forecast time is in the past compared to current time
        if forecast_time < current_time:
            print(
                f"Data in {filename} is outdated (forecast was made at {forecast_time_only}). Downloading new data..."
            )
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully.")
        else:
            print(
                f"Data in {filename} is current (forecast is valid until {forecast_time_only})."
            )
    except Exception as e:
        print(f"Error checking or updating data file: {e}")
        print(f"Continuing with existing file {filename}...")


def load_coordinates(filename: str) -> List[List[float]]:
    """
    Load coordinates from the ovation_aurora_latest.json file.

    Args:
        filename: Path to the JSON file containing coordinates

    Returns:
        List of [longitude, latitude, aurora] values
    """
    with open(filename, "r") as file:
        data = json.load(file)

    return data.get("coordinates", [])


def find_surrounding_points_and_interpolate(
    coordinates: List[List[float]], target_lon: float, target_lat: float
) -> Optional[Tuple[List, List, List, List, float]]:
    """
    Find the 4 points that form a box around the target coordinates and calculate bilinear interpolation.
    Optimized to walk the list only once.

    Args:
        coordinates: List of [longitude, latitude, aurora] values
        target_lon: Target longitude
        target_lat: Target latitude

    Returns:
        Tuple of 4 points (lower_left, lower_right, upper_left, upper_right) and interpolated value,
        or None if not found
    """

    # For regular integer grid (0-359 longitude, -90 to 90 latitude):
    # Calculate bounding coordinates mathematically if target is between grid points

    # Longitude calculations (0-359 with wrap-around)
    # Normalize longitude to 0-359 range first
    norm_target_lon = ((target_lon % 360) + 360) % 360

    lon_left = int(math.floor(norm_target_lon)) % 360
    lon_right = (lon_left + 1) % 360

    lat_bottom = int(math.floor(target_lat))
    if lat_bottom == 90:
        lat_bottom = 89
        lat_top = 90
    else:
        lat_top = lat_bottom + 1

    # Now find the four corner points that match these coordinates
    lower_left = None
    lower_right = None
    upper_left = None
    upper_right = None

    # Single pass through coordinates to find the needed points
    for point in coordinates:
        lon, lat, _ = point[0], point[1], point[2]

        # Check if this point matches any of the four corner coordinates
        if lon == lon_left and lat == lat_bottom:
            lower_left = point
        elif lon == lon_right and lat == lat_bottom:
            lower_right = point
        elif lon == lon_left and lat == lat_top:
            upper_left = point
        elif lon == lon_right and lat == lat_top:
            upper_right = point

    # Check if all four corner points were found
    if not all([lower_left, lower_right, upper_left, upper_right]):
        return None

    # Calculate bilinear interpolation using the bounding coordinates
    lon_min = max(lower_left[0], upper_left[0])  # actual left longitude boundary
    lon_max = min(lower_right[0], upper_right[0])  # actual right longitude boundary
    lat_min = max(lower_left[1], lower_right[1])  # actual bottom latitude boundary
    lat_max = min(upper_left[1], upper_right[1])  # actual top latitude boundary

    # Check for valid bounding box (no division by zero)
    if lon_max <= lon_min or lat_max <= lat_min:
        return None

    # Calculate interpolation weights
    x = (target_lon - lon_min) / (lon_max - lon_min)  # longitude weight
    y = (target_lat - lat_min) / (lat_max - lat_min)  # latitude weight

    # Get aurora values of the corner points
    f00 = lower_left[2]  # bottom-left
    f10 = lower_right[2]  # bottom-right
    f01 = upper_left[2]  # top-left
    f11 = upper_right[2]  # top-right

    # Perform bilinear interpolation
    interpolated_value = (
        f00 * (1 - x) * (1 - y) + f10 * x * (1 - y) + f01 * (1 - x) * y + f11 * x * y
    )

    return lower_left, lower_right, upper_left, upper_right, interpolated_value


def get_cloud_cover(lat: float, lon: float) -> Optional[int]:
    """
    Get cloud cover percentage from OpenWeatherMap API.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Cloud cover percentage or None if request fails
    """
    # OpenWeatherMap API key from the example
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")

    # Construct the API URL
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key}

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode())

        # Extract cloud cover percentage
        cloud_cover = data.get("clouds", {}).get("all")
        return cloud_cover
    except Exception as e:
        print(f"Error fetching cloud cover data: {e}")
        return None


def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python find_surrounding_points.py <latitude> <longitude>")
        sys.exit(1)

    try:
        target_lat = float(sys.argv[1])
        target_lon = float(sys.argv[2])

        # Validate longitude is in -180 to +180 range
        if target_lon < -180 or target_lon > 180:
            print("Error: Longitude must be between -180 and +180 degrees")
            sys.exit(1)

        # Validate latitude is in -90 to +90 range
        if target_lat < -90 or target_lat > 90:
            print("Error: Latitude must be between -90 and +90 degrees")
            sys.exit(1)
    except ValueError:
        print("Error: Please provide valid decimal numbers for latitude and longitude")
        sys.exit(1)

    # Check if data file needs updating
    data_url = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
    check_and_update_data_file("ovation_aurora_latest.json", data_url)

    # Load coordinates from the JSON file
    coordinates = load_coordinates("ovation_aurora_latest.json")

    if not coordinates:
        print("Error: No coordinates found in ovation_aurora_latest.json")
        sys.exit(1)

    # Convert longitude to 0-359 range for aurora data lookup only
    aurora_lookup_lon = ((target_lon % 360) + 360) % 360

    # Find the surrounding points and calculate interpolation
    result = find_surrounding_points_and_interpolate(
        coordinates, aurora_lookup_lon, target_lat
    )

    if result is None:
        print(f"Could not find a box around coordinates ({target_lon}, {target_lat})")
        print("The coordinates may be outside the bounds of the available data.")
        sys.exit(1)

    lower_left, lower_right, upper_left, upper_right, interpolated_value = result

    # Store original coordinates for output
    original_lat = float(sys.argv[1])  # Get the original latitude from command line
    original_lon = float(sys.argv[2])  # Get the original longitude from command line

    # Get cloud cover data using original longitude
    cloud_cover = get_cloud_cover(original_lat, original_lon)

    print(f"Target coordinates: ({original_lat}, {original_lon})")
    print("Surrounding points:")
    # Convert longitude back to -180 to +180 range for display
    ll_lon_display = lower_left[0] if lower_left[0] <= 180 else lower_left[0] - 360
    lr_lon_display = lower_right[0] if lower_right[0] <= 180 else lower_right[0] - 360
    ul_lon_display = upper_left[0] if upper_left[0] <= 180 else upper_left[0] - 360
    ur_lon_display = upper_right[0] if upper_right[0] <= 180 else upper_right[0] - 360

    print(
        f"  Lower left:     ({lower_left[1]}, {ll_lon_display}) -> Aurora: {lower_left[2]}"
    )
    print(
        f"  Lower right:    ({lower_right[1]}, {lr_lon_display}) -> Aurora: {lower_right[2]}"
    )
    print(
        f"  Upper left:     ({upper_left[1]}, {ul_lon_display}) -> Aurora: {upper_left[2]}"
    )
    print(
        f"  Upper right:    ({upper_right[1]}, {ur_lon_display}) -> Aurora: {upper_right[2]}"
    )
    print(f"Interpolated aurora value at target coordinates: {interpolated_value:.2f}")

    if cloud_cover is not None:
        print(f"Cloud cover at target location: {cloud_cover}%")
        print(f"Visibility for aurora: {100 - cloud_cover}% (clear of clouds)")
    else:
        print("Could not retrieve cloud cover data")


if __name__ == "__main__":
    main()
