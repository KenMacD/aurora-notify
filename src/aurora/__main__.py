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
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for aurora visibility conditions (can be adjusted)
NTFY_TOPIC = os.getenv(
    "NTFY_TOPIC", "aurora-alerts"
)  # Default to 'aurora-alerts' if not set
NTFY_URL = os.getenv(
    "NTFY_URL", "https://ntfy.sh"
)  # Default to public ntfy server if not set
MIN_AURORA_THRESHOLD = float(
    os.getenv("MIN_AURORA_THRESHOLD", "50.0")
)  # Minimum aurora value to trigger notification
MAX_CLOUD_COVER = int(
    os.getenv("MAX_CLOUD_COVER", "30")
)  # Maximum cloud cover percentage for good visibility


def check_and_update_data_file(filename: str, url: str) -> None:
    """
    Check if the data file is outdated and download a new version if needed.

    Args:
        filename: Path to the local data file
        url: URL to download the latest data from
    """
    try:
        # Get current time
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
            # Format is typically "2025-11-13T02:09:00Z", so handle the 'Z' suffix by replacing with '+00:00'
            forecast_time_str_tz = forecast_time_str.replace("Z", "+00:00")
            forecast_time = datetime.strptime(forecast_time_str_tz, parse_format)
        except ValueError:
            print(
                f"Could not parse Forecast Time '{forecast_time_str}' in {filename}. Downloading new data..."
            )
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename} successfully.")
            return

        # Add UTC timezone info to the parsed time if it doesn't have it
        if forecast_time.tzinfo is None:
            forecast_time = forecast_time.replace(tzinfo=timezone.utc)

        # Convert forecast time to local time and format without date
        forecast_local_time = forecast_time.astimezone()
        forecast_time_only = forecast_local_time.strftime("%H:%M %Z")
        if forecast_time_only.endswith(
            " "
        ):  # if timezone is not available, strftime might add empty space
            forecast_time_only = forecast_local_time.strftime("%H:%M")

        # Check if the forecast time is in the past compared to current time
        if forecast_time < current_time:
            print(
                f"Data in {filename} is outdated (forecast was only valid until {forecast_time_only}). Downloading new data..."
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


def is_nighttime(sunrise: int, sunset: int, current_time: int) -> bool:
    """
    Determine if it's currently nighttime based on sunrise and sunset times.

    Args:
        sunrise: Unix timestamp for sunrise (for the current day)
        sunset: Unix timestamp for sunset (for the current day)
        current_time: Current Unix timestamp

    Returns:
        True if it's nighttime (between sunset and sunrise), False otherwise
    """
    # The OpenWeatherMap API provides sunrise and sunset for the current day.
    # If current time is after sunset and before tomorrow's sunrise, it's nighttime.
    # If current time is before sunset and after today's sunrise, it's daytime.

    # Check if current time is after today's sunset
    if current_time >= sunset:
        # If so, it's nighttime until tomorrow's sunrise
        tomorrow_sunrise = sunrise + 86400  # Add 24 hours to get next day's sunrise
        if current_time < tomorrow_sunrise:
            return True
        # If current time is past tomorrow's sunrise, then sunrise/sunset values
        # might be for a different timezone context, so we'll default to the other checks

    # Check if current time is before today's sunrise (early morning, still night)
    if current_time < sunrise:
        # It's nighttime before today's sunrise
        return True

    # Otherwise, it's daytime (between sunrise and sunset)
    return False


def get_weather_data(lat: float, lon: float) -> Optional[dict]:
    """
    Get weather data from OpenWeatherMap API including cloud cover, sunrise, and sunset times.

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Dictionary containing cloud cover, sunrise, sunset, and current time, or None if request fails
    """
    # Get OpenWeatherMap API key from environment variable
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        print("Error: OPENWEATHERMAP_API_KEY environment variable not set")
        return None

    # Construct the API URL
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key}

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode())

        # Extract cloud cover percentage, sunrise, sunset, and current time
        weather_info = {
            "cloud_cover": data.get("clouds", {}).get("all"),
            "sunrise": data.get("sys", {}).get("sunrise"),
            "sunset": data.get("sys", {}).get("sunset"),
            "current_time": data.get("dt"),
        }
        return weather_info
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def is_good_aurora_visibility(
    interpolated_value: float, weather_data: Optional[dict]
) -> bool:
    """
    Determine if aurora visibility conditions are good based on interpolated value,
    cloud cover, and time of day.

    Args:
        interpolated_value: The interpolated aurora value at target coordinates
        weather_data: Dictionary containing cloud cover, sunrise, sunset, and current time

    Returns:
        True if conditions are good for aurora viewing, False otherwise
    """
    if weather_data is None:
        return False

    cloud_cover = weather_data.get("cloud_cover")
    sunrise = weather_data.get("sunrise")
    sunset = weather_data.get("sunset")
    current_time = weather_data.get("current_time")

    # Check if it's nighttime
    if sunrise is None or sunset is None or current_time is None:
        return False

    is_dark = is_nighttime(sunrise, sunset, current_time)

    # Check if aurora value is above threshold and it's dark enough
    if not is_dark:
        return False

    # Check if cloud cover is acceptable
    if cloud_cover is None or cloud_cover > MAX_CLOUD_COVER:
        return False

    # Check if aurora value is above minimum threshold
    if interpolated_value < MIN_AURORA_THRESHOLD:
        return False

    return True


def send_ntfy_notification(
    interpolated_value: float,
    weather_data: Optional[dict],
    target_lat: float,
    target_lon: float,
) -> None:
    """
    Send notification to ntfy if aurora visibility conditions are good.

    Args:
        interpolated_value: The interpolated aurora value at target coordinates
        weather_data: Dictionary containing cloud cover, sunrise, sunset, and current time
        target_lat: Target latitude
        target_lon: Target longitude
    """
    if not is_good_aurora_visibility(interpolated_value, weather_data):
        return

    # Prepare the notification message
    cloud_cover = weather_data.get("cloud_cover", "unknown")
    visibility_percentage = 100 - cloud_cover if cloud_cover is not None else "unknown"

    message = f"Aurora alert! High chance of aurora visibility at ({target_lat}, {target_lon})\n"
    message += f"Aurora value: {interpolated_value:.2f}\n"
    message += f"Cloud cover: {cloud_cover}%\n"
    message += f"Visibility: {visibility_percentage}% clear\n"
    message += f"Time: {datetime.fromtimestamp(weather_data.get('current_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}"

    try:
        # Send POST request to ntfy
        response = requests.post(
            f"{NTFY_URL}/{NTFY_TOPIC}",
            data=message.encode(encoding="utf-8"),
            headers={"Title": "Aurora Visibility Alert", "Priority": "default"},
        )

        if response.status_code == 200:
            print(f"Notification sent successfully to {NTFY_TOPIC} topic")
        else:
            print(f"Failed to send notification. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error sending notification: {e}")


def main():
    # Check command line arguments first, then environment variables
    if len(sys.argv) == 3:
        # Using command line arguments
        try:
            target_lat = float(sys.argv[1])
            target_lon = float(sys.argv[2])
        except ValueError:
            print(
                "Error: Please provide valid decimal numbers for latitude and longitude"
            )
            sys.exit(1)
    else:
        # Try environment variables
        env_lat = os.getenv("AURORA_LATITUDE")
        env_lon = os.getenv("AURORA_LONGITUDE")

        if env_lat is not None and env_lon is not None:
            try:
                target_lat = float(env_lat)
                target_lon = float(env_lon)
            except ValueError:
                print(
                    "Error: Environment variables AURORA_LATITUDE and AURORA_LONGITUDE must be valid decimal numbers"
                )
                sys.exit(1)
        else:
            print(f"Usage: {sys.argv[0]} <latitude> <longitude>")
            print(
                "   Or set environment variables AURORA_LATITUDE and AURORA_LONGITUDE"
            )
            sys.exit(1)

    # Validate longitude is in -180 to +180 range
    if target_lon < -180 or target_lon > 180:
        print("Error: Longitude must be between -180 and +180 degrees")
        sys.exit(1)

    # Validate latitude is in -90 to +90 range
    if target_lat < -90 or target_lat > 90:
        print("Error: Latitude must be between -90 and +90 degrees")
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
    original_lat = target_lat  # Use the latitude from either args or env
    original_lon = target_lon  # Use the longitude from either args or env

    # Get weather data using original longitude
    weather_data = get_weather_data(original_lat, original_lon)

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

    # Determine aurora visibility based on weather conditions
    if weather_data is not None:
        cloud_cover = weather_data.get("cloud_cover")
        sunrise = weather_data.get("sunrise")
        sunset = weather_data.get("sunset")
        current_time = weather_data.get("current_time")

        if cloud_cover is not None:
            print(f"Cloud cover at target location: {cloud_cover}%")
        else:
            print("Could not retrieve cloud cover data")

        if sunrise is not None and sunset is not None and current_time is not None:
            is_dark = is_nighttime(sunrise, sunset, current_time)
            print(
                f"Time conditions for aurora: {'Nighttime (dark enough)' if is_dark else 'Daytime (too bright)'}"
            )

            # Determine if aurora is likely visible
            if is_dark and cloud_cover is not None:
                visibility_percentage = 100 - cloud_cover
                if visibility_percentage > 50:
                    print(f"Aurora visibility: Good ({visibility_percentage}% clear)")
                elif 20 < visibility_percentage <= 50:
                    print(
                        f"Aurora visibility: Moderate ({visibility_percentage}% clear)"
                    )
                elif visibility_percentage <= 20:
                    print(f"Aurora visibility: Poor ({visibility_percentage}% clear)")
                else:
                    print("Aurora visibility: Uncertain due to cloud cover")
            elif not is_dark:
                print("Aurora visibility: Poor (not dark enough - daytime conditions)")
            else:
                print("Aurora visibility: Uncertain due to weather data")
        else:
            print("Could not retrieve sunrise/sunset times for nighttime check")
    else:
        print("Could not retrieve weather data")

    # Send ntfy notification if aurora visibility conditions are good
    send_ntfy_notification(interpolated_value, weather_data, original_lat, original_lon)


if __name__ == "__main__":
    main()
