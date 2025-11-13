#!/usr/bin/env python3
"""
Script to find the 4 points that make up the box around given decimal coordinates
from the summary.json file.
"""

import json
import sys
from typing import List, Tuple, Optional


def load_coordinates(filename: str) -> List[List[float]]:
    """
    Load coordinates from the summary.json file.
    
    Args:
        filename: Path to the JSON file containing coordinates
        
    Returns:
        List of [longitude, latitude, aurora] values
    """
    with open(filename, 'r') as file:
        data = json.load(file)
    
    return data.get('coordinates', [])


def find_surrounding_points(
    coordinates: List[List[float]], 
    target_lon: float, 
    target_lat: float
) -> Optional[Tuple[List, List, List, List]]:
    """
    Find the 4 points that form a box around the target coordinates.
    
    Args:
        coordinates: List of [longitude, latitude, aurora] values
        target_lon: Target longitude
        target_lat: Target latitude
        
    Returns:
        Tuple of 4 points (lower_left, lower_right, upper_left, upper_right) or None if not found
    """
    # Separate longitude and latitude values
    lons = sorted(set([point[0] for point in coordinates]))
    lats = sorted(set([point[1] for point in coordinates]))
    
    # Find longitude indices that surround the target longitude
    lon_idx = -1
    for i in range(len(lons)):
        if lons[i] <= target_lon:
            lon_idx = i
        else:
            break
    
    if lon_idx < 0 or lon_idx >= len(lons) - 1:
        # Target longitude is outside the range of available data
        return None
    
    # Find latitude indices that surround the target latitude
    lat_idx = -1
    for i in range(len(lats)):
        if lats[i] <= target_lat:
            lat_idx = i
        else:
            break
    
    if lat_idx < 0 or lat_idx >= len(lats) - 1:
        # Target latitude is outside the range of available data
        return None
    
    # Get the surrounding longitude and latitude values
    lon_left = lons[lon_idx]
    lon_right = lons[lon_idx + 1]
    lat_bottom = lats[lat_idx]
    lat_top = lats[lat_idx + 1]
    
    # Find the 4 corner points in the coordinates data
    lower_left = None
    lower_right = None
    upper_left = None
    upper_right = None
    
    for point in coordinates:
        lon, lat, aurora = point[0], point[1], point[2]
        
        if lon == lon_left and lat == lat_bottom:
            lower_left = point
        elif lon == lon_right and lat == lat_bottom:
            lower_right = point
        elif lon == lon_left and lat == lat_top:
            upper_left = point
        elif lon == lon_right and lat == lat_top:
            upper_right = point
    
    # Check if all 4 corners were found
    if not all([lower_left, lower_right, upper_left, upper_right]):
        return None
        
    return lower_left, lower_right, upper_left, upper_right


def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python find_surrounding_points.py <latitude> <longitude>")
        sys.exit(1)
    
    try:
        target_lat = float(sys.argv[1])
        target_lon = float(sys.argv[2])
    except ValueError:
        print("Error: Please provide valid decimal numbers for latitude and longitude")
        sys.exit(1)
    
    # Load coordinates from the JSON file
    coordinates = load_coordinates('ovation_aurora_latest.json')
    
    if not coordinates:
        print("Error: No coordinates found in summary.json")
        sys.exit(1)
    
    # Find the surrounding points
    result = find_surrounding_points(coordinates, target_lon, target_lat)
    
    if result is None:
        print(f"Could not find a box around coordinates ({target_lon}, {target_lat})")
        print("The coordinates may be outside the bounds of the available data.")
        sys.exit(1)
    
    lower_left, lower_right, upper_left, upper_right = result
    
    print(f"Target coordinates: ({target_lon}, {target_lat})")
    print("Surrounding points:")
    print(f"  Lower left:     ({lower_left[0]}, {lower_left[1]}) -> Aurora: {lower_left[2]}")
    print(f"  Lower right:    ({lower_right[0]}, {lower_right[1]}) -> Aurora: {lower_right[2]}")
    print(f"  Upper left:     ({upper_left[0]}, {upper_left[1]}) -> Aurora: {upper_left[2]}")
    print(f"  Upper right:    ({upper_right[0]}, {upper_right[1]}) -> Aurora: {upper_right[2]}")


if __name__ == "__main__":
    main()