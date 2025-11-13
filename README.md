# Aurora Forecast Coordinate Finder

This script finds the 4 points that make up the box around given decimal coordinates from the aurora forecast data file and calculates the interpolated aurora value using bilinear interpolation. It also retrieves cloud cover information to assess visibility conditions.

## Usage

Using Python directly:
```bash
python main.py <latitude> <longitude>
```

Using uv with script flag:
```bash
uv run -s main.py <latitude> <longitude>
```

Using uv with project script (recommended):
```bash
uv run aurora <latitude> <longitude>
```

### Example

```bash
uv run -s main.py 65 -147
```
or
```bash
uv run aurora 65 -147
```

This will output the 4 points forming a box around the given coordinates, with their aurora values, the interpolated value, and cloud cover information:

```
Data in ovation_aurora_latest.json is current (forecast is valid until 02:33 UTC).
Target coordinates: (0.5, -85.5)
Surrounding points:
  Lower left:     (0, -86) -> Aurora: 65
  Lower right:    (1, -86) -> Aurora: 64
  Upper left:     (0, -85) -> Aurora: 74
  Upper right:    (1, -85) -> Aurora: 73
Interpolated aurora value at target coordinates: 69.00
Cloud cover at target location: 25%
Visibility for aurora: 75% (clear of clouds)
```

## Features

- Automatically checks if the local data file is outdated based on the forecast time
- Downloads updated data from `https://services.swpc.noaa.gov/json/ovation_aurora_latest.json` when needed
- Calculates bilinear interpolation for accurate aurora value estimation at decimal coordinates
- Retrieves cloud cover percentage from OpenWeatherMap API to assess visibility
- Displays visibility percentage (100% - cloud cover %) for aurora observation
- Handles longitude conversion between -180/180 and 0/359 formats as needed
- Handles edge cases where coordinates are outside the data range

## How It Works

The script reads coordinates from `ovation_aurora_latest.json` and finds the 4 points that form a grid box enclosing the target coordinates. It then performs bilinear interpolation to estimate the aurora value at the exact decimal coordinates provided.

Bilinear interpolation is calculated using the formula:
f(x,y) ≈ f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy

Where the coordinates are normalized to the range [0,1] within the grid cell.

Cloud cover information is retrieved from OpenWeatherMap API using the original coordinates provided by the user.

## File Format

The script expects `ovation_aurora_latest.json` with the structure:
```json
{
  "Observation Time": "...",
  "Forecast Time": "...",
  "Data Format": "[Longitude, Latitude, Aurora]",
  "coordinates": [
    [lon, lat, aurora],
    ...
  ],
  "type": "MultiPoint"
}
```