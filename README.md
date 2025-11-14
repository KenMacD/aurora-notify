# Aurora Forecast and Alert System

This script provides an aurora forecast for a given geographic location. It fetches the latest aurora forecast data from NOAA, calculates the aurora intensity at the specified coordinates using bilinear interpolation, and assesses viewing conditions by checking cloud cover and whether it's nighttime using the OpenWeatherMap API.

Furthermore, it can send notifications via `ntfy` if the aurora intensity is above a certain threshold and viewing conditions are favorable.

## Features

- Automatically downloads the latest aurora forecast data from NOAA.
- Calculates an accurate aurora intensity value for any decimal coordinates using bilinear interpolation.
- Retrieves real-time cloud cover and sunrise/sunset times from the OpenWeatherMap API.
- Determines if it's dark enough for aurora viewing at the location.
- Sends a notification to a `ntfy` topic if aurora visibility is expected to be good.
- Configurable thresholds for aurora intensity and cloud cover via environment variables.

## Configuration

The script is configured via environment variables. You can create a `.env` file in the project root to store them.

- `OPENWEATHERMAP_API_KEY`: **(Required)** Your API key for the OpenWeatherMap API. You can get one for free from [their website](https://openweathermap.org/api).
- `NTFY_TOPIC`: The `ntfy` topic to send notifications to. Defaults to `aurora-alerts`.
- `NTFY_URL`: The URL of your `ntfy` server. Defaults to `https://ntfy.sh`.
- `MIN_AURORA_THRESHOLD`: The minimum aurora intensity value (0-100) to trigger a notification. Defaults to `50.0`.
- `MAX_CLOUD_COVER`: The maximum cloud cover percentage (0-100) considered good for viewing. Defaults to `30`.

### Example `.env` file
```
OPENWEATHERMAP_API_KEY="your_api_key_here"
NTFY_TOPIC="my-aurora-alerts"
MIN_AURORA_THRESHOLD="60"
MAX_CLOUD_COVER="25"
```

## Usage

Using Python directly:
```bash
python src/aurora/__main__.py <latitude> <longitude>
```

Using uv with project script (recommended):
```bash
uv run aurora <latitude> <longitude>
```

Alternatively, you can set the coordinates using environment variables:
```bash
export AURORA_LATITUDE=65
export AURORA_LONGITUDE=-147
uv run aurora
```

### Examples

With command line arguments:
```bash
uv run aurora 65 -147
```

With environment variables:
```bash
AURORA_LATITUDE=65 AURORA_LONGITUDE=-147 uv run aurora
```

This will output the forecast, viewing conditions, and send a notification if conditions are met:

```
Data in ovation_aurora_latest.json is current (forecast is valid until 22:45 UTC).
Target coordinates: (65, -147)
Surrounding points:
  Lower left:     (65, -147) -> Aurora: 80
  Lower right:    (65, -146) -> Aurora: 82
  Upper left:     (66, -147) -> Aurora: 85
  Upper right:    (66, -146) -> Aurora: 86
Interpolated aurora value at target coordinates: 80.00
Cloud cover at target location: 10%
Time conditions for aurora: Nighttime (dark enough)
Aurora visibility: Good (90% clear)
Notification sent successfully to aurora-alerts topic
```

## How It Works

The script reads aurora forecast data from `ovation_aurora_latest.json`. It finds the 4 grid points surrounding the target coordinates and uses bilinear interpolation to estimate the aurora value.

It then calls the OpenWeatherMap API to get current weather conditions for the location, specifically cloud cover percentage and sunrise/sunset times to determine if it's dark.

If the interpolated aurora value is above `MIN_AURORA_THRESHOLD`, cloud cover is below `MAX_CLOUD_COVER`, and it is nighttime, the script sends a notification to the configured `ntfy` topic.

Bilinear interpolation is calculated using the formula:
f(x,y) ≈ f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy

Where the coordinates are normalized to the range [0,1] within the grid cell.

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
