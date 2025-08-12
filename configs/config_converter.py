import json

# Input GeoJSON-like data
geojson = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              -3.3077654892968837,
              50.41390350853965
            ],
            [
              1.486616937835123,
              50.41390350853965
            ],
            [
              1.486616937835123,
              52.757681393202944
            ],
            [
              -3.3077654892968837,
              52.757681393202944
            ],
            [
              -3.3077654892968837,
              50.41390350853965
            ]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}

# Extract coordinates from the polygon
polygon = geojson["features"][0]["geometry"]["coordinates"][0]

# Separate latitudes and longitudes
lons = [coord[0] for coord in polygon]
lats = [coord[1] for coord in polygon]

# Calculate bounding box
start_lat = min(lats)
end_lat = max(lats)
start_lon = min(lons)
end_lon = max(lons)

# Construct the output
output = {
    "start_lat": start_lat,
    "end_lat": end_lat,
    "start_lon": start_lon,
    "end_lon": end_lon,
    "scale": 10,
    "pixels": 512,
    "overlap_pixels": 0
}

# Print the output as JSON
print(json.dumps(output, indent=4))

with open("south_uk.json", "w") as f:
    json.dump(output, f, indent=4)

print("Saved to south_uk.json")
