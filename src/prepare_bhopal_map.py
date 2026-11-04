
import os
import osmnx as ox

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OSM_FILE = os.path.join(
    BASE_DIR, "data", "raw", "bhopal_roads.osm"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "bhopal_roads.geojson"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Bhopal OSM road network...")

# Read the downloaded OSM file
G = ox.graph_from_xml(
    OSM_FILE,
    simplify=True,
    retain_all=True
)

print("Road network loaded!")

# Convert road network into GeoDataFrame
nodes, edges = ox.graph_to_gdfs(
    G,
    nodes=True,
    edges=True
)

# Convert to latitude/longitude coordinates
edges = edges.to_crs(epsg=4326)

# Keep only useful road information
available_columns = [
    col for col in
    ["osmid", "name", "highway", "length", "geometry"]
    if col in edges.columns
]

edges = edges[available_columns]

# Save road network
edges.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)

print("\n========== BHOPAL MAP DATA ==========")
print("Total road segments:", len(edges))
print("Total road nodes:", len(nodes))
print("Road types:")
print(edges["highway"].value_counts().head(10))
print("\nGeoJSON saved at:")
print(OUTPUT_FILE)
print("\nBHOPAL MAP DATA READY!")
