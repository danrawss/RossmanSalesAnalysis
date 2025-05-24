import geopandas as gpd
from shapely.geometry import Point
import random

def load_europe_shapefile():
    """
    Load Natural Earth countries and filter to Europe.
    """
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    return world[world.CONTINENT == "Europe"].to_crs(epsg=4326)

def load_germany_shapefile():
    """
    Load Natural Earth and return just the Germany polygon in lat/lon.
    """
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    germany = world[world.ADMIN == "Germany"]
    return germany.to_crs(epsg=4326)

def simulate_store_geodata(store_ids, country="Germany"):
    """
    Given a list of store IDs, generate a random lat/lon inside `country` polygon for each.
    Returns a GeoDataFrame with columns ['Store', 'geometry'] (lat/lon CRS).
    """
    germany = load_germany_shapefile()
    poly = germany.geometry.iloc[0]
    minx, miny, maxx, maxy = poly.bounds

    points, ids = [], []
    for sid in store_ids:
        while True:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            p = Point(x, y)
            if poly.contains(p):
                points.append(p)
                ids.append(sid)
                break

    return gpd.GeoDataFrame({"Store": ids, "geometry": points}, crs="EPSG:4326")
