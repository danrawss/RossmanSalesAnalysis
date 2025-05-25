import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

from scripts.data_utils import load_data
from scripts.geo_utils import simulate_store_geodata

st.title("🗺️ Geospatial Analysis of Rossmann Stores")
st.markdown("""
**Purpose of this page:**  
This page demonstrates how to map and spatially analyze Rossmann store performance across Germany.  
- We simulate geographic coordinates for each store.  
- We compute average daily sales per store.  
- We cluster stores by sales volume.  
- We render an interactive map showing each store’s location, circle size (sales), and color (cluster).  
- We provide interpretation of what each sales‐based cluster represents.  
""")

# 1) Load, average & simulate geo 
@st.cache_data
def prep_store_data(_store_ids: tuple):
    """
    - load & average sales
    - simulate geodata for each store ID tuple
    Returns a DataFrame (not GeoDataFrame) with Store, avg_sales, geometry.
    """
    df  = load_data()
    avg = df.groupby("Store").Sales.mean().reset_index(name="avg_sales")
    # only keep those store IDs passed in, to guarantee consistent caching
    avg = avg[avg.Store.isin(_store_ids)]
    # simulate geo inside Germany
    gdf = simulate_store_geodata(_store_ids)
    # merge geometry + sales
    merged = gdf.merge(avg, on="Store")
    # flatten for Folium
    merged["lat"] = merged.geometry.y
    merged["lon"] = merged.geometry.x
    return merged

# Convert list to tuple so it's hashable
store_ids = tuple(load_data().Store.unique())
stores = prep_store_data(store_ids)

# 2) Cluster on avg_sales 
k = st.slider("Number of clusters", 2, 6, 4)
km = KMeans(n_clusters=k, random_state=42)
stores["cluster"] = km.fit_predict(stores[["avg_sales"]])
st.write("Cluster sizes:", stores.cluster.value_counts())

# 2.1) Explain what the clusters represent
st.subheader("📊 Cluster Interpretation")
# compute min/mean/max sales per cluster
cluster_stats = (
    stores.groupby("cluster")
          .avg_sales
          .agg(["min", "mean", "max"])
          .reset_index()
)
for _, row in cluster_stats.iterrows():
    st.markdown(
        f"**Cluster {int(row.cluster)}:**\n"
        f"- Range: {row['min']:.0f} → {row['max']:.0f} avg sales/day  \n"
        f"- Mean: {row['mean']:.0f} avg sales/day\n"
    )

# 3) Build Folium map 
# Define color palette (hex)
palette = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33"][:k]

m = folium.Map(location=[51.2, 10.4], zoom_start=6)

# Prepare a FeatureGroup for each cluster
cluster_groups = {
    i: folium.FeatureGroup(name=f"Cluster {i}", show=True)
    for i in range(k)
}

# precompute max for sizing
max_sales = stores.avg_sales.max()

for _, row in stores.iterrows():
    radius = max(3, (row.avg_sales / max_sales) * 15)
    marker = folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=radius,
        color=palette[int(row.cluster)],
        fill=True,
        fill_opacity=0.7,
        popup=f"Store {row.Store} | Avg: {row.avg_sales:.0f}"
    )
    cluster_groups[row.cluster].add_child(marker)

# 4) Add all cluster groups to the map
for fg in cluster_groups.values():
    m.add_child(fg)

# 5) Add the layer control (the “legend”)
folium.LayerControl(position='topleft', collapsed=False).add_to(m)

# 6) Render
st.subheader("Interactive Store Clusters Map")
st.caption("Use the checkboxes to toggle cluster visibility")
st_folium(m, width=800, height=600)

st.markdown("""
- **Circle size** ∝ average daily sales.  
- **Toggle clusters** on/off via the layer control checkboxes.
""")