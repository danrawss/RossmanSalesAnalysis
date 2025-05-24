# pages/02_Geospatial_Analysis.py

import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt

st.header("🗺️ Geospatial Analysis")

@st.cache_data
def load_world():
    # Load Natural Earth admin_0 countries shapefile (remote ZIP)
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    return world

world = load_world()

# Filter to Europe
europe = world[world.CONTINENT == "Europe"]

# 1) Plot full Europe
fig1, ax1 = plt.subplots(figsize=(8, 6))
europe.plot(ax=ax1, edgecolor="black", facecolor="#f0f0f0")
ax1.set_title("Map of Europe")
ax1.axis("off")
st.pyplot(fig1)

# 2) Highlight Germany within Europe
germany = europe[europe.ADMIN == "Germany"]
fig2, ax2 = plt.subplots(figsize=(8, 6))
europe.plot(ax=ax2, edgecolor="lightgray", facecolor="#fafafa")
germany.plot(ax=ax2, edgecolor="black", facecolor="#a6bddb")
ax2.set_title("Rossmann Store Country Highlighted (Germany)")
ax2.axis("off")
st.pyplot(fig2)

st.markdown(
    """
    - **Europe** is shown in light gray as context.  
    - **Germany** (in blue) is where the Rossmann dataset’s stores are located.  

    If you need true multi‐country store points, you’d want a dataset with latitude/longitude (and country) columns for each store.  
    Let me know if you’d like to pull in a different dataset (e.g. Global Superstore) or see how to geocode and plot real store locations!
    """
)
