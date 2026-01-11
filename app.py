"""
Streamlit application to visualize satellite acquisition plans over a given area of
interest (AOI) for multiple constellations and their satellites/sensors.  The
application lets the user select which constellations, satellites and sensors
to include, choose a date range, and then explore the corresponding swath
footprints on an interactive map.  A table summarises each swath and a
coverage statistic shows how much of the AOI is covered by the combined
selection.

This app depends on a number of geospatial libraries that may not be installed
in every environment by default.  To run this script locally you will need to
install the following packages (for example via pip):

  pip install streamlit geopandas shapely pydeck pandas

The sample data referenced in this application (e.g. the AOI KML file and
constellation shapefiles) should be placed alongside this script in a folder
structure such as:

  ├── streamlit_app.py
  ├── Qatar_eez.kml
  ├── CSK_1G/
  │     ├── some_file.shp
  │     ├── ...
  ├── SAOCOM/
        ├── SAOCOM_qatar.shp
        ├── SAOCOM_qatar.dbf
        ├── SAOCOM_qatar.shx
        └── SAOCOM_qatar.prj

The directory names for the constellations will be used as labels in the
sidebar.  Each directory can contain one or more shapefiles (.shp files) which
are concatenated into a single GeoDataFrame for that constellation.  If your
data uses different column names for satellites, sensors or dates the script
attempts to infer reasonable defaults by scanning for keywords such as
"sat", "sensor" and "date".

Note: this file is designed to be run with Streamlit, for example by executing
``streamlit run streamlit_app.py`` from a terminal.  Within the notebook
environment of this coding task the geospatial libraries may not be available,
so the app may not execute here, but the code serves as a complete working
example for use elsewhere.
"""

import os
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    import geopandas as gpd
    from shapely.ops import unary_union
    from shapely.geometry import mapping
except ImportError:
    gpd = None  # type: ignore
    unary_union = None  # type: ignore
    mapping = None  # type: ignore

try:
    import pydeck as pdk
except ImportError:
    pdk = None  # type: ignore


@st.cache_data(show_spinner=False)
def load_aoi(aoi_path: str) -> Optional[gpd.GeoDataFrame]:
    """Load the Area of Interest (AOI) from a KML or other vector file.

    Parameters
    ----------
    aoi_path: str
        Path to the AOI KML/shapefile/etc.

    Returns
    -------
    gpd.GeoDataFrame or None
        A GeoDataFrame containing the AOI geometry in WGS84 coordinates,
        or None if the geopandas library is unavailable.
    """
    if gpd is None:
        st.error(
            "Geopandas is required to read the AOI geometry. Please install it with `pip install geopandas`.")
        return None
    aoi = gpd.read_file(aoi_path, engine="pyogrio")
    # Ensure the coordinate reference system is WGS84 (lat/lon)
    if aoi.crs and aoi.crs.to_string() not in ("epsg:4326", "EPSG:4326"):
        aoi = aoi.to_crs("EPSG:4326")
    return aoi


@st.cache_data(show_spinner=False)
def load_constellations_data(base_dir: str) -> Dict[str, gpd.GeoDataFrame]:
    """Load all constellations from the given base directory.

    Each subdirectory within the base directory is treated as a separate
    constellation.  All .shp files inside a constellation directory are
    concatenated into a single GeoDataFrame.  The resulting dictionary maps
    constellation names to their GeoDataFrames.

    Parameters
    ----------
    base_dir: str
        Directory containing subdirectories for each constellation.

    Returns
    -------
    dict
        Mapping of constellation name to concatenated GeoDataFrame.
    """
    const_data: Dict[str, gpd.GeoDataFrame] = {}
    if gpd is None:
        return const_data
    for name in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, name)
        if not os.path.isdir(sub_path):
            continue
        # find shapefiles in this directory
        shp_files = [os.path.join(sub_path, f) for f in os.listdir(sub_path) if f.lower().endswith(".shp")]
        gdfs: List[gpd.GeoDataFrame] = []
        for shp_path in shp_files:
            try:
                gdf = gpd.read_file(shp_path, engine="pyogrio")
            except Exception as exc:
                st.warning(f"Failed to read {shp_path}: {exc}")
                continue
            # Standardise to WGS84
            try:
                if gdf.crs and gdf.crs.to_string() not in ("epsg:4326", "EPSG:4326"):
                    gdf = gdf.to_crs("EPSG:4326")
            except Exception:
                # if no CRS information is present, assume WGS84
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf["constellation"] = name
            # Normalise column names by stripping whitespace
            gdf.columns = [c.strip() for c in gdf.columns]
            gdfs.append(gdf)
        if gdfs:
            const_data[name] = pd.concat(gdfs, ignore_index=True)
    return const_data


def infer_column_name(columns: List[str], keywords: List[str]) -> Optional[str]:
    """Return the first column name that contains any of the given keywords.

    Parameters
    ----------
    columns: List[str]
        Available column names.
    keywords: List[str]
        Keywords to search for (case insensitive).

    Returns
    -------
    Optional[str]
        The first matching column name, or None if none match.
    """
    lower_cols = [c.lower() for c in columns]
    for key in keywords:
        for idx, col in enumerate(lower_cols):
            if key in col:
                # return the original column name preserving case
                return columns[idx]
    return None


def parse_date_column(df: gpd.GeoDataFrame, col: str) -> pd.Series:
    """Attempt to parse a date or datetime column into pandas Timestamps.

    Parameters
    ----------
    df: GeoDataFrame
        DataFrame containing the date column.
    col: str
        Column name to parse.

    Returns
    -------
    pd.Series
        A Series of datetime64 values.  Unparsable entries become NaT.
    """
    try:
        # Some datasets may store the date/time as string.  Try multiple formats.
        return pd.to_datetime(df[col], errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(df)), errors="coerce")


def build_map_layers(selected_gdf: gpd.GeoDataFrame, aoi: Optional[gpd.GeoDataFrame], show_aoi: bool) -> List[pdk.Layer]:
    """Create PyDeck layers for the map.

    Parameters
    ----------
    selected_gdf: GeoDataFrame
        GeoDataFrame containing the swaths to be displayed.
    aoi: GeoDataFrame or None
        AOI geometry.  If None, no AOI layer is added.
    show_aoi: bool
        Whether to include the AOI layer on the map.

    Returns
    -------
    list of pdk.Layer
        Layers for the PyDeck map.
    """
    layers: List[pdk.Layer] = []
    if pdk is None:
        return layers
    # Create a layer for the swaths
    if not selected_gdf.empty:
        # Convert GeoDataFrame to GeoJSON-like structures
        # Use mapping() to produce a dict representation of each geometry
        geojson_list = [
            {
                "type": "Feature",
                "properties": row.drop(labels="geometry").to_dict(),
                "geometry": mapping(row.geometry) if hasattr(row, "geometry") else None,
            }
            for _, row in selected_gdf.iterrows()
        ]
        swath_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson_list,
            get_fill_color="[200, 30, 0, 60]",  # semi-transparent red fill
            get_line_color="[200, 30, 0, 200]",
            pickable=True,
        )
        layers.append(swath_layer)
    # Add AOI layer
    if show_aoi and aoi is not None and len(aoi) > 0:
        aoi_geojson = [
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(geom) if hasattr(geom, "__geo_interface__") else mapping(geom)
            }
            for geom in aoi.geometry
        ]
        aoi_layer = pdk.Layer(
            "GeoJsonLayer",
            data=aoi_geojson,
            get_fill_color="[0, 0, 200, 40]",  # semi-transparent blue
            get_line_color="[0, 0, 200, 200]",
            pickable=False,
        )
        layers.append(aoi_layer)
    return layers


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(page_title="Acquisition Plans Viewer", layout="wide")
    # Display a linked logo at the top of the page.  The logo is read from the
    # local file system and encoded as base64 so that it can be embedded in
    # Markdown.  Clicking on the logo will navigate the user to the Arias Tech
    # Solutions website.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "ats_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as logo_file:
            logo_bytes = logo_file.read()
        logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")
        # Compose a single‑line HTML string for the linked logo.  The triple
        # quotes allow us to embed both the anchor and image tags without
        # introducing stray newlines into the string itself.
        logo_html = (
            f'<a href="https://www.ariastechsolutions.com/" target="_blank">'
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'alt="Arias Tech Solutions Logo" style="height:80px;" /></a>'
        )
        st.markdown(logo_html, unsafe_allow_html=True)
    # Page title displayed underneath the logo
    st.title("Satellite Acquisition Plans over Qatar EEZ")

    # Determine base directory relative to this script
    # It is safe to reuse script_dir computed above.
    base_dir = script_dir
    # Path to the AOI file (Qatar EEZ)
    aoi_path = os.path.join(base_dir, "Qatar_eez.kml")
    aoi = load_aoi(aoi_path)
    const_data = load_constellations_data(base_dir)

    # Define the date limits for the application
    default_start = datetime(2026, 2, 1)
    default_end = datetime(2026, 8, 31)

    # Sidebar for constellation selection
    st.sidebar.header("Constellations and Selection")
    selection: Dict[str, Tuple[List[str], List[str]]] = {}

    for const, gdf in const_data.items():
        with st.sidebar.expander(const, expanded=False):
            # Determine candidate columns for satellites and sensors.  Use a
            # broader set of keywords to increase the chance of finding the
            # correct column.
            columns = list(gdf.columns)
            sat_col = infer_column_name(columns, [
                "satellite", "satelliteid", "satellite_id", "sat_id", "sat",
                "platform", "spacecraft"
            ])
            sensor_col = infer_column_name(columns, [
                "sensor", "instrument", "payload", "sensor_id", "mode"
            ])
            # Unique values for satellites and sensors
            sats: List[str] = sorted(gdf[sat_col].dropna().unique().tolist()) if sat_col else []
            sensors: List[str] = sorted(gdf[sensor_col].dropna().unique().tolist()) if sensor_col else []
            # Provide a top‑level check box to quickly select everything for this constellation.
            select_all = st.checkbox(
                f"Select all {const}", value=False, key=f"{const}_all"
            )
            # Render a checkbox for each satellite.  If the top‑level select_all is
            # checked the default for each satellite will be True.
            selected_sats: List[str] = []
            if sats:
                st.markdown("**Satellites**")
                for i, sat in enumerate(sats):
                    # Use index as part of the key to avoid invalid characters in names
                    sat_key = f"{const}_sat_{i}"
                    checked = st.checkbox(
                        sat, value=select_all, key=sat_key
                    )
                    if checked:
                        selected_sats.append(sat)
            # Render a checkbox for each sensor.  Use the same select_all default.
            selected_sensors: List[str] = []
            if sensors:
                st.markdown("**Sensors**")
                for j, sensor in enumerate(sensors):
                    sens_key = f"{const}_sens_{j}"
                    checked = st.checkbox(
                        sensor, value=select_all, key=sens_key
                    )
                    if checked:
                        selected_sensors.append(sensor)
            selection[const] = (selected_sats, selected_sensors)

    # Date picker for acquisition dates
    st.sidebar.header("Date Range")
    start_date, end_date = st.sidebar.date_input(
        "Acquisition period",
        value=(default_start.date(), default_end.date()),
        min_value=default_start.date(),
        max_value=default_end.date()
    )
    # AOI visibility toggle
    show_aoi = st.sidebar.checkbox("Show AOI boundary", value=True)
    apply = st.sidebar.button("Apply filters")

    if apply:
        if gpd is None or unary_union is None or pdk is None:
            st.error(
                "One or more required libraries are missing. Please install geopandas, shapely and pydeck to run this app."
            )
            return
        # Gather selected rows across all constellations
        filtered_frames: List[gpd.GeoDataFrame] = []
        for const, (sat_list, sens_list) in selection.items():
            gdf = const_data.get(const)
            if gdf is None or gdf.empty:
                continue
            # If the user did not select any satellites or sensors for this constellation,
            # skip it entirely.  This prevents all data from the constellation being
            # included when no boxes are checked.
            if not sat_list and not sens_list:
                continue
            # Copy to avoid modifying the original
            df = gdf.copy()
            # Identify satellite and sensor columns using a broader set of keywords
            sat_col = infer_column_name(list(df.columns), [
                "satellite", "satelliteid", "satellite_id", "sat_id", "sat",
                "platform", "spacecraft"
            ])
            sensor_col = infer_column_name(list(df.columns), [
                "sensor", "instrument", "payload", "sensor_id", "mode"
            ])
            # Apply satellite filter if satellites were selected
            if sat_col and sat_list:
                df = df[df[sat_col].isin(sat_list)]
            # Apply sensor filter if sensors were selected
            if sensor_col and sens_list:
                df = df[df[sensor_col].isin(sens_list)]
            # If the filter removed all rows, skip further processing
            if df.empty:
                continue
            # Identify a date column by searching for keywords
            date_col = infer_column_name(list(df.columns), ["date", "start", "time", "acq"])
            if date_col:
                dates = parse_date_column(df, date_col)
                # Add parsed dates into the DataFrame to simplify filtering later
                df["__acq_datetime__"] = dates
                # Use midnight times for naive comparison (drop timezone information)
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                mask = (df["__acq_datetime__"] >= start_dt) & (df["__acq_datetime__"] <= end_dt)
                df = df.loc[mask]
            # Append to the list if there is any data left
            if not df.empty:
                filtered_frames.append(df)
        # Combine all selected frames
        if filtered_frames:
            result_gdf = gpd.GeoDataFrame(pd.concat(filtered_frames, ignore_index=True), crs="EPSG:4326")
        else:
            result_gdf = gpd.GeoDataFrame(columns=list(const_data[list(const_data.keys())[0]].columns))

        # Display map
        if not result_gdf.empty:
            # Compute combined coverage of selected swaths over the AOI
            coverage_message = ""
            if aoi is not None and not aoi.empty:
                try:
                    # Union of all selected swaths
                    swath_union = unary_union(result_gdf.geometry)
                    # Intersection with AOI (union of AOI geometries)
                    aoi_union = unary_union(aoi.geometry)
                    intersection = swath_union.intersection(aoi_union)
                    # Compute coverage ratio (area of intersection divided by AOI area)
                    # To compute area in square kilometres, project to a metric CRS such as EPSG:3857
                    result_area_gdf = gpd.GeoSeries([intersection], crs="EPSG:4326").to_crs("EPSG:3857")
                    aoi_area_gdf = gpd.GeoSeries([aoi_union], crs="EPSG:4326").to_crs("EPSG:3857")
                    covered_area = result_area_gdf.area.iloc[0] / 1e6  # convert m^2 to km^2
                    total_area = aoi_area_gdf.area.iloc[0] / 1e6
                    coverage_percent = (covered_area / total_area) * 100 if total_area > 0 else 0
                    coverage_message = f"Coverage of AOI: {coverage_percent:.1f}% (\ncovered {covered_area:.1f} km² out of {total_area:.1f} km²)"
                except Exception as exc:
                    coverage_message = f"Unable to compute coverage: {exc}"
            # Build map layers
            layers = build_map_layers(result_gdf, aoi, show_aoi)
            # Determine a suitable initial view state.  If there are swaths, centre on their centroid;
            # otherwise centre on the AOI.
            if not result_gdf.empty:
                bounds = result_gdf.total_bounds  # [minx, miny, maxx, maxy]
            elif aoi is not None and not aoi.empty:
                bounds = aoi.total_bounds
            else:
                # Default to a global view if nothing to show
                bounds = [-10, -10, 10, 10]
            minx, miny, maxx, maxy = bounds
            mid_lat = (miny + maxy) / 2
            mid_lon = (minx + maxx) / 2
            view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=6, bearing=0, pitch=0)
            deck = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{constellation}\nSatellite: {sat}\nSensor: {sensor}"})
            # Layout: map on top, table below
            st.subheader("Map of selected swaths")
            st.pydeck_chart(deck)
            st.subheader("Details of selected swaths")
            # Display selected attributes in a table; include only non-geometry columns
            display_df = result_gdf.drop(columns="geometry")
            st.dataframe(display_df)
            if coverage_message:
                st.subheader("Coverage statistic")
                st.write(coverage_message)
        else:
            st.warning("No swaths match the selected filters.")


if __name__ == "__main__":
    main()