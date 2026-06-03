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
import sqlite3
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

    Prefer a consolidated GeoPackage named ``qatar_calendar.gpkg`` when it is
    present. Each layer in that package is treated as a separate constellation.
    If the GeoPackage is missing, fall back to the legacy shapefile folder
    layout and build the same constellation mapping from those files.

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

    gpkg_path = os.path.join(base_dir, "qatar_calendar.gpkg")
    if os.path.exists(gpkg_path):
        try:
            with sqlite3.connect(gpkg_path) as conn:
                rows = conn.execute(
                    "SELECT table_name FROM gpkg_contents WHERE data_type = 'features' ORDER BY table_name"
                ).fetchall()
            layer_names = [row[0] for row in rows]
        except Exception as exc:
            st.warning(f"Failed to inspect {gpkg_path}: {exc}")
            layer_names = []

        for name in sorted(layer_names):
            try:
                gdf = gpd.read_file(gpkg_path, layer=name, engine="pyogrio")
            except Exception as exc:
                st.warning(f"Failed to read layer {name} from {gpkg_path}: {exc}")
                continue
            try:
                if gdf.crs and gdf.crs.to_string() not in ("epsg:4326", "EPSG:4326"):
                    gdf = gdf.to_crs("EPSG:4326")
            except Exception:
                gdf.set_crs("EPSG:4326", inplace=True)
            gdf["constellation"] = name
            gdf.columns = [c.strip() for c in gdf.columns]
            try:
                sat_orig = infer_column_name(gdf.columns.tolist(), [
                    "satellite", "satelliteid", "satellite_id", "sat_id", "sat",
                    "platform", "spacecraft"
                ])
                sensor_orig = infer_column_name(gdf.columns.tolist(), [
                    "sensor", "instrument", "payload", "sensor_id", "mode"
                ])
            except Exception:
                sat_orig = None
                sensor_orig = None
            if sat_orig and sat_orig not in gdf.columns:
                sat_orig = None
            if sensor_orig and sensor_orig not in gdf.columns:
                sensor_orig = None
            if sat_orig:
                gdf["sat"] = gdf[sat_orig]
            if sensor_orig:
                gdf["sensor"] = gdf[sensor_orig]
            const_data[name] = gdf
        if const_data:
            return const_data

    # Fallback: walk the project tree so nested delivery folders such as
    # qc_2/CSK_1G are loaded alongside the original constellation folders.
    shp_by_constellation: Dict[str, List[str]] = {}
    for root, _, files in os.walk(base_dir):
        rel_parts = os.path.relpath(root, base_dir).split(os.sep)
        if rel_parts and rel_parts[0].startswith("."):
            continue
        shp_files = [os.path.join(root, f) for f in files if f.lower().endswith(".shp")]
        if not shp_files:
            continue
        constellation_name = os.path.basename(root)
        if constellation_name == os.path.basename(base_dir):
            continue
        shp_by_constellation.setdefault(constellation_name, []).extend(shp_files)

    for name in sorted(shp_by_constellation):
        shp_files = sorted(shp_by_constellation[name])
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
            # Try to create unified satellite ('sat') and sensor ('sensor') columns to
            # simplify downstream processing.  Use the infer_column_name helper to
            # locate the original columns and copy their values into new fields.
            try:
                sat_orig = infer_column_name(gdf.columns.tolist(), [
                    "satellite", "satelliteid", "satellite_id", "sat_id", "sat",
                    "platform", "spacecraft"
                ])
                sensor_orig = infer_column_name(gdf.columns.tolist(), [
                    "sensor", "instrument", "payload", "sensor_id", "mode"
                ])
            except Exception:
                sat_orig = None
                sensor_orig = None
            if sat_orig and sat_orig not in gdf.columns:
                sat_orig = None
            if sensor_orig and sensor_orig not in gdf.columns:
                sensor_orig = None
            if sat_orig:
                gdf["sat"] = gdf[sat_orig]
            if sensor_orig:
                gdf["sensor"] = gdf[sensor_orig]
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


def infer_acquisition_date_column(columns: List[str]) -> Optional[str]:
    """Return the most likely acquisition date column.

    Many of the shapefiles contain fields such as ``MLST_Start`` that include
    the word "start" but do not store the acquisition date itself.  Prefer
    explicit date-like fields first so date filtering uses the correct column.
    """
    lower_map = {col.lower(): col for col in columns}
    preferred_exact = ["date", "acq_date", "acqdate"]
    for key in preferred_exact:
        if key in lower_map:
            return lower_map[key]

    preferred_contains = ["_date", " date", "date_", "acq"]
    for key in preferred_contains:
        for col in columns:
            if key in col.lower():
                return col

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


def get_data_date_span(df: gpd.GeoDataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Return the min/max acquisition dates found in a GeoDataFrame."""
    date_col = infer_acquisition_date_column(list(df.columns))
    if date_col is None:
        date_col = infer_column_name(list(df.columns), ["date", "acq", "start", "time"])
    if date_col is None:
        return None, None
    dates = parse_date_column(df, date_col)
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return None, None
    return valid_dates.min(), valid_dates.max()


def format_time_only(value: object) -> str:
    """Format a timestamp-like value as HH:MM:SS when possible."""
    if value is None or value == "":
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.strftime("%H:%M:%S")


def build_summary_table(result_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Build the compact editable summary table shown under the map."""
    table = pd.DataFrame(index=range(len(result_gdf)))
    table["Visible"] = True
    table["Satellite"] = result_gdf["sat"] if "sat" in result_gdf.columns else ""
    table["Sensor"] = result_gdf["sensor"] if "sensor" in result_gdf.columns else ""
    table["Constellation"] = (
        result_gdf["constellation"]
        if "constellation" in result_gdf.columns
        else (result_gdf["Constellat"] if "Constellat" in result_gdf.columns else "")
    )

    if "Date" in result_gdf.columns:
        table["Date"] = pd.to_datetime(result_gdf["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        table["Date"] = ""

    if "AreaCovere" in result_gdf.columns:
        table["Area covered"] = pd.to_numeric(result_gdf["AreaCovere"], errors="coerce")
    else:
        table["Area covered"] = pd.NA

    if "Start" in result_gdf.columns:
        table["Frame start"] = result_gdf["Start"].apply(format_time_only)
    else:
        table["Frame start"] = ""

    if "End" in result_gdf.columns:
        table["Frame end"] = result_gdf["End"].apply(format_time_only)
    else:
        table["Frame end"] = ""

    table["Orbit"] = result_gdf["Orbit"] if "Orbit" in result_gdf.columns else ""
    if "LookAngle" in result_gdf.columns:
        table["Look angle"] = pd.to_numeric(result_gdf["LookAngle"], errors="coerce")
    else:
        table["Look angle"] = pd.NA
    return table


def inject_brand_styles() -> None:
    """Apply a satellite-tasking visual theme using the logo palette."""
    st.markdown(
        """
        <style>
        :root {
            --qc-bg: #05070b;
            --qc-panel: #0b1118;
            --qc-panel-2: #101923;
            --qc-border: rgba(76, 209, 183, 0.22);
            --qc-cyan: #30d5ff;
            --qc-blue: #2c8cff;
            --qc-green: #62d36d;
            --qc-text: #eef6ff;
            --qc-muted: rgba(238, 246, 255, 0.72);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(48, 213, 255, 0.10), transparent 32%),
                radial-gradient(circle at top right, rgba(98, 211, 109, 0.10), transparent 30%),
                linear-gradient(180deg, #06080d 0%, #0a0f16 48%, #05070b 100%);
            color: var(--qc-text);
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 13, 20, 0.98), rgba(11, 18, 28, 0.98)),
                linear-gradient(135deg, rgba(48, 213, 255, 0.06), rgba(98, 211, 109, 0.04));
            border-right: 1px solid rgba(48, 213, 255, 0.18);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--qc-text) !important;
        }

        h1, h2, h3 {
            color: var(--qc-text);
            letter-spacing: 0.01em;
        }

        h1 {
            background: linear-gradient(90deg, var(--qc-cyan), var(--qc-blue), var(--qc-green));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-weight: 800;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .qc-hero {
            margin: 0 0 1rem 0;
            padding: 1rem 1.1rem;
            border: 1px solid var(--qc-border);
            border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(10, 16, 24, 0.88), rgba(16, 25, 35, 0.80)),
                radial-gradient(circle at left top, rgba(48, 213, 255, 0.10), transparent 40%);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.35);
        }

        .qc-hero .eyebrow {
            display: inline-block;
            margin-bottom: 0.4rem;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.76rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #06111a;
            background: linear-gradient(90deg, var(--qc-cyan), var(--qc-green));
            font-weight: 800;
        }

        .qc-hero .subtitle {
            margin-top: 0.35rem;
            color: var(--qc-muted);
            font-size: 0.98rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(48, 213, 255, 0.16);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(12, 18, 26, 0.96), rgba(9, 13, 20, 0.96));
        }

        div[data-testid="stExpander"] details summary {
            color: var(--qc-text);
            font-weight: 700;
        }

        button[kind="secondary"],
        button[kind="primary"] {
            border-radius: 999px !important;
            border: 1px solid rgba(48, 213, 255, 0.22) !important;
            background: linear-gradient(90deg, rgba(48, 213, 255, 0.12), rgba(98, 211, 109, 0.10)) !important;
            color: var(--qc-text) !important;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
        }

        button[kind="primary"] {
            background: linear-gradient(90deg, var(--qc-cyan), var(--qc-blue)) !important;
            color: #051018 !important;
            font-weight: 800 !important;
        }

        button[kind="primary"]:disabled,
        button[kind="secondary"]:disabled {
            opacity: 0.45 !important;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stDataEditor"] {
            border: 1px solid rgba(48, 213, 255, 0.16);
            border-radius: 16px;
            overflow: hidden;
            background: rgba(8, 12, 18, 0.94);
        }

        div[data-testid="stDataFrame"] table,
        div[data-testid="stDataEditor"] table {
            font-size: 0.92rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        .stCaption, caption {
            color: rgba(238, 246, 255, 0.70) !important;
        }

        hr {
            border-color: rgba(48, 213, 255, 0.12);
        }

        footer, #MainMenu {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.fragment
def render_results_panel(result_gdf: gpd.GeoDataFrame, aoi: Optional[gpd.GeoDataFrame], show_aoi: bool) -> None:
    """Render the interactive results area without rerunning the whole page."""
    visibility_state = st.session_state.get("frame_visibility_state")
    if not isinstance(visibility_state, list) or len(visibility_state) != len(result_gdf):
        visibility_state = [True] * len(result_gdf)
        st.session_state["frame_visibility_state"] = visibility_state

    summary_df = build_summary_table(result_gdf).reset_index(drop=True)
    summary_df["Visible"] = visibility_state
    selected_rows: List[int] = []
    selection_event = st.dataframe(
        summary_df.drop(columns="Visible"),
        key="frame_selection_table_v2",
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row",
        height=260,
    )
    try:
        selected_rows = list(selection_event.selection.rows)  # type: ignore[assignment]
    except Exception:
        selected_rows = []

    control_cols = st.columns(4)
    with control_cols[0]:
        if st.button("Hide selected", key="hide_selected_frames", disabled=not selected_rows):
            for idx in selected_rows:
                if 0 <= idx < len(visibility_state):
                    visibility_state[idx] = False
            st.session_state["frame_visibility_state"] = visibility_state
            st.rerun()
    with control_cols[1]:
        if st.button("Show selected", key="show_selected_frames", disabled=not selected_rows):
            for idx in selected_rows:
                if 0 <= idx < len(visibility_state):
                    visibility_state[idx] = True
            st.session_state["frame_visibility_state"] = visibility_state
            st.rerun()
    with control_cols[2]:
        if st.button("Hide all", key="hide_all_frames"):
            st.session_state["frame_visibility_state"] = [False] * len(result_gdf)
            st.rerun()
    with control_cols[3]:
        if st.button("Show all", key="show_all_frames"):
            st.session_state["frame_visibility_state"] = [True] * len(result_gdf)
            st.rerun()

    visible_mask = pd.Series(st.session_state["frame_visibility_state"], index=result_gdf.index)
    visible_result_gdf = result_gdf.loc[visible_mask.to_numpy()].copy()

    st.subheader("Map of selected swaths")
    if visible_result_gdf.empty:
        st.warning("All frames are hidden. Re-enable a row to show it on the map.")
    else:
        frames_gdf = visible_result_gdf.copy()
        area_col: Optional[str] = None
        for col in frames_gdf.columns:
            lc = col.lower()
            if "area" in lc and "cover" in lc:
                area_col = col
                break
        if area_col:
            area_series = pd.to_numeric(frames_gdf[area_col], errors="coerce")
            frames_gdf = frames_gdf[area_series.notna() & (area_series > 0)].copy()

        coverage_message = ""
        if aoi is not None and not aoi.empty and not frames_gdf.empty:
            try:
                swath_union = unary_union(frames_gdf.geometry)
                aoi_union = unary_union(aoi.geometry)
                intersection = swath_union.intersection(aoi_union)
                result_area_gdf = gpd.GeoSeries([intersection], crs="EPSG:4326").to_crs("EPSG:3857")
                aoi_area_gdf = gpd.GeoSeries([aoi_union], crs="EPSG:4326").to_crs("EPSG:3857")
                covered_area = result_area_gdf.area.iloc[0] / 1e6
                total_area = aoi_area_gdf.area.iloc[0] / 1e6
                coverage_percent = (covered_area / total_area) * 100 if total_area > 0 else 0
                coverage_message = f"Coverage of AOI: {coverage_percent:.1f}% (covered {covered_area:.1f} km?? out of {total_area:.1f} km??)"
            except Exception as exc:
                coverage_message = f"Unable to compute coverage: {exc}"

        layers = build_map_layers(frames_gdf, aoi, show_aoi)
        if not frames_gdf.empty:
            bounds = frames_gdf.total_bounds
        elif aoi is not None and not aoi.empty:
            bounds = aoi.total_bounds
        else:
            bounds = [-10, -10, 10, 10]
        minx, miny, maxx, maxy = bounds
        mid_lat = (miny + maxy) / 2
        mid_lon = (minx + maxx) / 2
        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=6, bearing=0, pitch=0)
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{constellation}\\nSatellite: {sat}\\nSensor: {sensor}"},
        )
        st.pydeck_chart(deck)

        if coverage_message:
            st.subheader("Coverage statistic")
            st.write(coverage_message)

    st.subheader("Details of selected swaths")
    st.caption("Select one or more rows, then use the buttons above to hide or show them.")
    if st.checkbox("Show full table", value=False, key="show_full_table_v2"):
        with st.expander("Full table", expanded=True):
            st.dataframe(result_gdf.drop(columns="geometry"), use_container_width=True, hide_index=True)


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
        # Convert GeoDataFrame to a list of GeoJSON features.  Each geometry
        # is flattened so that MultiPolygons are broken into individual polygon
        # features.  The swath properties are attached to each frame so that
        # the table remains swath‑centric while the map displays individual
        # footprints.
        geojson_list: List[Dict[str, object]] = []
        for _, row in selected_gdf.iterrows():
            props = row.drop(labels="geometry").to_dict()
            geom = row.geometry
            if geom is None:
                continue
            try:
                geom_type = geom.geom_type
            except Exception:
                geom_type = None
            # Flatten MultiPolygons into separate polygons
            if geom_type == "MultiPolygon":
                # type: ignore[attr-defined] — geoms attribute is present for MultiPolygon
                for poly in geom.geoms:  # type: ignore[attr-defined]
                    geojson_list.append({
                        "type": "Feature",
                        "properties": props,
                        "geometry": mapping(poly),
                    })
            else:
                geojson_list.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": mapping(geom),
                })
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


def toggle_all_constellation(const: str, keys: List[str]) -> None:
    """Callback to toggle all satellite and sensor checkboxes for a constellation.

    When the top‑level 'Select all' checkbox is toggled, this function updates
    the Streamlit session state for each underlying checkbox key to match the
    new state.

    Parameters
    ----------
    const: str
        The name of the constellation; used to find the state of the select all
        checkbox in st.session_state.
    keys: List[str]
        A list of checkbox keys for satellites and sensors.
    """
    # Retrieve the current value of the select all checkbox
    select_state = st.session_state.get(f"{const}_all", False)
    for key in keys:
        st.session_state[key] = select_state


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
    const_spans: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {
        const: get_data_date_span(gdf) for const, gdf in const_data.items()
    }

    # Define the date limits for the application. The dataset now spans the
    # full year from 2026-02-01 through 2027-01-31.
    default_start = datetime(2026, 2, 1)
    default_end = datetime(2027, 1, 31)

    # Sidebar for constellation selection
    st.sidebar.header("Constellations and Selection")
    selection: Dict[str, Tuple[List[str], List[str]]] = {}

    for const, gdf in const_data.items():
        with st.sidebar.expander(const, expanded=False):
            span_start, span_end = const_spans.get(const, (None, None))
            if span_start is not None and span_end is not None:
                st.caption(f"Available dates: {span_start.date()} to {span_end.date()}")
            else:
                st.caption("Available dates: unavailable")
            # Pull the unified 'sat' and 'sensor' columns created when loading the
            # constellation data.  Fallback to empty lists if columns are not present.
            sat_values: List[str] = sorted(gdf["sat"].dropna().unique().tolist()) if "sat" in gdf.columns else []
            sensor_values: List[str] = sorted(gdf["sensor"].dropna().unique().tolist()) if "sensor" in gdf.columns else []
            # Precompute keys for satellites and sensors so we can pass them to the callback
            sat_keys = [f"{const}_sat_{i}" for i in range(len(sat_values))]
            sensor_keys = [f"{const}_sens_{j}" for j in range(len(sensor_values))]
            # Top‑level checkbox to select/deselect all satellites and sensors.  Use
            # on_change to update the state of all subordinate checkboxes via the
            # toggle_all_constellation callback.
            st.checkbox(
                f"Select all {const}", key=f"{const}_all",
                on_change=toggle_all_constellation, args=(const, sat_keys + sensor_keys)
            )
            selected_sats: List[str] = []
            if sat_values:
                st.markdown("**Satellites**")
                for i, sat in enumerate(sat_values):
                    sat_key = sat_keys[i]
                    # If the checkbox state hasn't been initialised in session_state,
                    # default to False.  The toggle_all_constellation callback will
                    # overwrite these values when the user interacts with the select all.
                    if sat_key not in st.session_state:
                        st.session_state[sat_key] = False
                    checked = st.checkbox(sat, key=sat_key)
                    if checked:
                        selected_sats.append(sat)
            selected_sensors: List[str] = []
            if sensor_values:
                st.markdown("**Sensors**")
                for j, sens in enumerate(sensor_values):
                    sens_key = sensor_keys[j]
                    if sens_key not in st.session_state:
                        st.session_state[sens_key] = False
                    checked = st.checkbox(sens, key=sens_key)
                    if checked:
                        selected_sensors.append(sens)
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
            df = gdf.copy()
            # Filter by unified 'sat' and 'sensor' columns only when the user
            # has made an explicit selection. If nothing is selected, include
            # all rows from the constellation so date-only filtering still works.
            if sat_list and "sat" in df.columns:
                df = df[df["sat"].isin(sat_list)]
            if sens_list and "sensor" in df.columns:
                df = df[df["sensor"].isin(sens_list)]
            if df.empty:
                continue
            # Identify a date column by searching for keywords
            date_col = infer_acquisition_date_column(list(df.columns))
            if date_col is None:
                date_col = infer_column_name(list(df.columns), ["date", "acq", "start", "time"])
            if date_col:
                dates = parse_date_column(df, date_col)
                df["__acq_datetime__"] = dates
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                mask = (df["__acq_datetime__"] >= start_dt) & (df["__acq_datetime__"] <= end_dt)
                df = df.loc[mask]
            if not df.empty:
                filtered_frames.append(df)
        # Combine all selected frames
        if filtered_frames:
            result_gdf = gpd.GeoDataFrame(pd.concat(filtered_frames, ignore_index=True), crs="EPSG:4326")
        else:
            fallback_columns: List[str] = ["geometry"]
            if const_data:
                first_constellation = next(iter(const_data.values()))
                fallback_columns = list(first_constellation.columns)
            result_gdf = gpd.GeoDataFrame(columns=fallback_columns, geometry="geometry", crs="EPSG:4326")

        # Display map
        if not result_gdf.empty:
            # For the map, use only the geometries representing individual frames.  Frames are
            # identified by the presence of an area coverage column (e.g. 'areacovere').  If
            # such a column exists, only rows with non‑null and positive values are kept.
            frames_gdf = result_gdf.copy()
            area_col: Optional[str] = None
            for col in frames_gdf.columns:
                lc = col.lower()
                if "area" in lc and "cover" in lc:
                    area_col = col
                    break
            if area_col:
                # Convert to numeric to compare
                area_series = pd.to_numeric(frames_gdf[area_col], errors="coerce")
                frames_gdf = frames_gdf[area_series.notna() & (area_series > 0)]
                frames_gdf = frames_gdf.copy()
            # Compute combined coverage of selected swaths over the AOI using frames_gdf
            coverage_message = ""
            if aoi is not None and not aoi.empty and not frames_gdf.empty:
                try:
                    swath_union = unary_union(frames_gdf.geometry)
                    aoi_union = unary_union(aoi.geometry)
                    intersection = swath_union.intersection(aoi_union)
                    result_area_gdf = gpd.GeoSeries([intersection], crs="EPSG:4326").to_crs("EPSG:3857")
                    aoi_area_gdf = gpd.GeoSeries([aoi_union], crs="EPSG:4326").to_crs("EPSG:3857")
                    covered_area = result_area_gdf.area.iloc[0] / 1e6
                    total_area = aoi_area_gdf.area.iloc[0] / 1e6
                    coverage_percent = (covered_area / total_area) * 100 if total_area > 0 else 0
                    coverage_message = (
                        f"Coverage of AOI: {coverage_percent:.1f}% (\ncovered "
                        f"{covered_area:.1f} km² out of {total_area:.1f} km²)"
                    )
                except Exception as exc:
                    coverage_message = f"Unable to compute coverage: {exc}"
            # Build map layers using frames_gdf instead of result_gdf
            layers = build_map_layers(frames_gdf, aoi, show_aoi)
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
            selected_range = f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
            overlap_notes: List[str] = []
            for const, (span_start, span_end) in const_spans.items():
                if span_start is None or span_end is None:
                    continue
                if span_end < pd.Timestamp(start_date) or span_start > pd.Timestamp(end_date):
                    overlap_notes.append(
                        f"{const}: available {span_start.date()} to {span_end.date()}"
                    )
            if overlap_notes:
                st.warning(
                    f"No swaths match the selected filters for {selected_range}. "
                    f"Some loaded constellations do not overlap that window: "
                    + "; ".join(overlap_notes[:4])
                )
            else:
                st.warning(f"No swaths match the selected filters for {selected_range}.")



def main_v2() -> None:
    """Updated entry point with a compact editable summary table."""
    st.set_page_config(page_title="Acquisition Plans Viewer", layout="wide")
    inject_brand_styles()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "ats_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as logo_file:
            logo_bytes = logo_file.read()
        logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")
        logo_html = (
            f'<a href="https://www.ariastechsolutions.com/" target="_blank">'
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'alt="Arias Tech Solutions Logo" style="height:80px;" /></a>'
        )
        st.markdown(logo_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="qc-hero">
            <div class="eyebrow">Tasking Console</div>
            <h1 style="margin:0;">Satellite Acquisition Plans over Qatar EEZ</h1>
            <div class="subtitle">Tasking windows, sensor selection, and frame visibility controls tuned for operational review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    base_dir = script_dir
    aoi = load_aoi(os.path.join(base_dir, "Qatar_eez.kml"))
    const_data = load_constellations_data(base_dir)
    const_spans: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {
        const: get_data_date_span(gdf) for const, gdf in const_data.items()
    }

    default_start = datetime(2026, 2, 1)
    default_end = datetime(2027, 1, 31)

    st.sidebar.header("Constellations and Selection")
    selection: Dict[str, Tuple[List[str], List[str]]] = {}
    for const, gdf in const_data.items():
        with st.sidebar.expander(const, expanded=False):
            span_start, span_end = const_spans.get(const, (None, None))
            if span_start is not None and span_end is not None:
                st.caption(f"Available dates: {span_start.date()} to {span_end.date()}")
            else:
                st.caption("Available dates: unavailable")

            sat_values = sorted(gdf["sat"].dropna().unique().tolist()) if "sat" in gdf.columns else []
            sensor_values = sorted(gdf["sensor"].dropna().unique().tolist()) if "sensor" in gdf.columns else []
            sat_keys = [f"{const}_sat_{i}" for i in range(len(sat_values))]
            sensor_keys = [f"{const}_sens_{j}" for j in range(len(sensor_values))]
            st.checkbox(f"Select all {const}", key=f"{const}_all", on_change=toggle_all_constellation, args=(const, sat_keys + sensor_keys))

            selected_sats: List[str] = []
            if sat_values:
                st.markdown("**Satellites**")
                for i, sat in enumerate(sat_values):
                    sat_key = sat_keys[i]
                    if sat_key not in st.session_state:
                        st.session_state[sat_key] = False
                    if st.checkbox(sat, key=sat_key):
                        selected_sats.append(sat)

            selected_sensors: List[str] = []
            if sensor_values:
                st.markdown("**Sensors**")
                for j, sens in enumerate(sensor_values):
                    sens_key = sensor_keys[j]
                    if sens_key not in st.session_state:
                        st.session_state[sens_key] = False
                    if st.checkbox(sens, key=sens_key):
                        selected_sensors.append(sens)

            selection[const] = (selected_sats, selected_sensors)

    has_any_selection = any(
        sat_list or sens_list for sat_list, sens_list in selection.values()
    )

    st.sidebar.header("Date Range")
    start_date, end_date = st.sidebar.date_input(
        "Acquisition period",
        value=(default_start.date(), default_end.date()),
        min_value=default_start.date(),
        max_value=default_end.date(),
    )
    show_aoi = st.sidebar.checkbox("Show AOI boundary", value=True)
    apply = st.sidebar.button("Apply filters", disabled=not has_any_selection)
    if not has_any_selection:
        st.sidebar.caption("Select at least one satellite or sensor to enable filtering.")

    if apply:
        if gpd is None or unary_union is None or pdk is None:
            st.error("One or more required libraries are missing. Please install geopandas, shapely and pydeck to run this app.")
            return

        if not has_any_selection:
            st.warning("no satellite/sensor was selected")
            st.session_state.pop("frame_result_gdf", None)
            st.session_state.pop("frame_summary_df", None)
            st.session_state["no_results_message"] = "no satellite/sensor was selected"
            return

        filtered_frames: List[gpd.GeoDataFrame] = []
        for const, (sat_list, sens_list) in selection.items():
            gdf = const_data.get(const)
            if gdf is None or gdf.empty:
                continue
            df = gdf.copy()
            if not sat_list and not sens_list:
                continue
            if sat_list and "sat" in df.columns:
                df = df[df["sat"].isin(sat_list)]
            if sens_list and "sensor" in df.columns:
                df = df[df["sensor"].isin(sens_list)]
            if df.empty:
                continue

            date_col = infer_acquisition_date_column(list(df.columns))
            if date_col is None:
                date_col = infer_column_name(list(df.columns), ["date", "acq", "start", "time"])
            if date_col:
                df["__acq_datetime__"] = parse_date_column(df, date_col)
                start_dt = pd.Timestamp(start_date)
                end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                df = df.loc[(df["__acq_datetime__"] >= start_dt) & (df["__acq_datetime__"] <= end_dt)]
            if not df.empty:
                filtered_frames.append(df)

        if filtered_frames:
            result_gdf = gpd.GeoDataFrame(pd.concat(filtered_frames, ignore_index=True), crs="EPSG:4326").reset_index(drop=True)
            st.session_state["frame_result_gdf"] = result_gdf
            st.session_state["frame_visibility_state"] = [True] * len(result_gdf)
            st.session_state["frame_filter_range"] = f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
            st.session_state.pop("no_results_message", None)
        else:
            st.session_state.pop("frame_result_gdf", None)
            st.session_state.pop("frame_visibility_state", None)
            selected_range = f"{start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
            overlap_notes: List[str] = []
            for const, (span_start, span_end) in const_spans.items():
                if span_start is None or span_end is None:
                    continue
                if span_end < pd.Timestamp(start_date) or span_start > pd.Timestamp(end_date):
                    overlap_notes.append(f"{const}: available {span_start.date()} to {span_end.date()}")
            if overlap_notes:
                st.session_state["no_results_message"] = (
                    f"No swaths match the selected filters for {selected_range}. Some loaded constellations do not overlap that window: "
                    + "; ".join(overlap_notes[:4])
                )
            else:
                st.session_state["no_results_message"] = f"No swaths match the selected filters for {selected_range}."

    if "frame_result_gdf" in st.session_state:
        render_results_panel(st.session_state["frame_result_gdf"], aoi, show_aoi)
    else:
        no_results_message = st.session_state.pop("no_results_message", None)
        if no_results_message:
            st.warning(no_results_message)


if __name__ == "__main__":
    main_v2()
