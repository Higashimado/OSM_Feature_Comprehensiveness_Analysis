import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
import requests
from lxml import etree
from pypinyin import lazy_pinyin
from tqdm import tqdm


# ==================================================
# CONFIGURATION
# ==================================================

OFFLINE = False
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "OSM-Mapper"}

# Sentinel values for node_id
NODE_NOT_FOUND = -1   # no place node located
NODE_BAD_TAG   = -2   # node found via boundary member but lacks proper place/capital tag

CACHE_DIR = Path("osm_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ==================================================
# SAFE HTTP LAYER
# ==================================================

def safe_post(url, data, headers, timeout=180, max_retry=5):
    for attempt in range(1, max_retry + 1):
        try:
            resp = requests.post(url, data=data, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            print(f"    HTTP error (attempt {attempt}/{max_retry}): {e}")
            time.sleep(2 * attempt)
    return None


def safe_overpass_xml(query):
    resp = safe_post(OVERPASS_URL, data=query, headers=HEADERS, timeout=180)
    if resp is None:
        return None
    return resp.text


def geocode(addr, timeout=5, max_retry=5):
    """Return 'lon,lat' string for the given address, or None on failure.
    Override this function to plug in a real geocoding API.
    """
    if OFFLINE:
        return None
    raise NotImplementedError("Please override 'geocode()' to provide a valid API implementation.")


# ==================================================
# UTILITIES
# ==================================================

def to_pinyin_slug(addr: str) -> str:
    """Convert a space-separated Chinese address to a CamelCase_pinyin slug."""
    parts = addr.strip().split()
    return "_".join("".join(lazy_pinyin(p)).capitalize() for p in parts if p)


def write_osm(content, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def read_osm(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read()


def parse_osm(osm_xml):
    return etree.fromstring(osm_xml.encode("utf-8"))


def tags_of(elem):
    return dict(zip(elem.xpath("tag/@k"), elem.xpath("tag/@v")))


# ==================================================
# OSM DATA ACQUISITION WITH CACHE
# ==================================================

def get_osm_data(lon, lat, radius_m, osm_file, force_update=False):
    if osm_file.exists() and not force_update:
        return read_osm(osm_file)

    print(f"    Downloading OSM ({radius_m} m)")

    query = f"""
    [out:xml][timeout:180];
    (
      node(around:{radius_m},{lat},{lon});
      way(around:{radius_m},{lat},{lon});
      relation(around:{radius_m},{lat},{lon});
    );
    out body;
    >;
    out skel qt;
    """

    osm_xml = safe_overpass_xml(query)
    if osm_xml is None:
        print("    Overpass failed after retries.")
        return None

    write_osm(osm_xml, osm_file)
    time.sleep(2)
    return osm_xml


# ==================================================
# EXISTENCE CHECK FROM OSM CACHE
# ==================================================

def find_admin_node(addr, lon, lat, osm_root=None):
    """Infer the administrative place node and boundary relation for addr.

    Returns (node_id, boundary_id, node_lon, node_lat).
    node_id == NODE_NOT_FOUND  : no matching node found
    node_id == NODE_BAD_TAG    : node found via boundary member but tag quality is low
    node_id > 0                : clean match
    """

    name_ = addr.split()[-1]
    node_id     = NODE_NOT_FOUND
    boundary_id = -1
    node_lon = node_lat = None

    # Priority order for name matching (lower index = higher priority).
    # not_name / not_name:zh are intentional exclusion markers, so they rank last.
    NAME_PRIORITY = [
        "official_name:zh", "official_name",
        "name:zh",          "name",
        "short_name:zh",    "short_name",
        "old_name:zh",      "old_name",
        "alt_name:zh",      "alt_name",
        "not_name:zh",      "not_name",
    ]

    def name_priority(name, tags, exact=False):
        """Return the priority index of the best matching name key, or None if no match.

        Lower return value = higher priority match.
        exact=True  : requires name == tag value
        exact=False : requires name to appear as a substring of tag value
        """
        for pri, key in enumerate(NAME_PRIORITY):
            val = tags.get(key)
            if val is None:
                continue
            if exact and name == val:
                return pri
            if not exact and name in val:
                return pri
        return None

    if osm_root is not None:

        node_pool = {}

        def dist2(nd):
            """Squared pseudo-distance to the prior coordinates (no sqrt needed)."""
            return (float(nd.get("lon")) - lon) ** 2 + (float(nd.get("lat")) - lat) ** 2

        # Track the best-priority match found so far for the place node.
        # Tie-break on distance to prior coordinates (closer = better).
        best_pri  = None   # priority index of current best candidate
        best_dist = None   # squared distance of current best candidate
        best_nd   = None   # the candidate node element

        for nd in osm_root.xpath("//node"):
            tags = tags_of(nd)
            if "name" not in tags:
                continue
            if ("capital" in tags or "place" in tags or "place:CN" in tags
                    or ("amenity" in tags and tags.get("amenity") == "townhall")
                    or ("office"  in tags and tags.get("office")  == "government")):
                node_pool[nd.get("id")] = nd
                pri = name_priority(name_, tags)
                if pri is None:
                    continue
                # Accept this candidate if it beats the current best priority,
                # and additionally has a proper place tag with capital or exact match.
                if "place" in tags and (
                        "capital" in tags or name_priority(name_, tags, exact=True) is not None):
                    d = dist2(nd)
                    if (best_pri is None
                            or pri < best_pri
                            or (pri == best_pri and d < best_dist)):
                        best_pri  = pri
                        best_dist = d
                        best_nd   = nd
                        node_lon  = float(nd.get("lon"))
                        node_lat  = float(nd.get("lat"))
                        node_id   = int(nd.get("id"))
                elif best_nd is None:
                    # Fallback: record coords even without a confirmed node_id
                    node_lon = float(nd.get("lon"))
                    node_lat = float(nd.get("lat"))

        for rel in osm_root.xpath("//relation"):
            tags = tags_of(rel)
            if "name" not in tags:
                continue
            if (tags.get("boundary") in ("administrative", "historic")
                    and name_priority(name_, tags) is not None):
                boundary_id = int(rel.get("id"))
                for m in rel.findall("member"):
                    if m.get("type") != "node":
                        continue
                    if m.get("role") not in ("label", "admin_centre"):
                        continue
                    nd = node_pool.get(m.get("ref"))
                    if nd is None:
                        continue
                    nd_tags = tags_of(nd)
                    nd_pri  = name_priority(name_, nd_tags)
                    node_id  = NODE_BAD_TAG
                    node_lon = float(nd.get("lon"))
                    node_lat = float(nd.get("lat"))
                    if "place" in nd_tags and nd_pri is not None:
                        if "capital" in nd_tags or name_priority(name_, tags, exact=True) is not None:
                            node_id = int(nd.get("id"))
                            break
                if name_priority(name_, tags, exact=True) is not None:
                    break

    else:

        if OFFLINE:
            return NODE_NOT_FOUND, -1, None, None
 
        # Fetch all candidate place nodes and boundary relations within radius,
        # then let the same name_priority logic above do the matching.
        radius_m = 15000
        query = (
            f"[out:xml][timeout:60];\n("
            f"  node(around:{radius_m},{lat},{lon})['place']['name'];\n"
            f"  node(around:{radius_m},{lat},{lon})['place']['capital'];\n"
            f"  node(around:{radius_m},{lat},{lon})['amenity'='townhall']['name'];\n"
            f"  node(around:{radius_m},{lat},{lon})['office'='government']['name'];\n"
            f"  relation(around:{radius_m},{lat},{lon})['boundary'='administrative']['name'];\n"
            f"  relation(around:{radius_m},{lat},{lon})['boundary'='historic']['name'];\n"
            f");\nout body;"
        )
        xml = safe_overpass_xml(query)
        if xml is None:
            return NODE_NOT_FOUND, -1, None, None
 
        return find_admin_node(addr, lon, lat, osm_root=parse_osm(xml))

    return node_id, boundary_id, node_lon, node_lat


# ==================================================
# FEATURE COUNTING
# ==================================================

def count_osm_features(osm_root):
    hw = {          # highway / transport counts
        "trunk":    0,
        "primary":  0,
        "secondary":0,
        "tertiary": 0,
        "res_uncl": 0,
        "service":  0,
        "cycleway": 0,
        "footway":  0,
        "bus_stop": 0,
        "parking":  0,
        "fuel":     0,
    }
    hw_types = set()
    am = {          # amenity / POI counts
        "gov":        0,
        "medical":    0,
        "healthcare": 0,
        "school":     0,
        "education":  0,
        "police":     0,
        "post":       0,
        "bank":       0,
        "shop":       0,
    }
    places     = set()
    buildings  = set()
    bld_parts  = set()
    bld_levels = set()
    man_mades  = set()
    addresses  = set()
    lu = {          # landuse counts
        "residential": 0,
        "commercial":  0,
        "industrial":  0,
        "retail":      0,
        "construction":0,
    }
    leisures   = set()
    tourisms   = set()
    lu_types   = set()
    wikidata   = set()

    # -- highway (ways / relations only) --
    for rw in osm_root.xpath(
            "//node[tag[@k='highway']] | //way[tag[@k='highway']] | //relation[tag[@k='highway']]"):
        h = rw.xpath("string(tag[@k='highway']/@v)")
        if h in hw:
            hw[h] += 1
        elif h in ("residential", "unclassified"):
            hw["res_uncl"] += 1
        elif h in ("corridor", "elevator", "ladder", "path", "pedestrian", "steps"):
            hw["footway"] += 1
        hw_types.add(h)
 
    # -- building / building:part (ways / relations only) --
    for rw in osm_root.xpath(
            "//way[tag[@k='building'] or tag[@k='building:part']]"
            " | //relation[tag[@k='building'] or tag[@k='building:part']]"):
        eid = rw.get("id")
        if rw.xpath("tag[@k='building']"):
            buildings.add(eid)
        if rw.xpath("tag[@k='building:part']"):
            bld_parts.add(eid)
        if rw.xpath("tag[@k='building:levels']"):
            bld_levels.add(eid)
 
    # -- landuse / natural (ways / relations only) --
    for rw in osm_root.xpath(
            "//way[tag[@k='landuse'] or tag[@k='natural']]"
            " | //relation[tag[@k='landuse'] or tag[@k='natural']]"):
        lv = rw.xpath("string(tag[@k='landuse']/@v)")
        if lv:
            if lv in lu:
                lu[lv] += 1
            lu_types.add(lv)
 
    # ------------------------------------------------------------------
    # Amenity / man_made / transport / leisure -- shared across ways, relations,
    # and nodes; XPath pre-selects only elements with a relevant key.
    # ------------------------------------------------------------------
    _AM_HW_KEYS = (
        "tag[@k='amenity'] or tag[@k='man_made'] or tag[@k='office']"
        " or tag[@k='shop'] or tag[@k='cuisine'] or tag[@k='leisure'] or tag[@k='tourism']"
        " or tag[@k='bus'] or tag[@k='public_transport']"
        " or tag[@k='healthcare'] or tag[@k='education']"
        " or tag[@k='natural'] or tag[@k='waterway']"
        " or tag[@k='wikidata'] or tag[starts-with(@k, 'addr:')]"
    )
    for elem in osm_root.xpath(
            f"//node[{_AM_HW_KEYS}]"
            f" | //way[{_AM_HW_KEYS}]"
            f" | //relation[{_AM_HW_KEYS}]"):
        tags  = tags_of(elem)
        eid   = elem.get("id")
        av    = tags.get("amenity", "")
        edu_v = tags.get("education", "")
 
        if av == "townhall" or tags.get("office") == "government":
            am["gov"] += 1
        if av in ("hospital", "clinic", "doctors", "health_post"):
            am["medical"] += 1
        if av in ("hospital", "clinic", "doctors", "health_post", "dentist", "pharmacy") or tags.get("healthcare"):
            am["healthcare"] += 1
        if av == "school" or edu_v == "school":
            am["school"] += 1
        if av in ("kindergarten", "school", "college", "university", "library") or edu_v:
            am["education"] += 1
        if av == "police":
            am["police"] += 1
        if av == "post_office":
            am["post"] += 1
        if av == "bank":
            am["bank"] += 1
        if "shop" in tags or "cuisine" in tags or tags.get("tourism") in ("apartment", "hotel", "hostel"):
            am["shop"] += 1
 
        if "bus" in tags or tags.get("highway") == "bus_stop" or "public_transport" in tags:
            hw["bus_stop"] += 1
        if av == "parking":
            hw["parking"] += 1
        if av == "fuel":
            hw["fuel"] += 1

        if "man_made" in tags and eid not in buildings:
            man_mades.add(eid)
 
        if "leisure" in tags:
            lu_types.add("leisure")
            leisures.add(eid)
        if "tourism" in tags:
            lu_types.add("tourism")
            tourisms.add(eid)
        if "waterway" in tags:
            lu_types.add("waterway")
        if "natural" in tags:
            lu_types.add(tags["natural"])
        if "wikidata" in tags:
            wikidata.add(tags["wikidata"])

        if any(k.startswith("addr:") for k in tags):
            addresses.add(eid)

    # -- place nodes --
    for nd in osm_root.xpath("//node[tag[@k='place'] and tag[@k='name']]"):
        places.add(nd.get("id"))

    return {
        "hw":          hw,
        "am":          am,
        "lu":          lu,
        "n_pl":        len(places),
        "n_bld":       len(buildings),
        "n_blp":       len(bld_parts),
        "n_bll":       len(bld_levels),
        "n_mm":        len(man_mades),
        "n_ad":        len(addresses),
        "n_lsr":       len(leisures),
        "n_trs":       len(tourisms),
        "n_wd":        len(wikidata),
        "n_hw_types":  len(hw_types),
        "n_lu_types":  len(lu_types),
    }


def _parse_and_count(osm_file):
    """Worker for process pool: read, parse and count a cached OSM file.
    Returns (radius_km, count_dict) or (radius_km, None) on failure.
    radius_km is inferred from the filename stem (e.g. 'Slug_3km.osm').
    """
    radius_km = int(str(osm_file).split("_")[-1].replace("km.osm", ""))
    try:
        xml   = read_osm(osm_file)
        root  = parse_osm(xml)
        return radius_km, count_osm_features(root)
    except Exception as e:
        print(f"    Parse/count error for {osm_file}: {e}")
        return radius_km, None


# ==================================================
# SCORING
# ==================================================

def cap(x, m):
    return min(x, m)


def score_row(row):
    scores = [0, 0, 0, 0]

    # --- score_1: Nodes & boundary (20) ---
    if row["node"] > 0:
        scores[0] += 7
    if row["boundary"] > 0:
        scores[0] += 8
        if row["node"] == NODE_NOT_FOUND:
            scores[0] += 7
    scores[0] += cap(row["pl_3km"], 5)
    scores[0] = cap(scores[0], 20)

    # --- score_2: Roads & transport (30) ---
    if row["hw_trunk_3km"] + row["hw_primary_3km"] + row["hw_secondary_3km"] > 0:
        scores[1] += 5
    scores[1] += cap(row["hw_tertiary_3km"], 5)
    scores[1] += row["hw_res_uncl_1km"] * 0.3 + row["hw_res_uncl_3km"] * 0.2
    scores[1] = cap(scores[1], 20)

    if row["hw_bus_stop_3km"] > 0:
        scores[1] += 2
    if row["hw_parking_3km"] > 0:
        scores[1] += 2
    if row["hw_fuel_3km"] > 0:
        scores[1] += 2
    scores[1] += cap(row["hw_types_3km"] * 0.5, 4)
    scores[1] = cap(scores[1], 30)

    # --- score_3: Amenities (30) ---
    if row["am_gov_3km"] > 0:
        scores[2] += 5
    if row["am_medical_1km"] > 0:
        scores[2] += 5
    if row["am_school_1km"] > 0:
        scores[2] += 5
    if row["am_police_1km"] > 0:
        scores[2] += 5
    if row["am_post_1km"] > 0:
        scores[2] += 2
    if row["am_bank_1km"] > 0:
        scores[2] += 2
    scores[2] += cap(row["am_shop_1km"], 6)
    scores[2] = cap(scores[2], 30)

    # --- score_4: Landuses & Buildings (20) ---
    scores[3] += cap(
        (row["bld_1km"] + row["bld_3km"] + row["mm_1km"] + row["mm_3km"]) * 0.1,
        16
    )
    scores[3] += cap(row["lu_types_3km"] * 0.5, 4)
    scores[3] = cap(scores[3], 20)

    return scores


# ==================================================
# MAIN PIPELINE
# ==================================================

# Columns that are always recomputed from raw row data and must never be
# carried over verbatim from a cached CSV (scores, aggregates, addr keys).
_RECOMPUTED_COLS = {
    "u_addr_3", "u_addr_4",
    "s_nd", "s_hw", "s_am", "s_lu", "score",
    "u_addr_3_avg_score", "addr_2_avg_score", "addr_1_avg_score",
}

def _expected_columns():
    """Return the ordered list of output column labels that run_pipeline produces.
    Used to validate an existing CSV before deciding whether to reuse it.
    """
    base = ["addr_1", "addr_2", "addr_3", "addr_4", "lon", "lat"]
    per_radius = []
    for r in (3, 1):
        hw_keys  = ["trunk","primary","secondary","tertiary","res_uncl",
                    "service","cycleway","footway","bus_stop","parking","fuel"]
        am_keys  = ["gov","medical","healthcare","school","education",
                    "police","post","bank","shop"]
        lu_keys  = ["residential","commercial","industrial","retail","construction"]
        per_radius += [f"hw_{k}_{r}km"  for k in hw_keys]
        per_radius += [f"am_{k}_{r}km"  for k in am_keys]
        per_radius += [f"lu_{k}_{r}km"  for k in lu_keys]
        per_radius += [f"pl_{r}km", f"bld_{r}km", f"blp_{r}km", f"bll_{r}km", f"mm_{r}km",
                       f"ad_{r}km", f"lsr_{r}km", f"trs_{r}km", f"wd_{r}km",
                       f"hw_types_{r}km", f"lu_types_{r}km"]
    tail = ["node", "boundary", "u_addr_3", "u_addr_4",
            "score_1", "score_2", "score_3", "score_4", "score",
            "u_addr_3_avg_score", "addr_2_avg_score", "addr_1_avg_score"]
    return base + per_radius + tail
 
 
def _load_csv_cache(output_csv):
    """Try to load an existing output CSV and return an index for fast lookup.
 
    Returns (cache_df, index) where:
      cache_df : the full DataFrame (or None if file absent / columns mismatch)
      index    : dict mapping lookup-key -> row dict, or {} on failure
 
    Lookup keys (both tried for each input row):
      with coords    : (addr_1, addr_2, addr_3, addr_4, lon_str, lat_str)
      without coords : (addr_1, addr_2, addr_3, addr_4)
    """
    csv_path = Path(output_csv)
    if not csv_path.exists():
        return None, {}
 
    expected = _expected_columns()
#   print(expected)
#   exit(0)
    try:
        cache_df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    except Exception as e:
        print(f"  Warning: could not read existing CSV ({e}); running full pipeline.")
        return None, {}
 
    if set(cache_df.columns) != set(expected):
        missing  = [c for c in expected        if c not in cache_df.columns]
        extra    = [c for c in cache_df.columns if c not in expected]
        print("  Warning: existing CSV columns do not match expected schema.")
        if missing:
            print(f"    Missing : {missing}")
        if extra:
            print(f"    Extra   : {extra}")
        print("  Running full pipeline from OSM files.")
        return None, {}
 
    index = {}
    for _, r in cache_df.iterrows():
        key4  = (r["addr_1"], r["addr_2"], r["addr_3"], r["addr_4"])
        key6  = key4 + (r["lon"], r["lat"])
        row_d = r.to_dict()
        index.setdefault(key6,  row_d)
        index.setdefault(key4,  row_d)   # weaker key; only used when coords absent
 
    print(f"  Loaded {len(cache_df)} cached rows from {csv_path}.")
    return cache_df, index


def run_pipeline(input_file, output_csv):

    rows = []
 
    # ---- CSV cache: load once and reuse matching rows ----
    _cache_df, _cache_idx = _load_csv_cache(output_csv)
    _use_cache = bool(_cache_idx)

    with open(input_file, encoding="utf-8") as f:
        places = [line.strip() for line in f if line.strip()]

    for addr in tqdm(places):

        parts = addr.split()
        if len(parts) not in (4, 6):
            continue

        addr_1, addr_2, addr_3, addr_4 = parts[0:4]
        addr = " ".join([addr_1, addr_2, addr_3, addr_4])
        if "addr" in addr:   # skip header line
            continue

        try:
            if len(parts) == 6:
                lon, lat = map(float, parts[4:6])
            else:
                geo = geocode(addr=addr)
                if geo is None or "," not in geo:
                    continue
                lon, lat = map(float, geo.split(","))
        except Exception as e:
            print(f"Caught an exception: {e}")
            print("Skipped ...")
            continue

        slug = to_pinyin_slug(addr)

        # ---- CSV cache lookup ----
        if _use_cache:
            lon_s = f"{lon}"
            lat_s = f"{lat}"
            cached = (_cache_idx.get((addr_1, addr_2, addr_3, addr_4, lon_s, lat_s))
                      if lon is not None
                      else _cache_idx.get((addr_1, addr_2, addr_3, addr_4)))
            if cached is not None:
                _STR_COLS = {"addr_1", "addr_2", "addr_3", "addr_4"}
                row = {}
                for k, v in cached.items():
                    if k in _RECOMPUTED_COLS:
                        continue
                    if k in _STR_COLS:
                        row[k] = v
                    else:
                        row[k] = float(v) if v not in ("", None) else 0.0
                rows.append(row)
                continue

        row = {
            "addr_1": addr_1,
            "addr_2": addr_2,
            "addr_3": addr_3,
            "addr_4": addr_4,
            "lon": lon,
            "lat": lat,
        }

        # ---- Existence inference from 3 km OSM cache ----
        radius_km = 3
        osm_file  = CACHE_DIR / f"{slug}_{radius_km}km.osm"
        osm_root  = None
        if osm_file.is_file():
            xml      = get_osm_data(lon, lat, radius_km * 1000, osm_file)
            osm_root = parse_osm(xml)

        force_update = False

        node_id, boundary_id, nlon, nlat = find_admin_node(addr, lon, lat, osm_root)
        if (nlon is not None and nlat is not None) and (nlon != lon or nlat != lat):
            lon, lat = nlon, nlat
            row["lon"] = lon
            row["lat"] = lat
            force_update = True

        print(f"\nProcessing: {addr} {lon} {lat}")

        # -- Serial download: Overpass rejects concurrent requests --
        osm_files = {}
        for radius_km in (3, 1):
            osm_file = CACHE_DIR / f"{slug}_{radius_km}km.osm"
#           if node_id == NODE_NOT_FOUND and not OFFLINE:
#               force_update = True
            xml = get_osm_data(lon, lat, radius_km * 1000, osm_file, force_update)
            if xml is None:
                break
            osm_files[radius_km] = osm_file
 
        if not osm_files:
            continue
 
        # -- Parallel parse & count across the (up to) two cached files --
        def _write_count(row, radius_km, count):
            for k, v in count["hw"].items():
                row[f"hw_{k}_{radius_km}km"] = v
            for k, v in count["am"].items():
                row[f"am_{k}_{radius_km}km"] = v
            for k, v in count["lu"].items():
                row[f"lu_{k}_{radius_km}km"] = v
            row[f"pl_{radius_km}km"]       = count["n_pl"]
            row[f"bld_{radius_km}km"]      = count["n_bld"]
            row[f"blp_{radius_km}km"]      = count["n_blp"]
            row[f"bll_{radius_km}km"]      = count["n_bll"]
            row[f"mm_{radius_km}km"]       = count["n_mm"]
            row[f"ad_{radius_km}km"]       = count["n_ad"]
            row[f"lsr_{radius_km}km"]      = count["n_lsr"]
            row[f"trs_{radius_km}km"]      = count["n_trs"]
            row[f"wd_{radius_km}km"]       = count["n_wd"]
            row[f"hw_types_{radius_km}km"] = count["n_hw_types"]
            row[f"lu_types_{radius_km}km"] = count["n_lu_types"]

        if len(osm_files) == 1:
            # Only one file available; no benefit from spawning a process.
            r, count = _parse_and_count(next(iter(osm_files.values())))
            if count is not None:
                _write_count(row, r, count)
        else:
            with ProcessPoolExecutor(max_workers=2) as pool:
                futs = [(r, pool.submit(_parse_and_count, f))
                    for r, f in sorted(osm_files.items(), reverse=True)]
                for r, fut in futs:
                    _, count = fut.result()
                    if count is not None:
                        _write_count(row, r, count)

        row["node"]     = node_id
        row["boundary"] = boundary_id

        rows.append(row)

    if not rows:
        print("\nNo valid records!")
        exit(1)

    df = pd.DataFrame(rows)
    df["u_addr_3"] = df["addr_2"].astype(str) + df["addr_3"].astype(str)
    df["u_addr_4"] = df["u_addr_3"].astype(str) + df["addr_4"].astype(str)

    score_cols = df.apply(score_row, axis=1, result_type="expand")
    score_cols.columns = ["score_1", "score_2", "score_3", "score_4"]
    df[score_cols.columns] = score_cols
    df["score"] = score_cols.sum(axis=1)

    for lvl in ("u_addr_3", "addr_2", "addr_1"):
        df[f"{lvl}_avg_score"] = df.groupby(lvl)["score"].transform("mean")

    df = df[_expected_columns()]
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    addr3_score = (
        df.groupby("u_addr_3", as_index=False)["score"]
          .mean()
          .rename(columns={"score": "avg_score"})
          .sort_values("avg_score", ascending=False)
    )
    addr3_score["avg_score"] = addr3_score["avg_score"].round(2)

    print("\nTop 10 addr_3 by average score:")
    print(addr3_score.head(10).to_string(index=False))

    print("\nBottom 10 addr_3 by average score:")
    print(addr3_score.tail(10).to_string(index=False))


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: %s [place_list]" % os.path.basename(__file__))
        print()
        print("Example of record in place_list:")
        print(" 安徽省 合肥市 瑶海区 明光路街道 117.3016267 31.8584716")
        print(" 安徽省 合肥市 瑶海区 胜利路街道 117.2963607 31.8650544")
        print()
        exit(0)

    run_pipeline(
        input_file=sys.argv[1],
        output_csv="feature_comprehensiveness_statistics.csv",
    )
