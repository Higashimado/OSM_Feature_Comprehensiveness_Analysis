import math
import numpy as np
import os
import pandas as pd
import requests
import sys
import time
from pathlib import Path
from pypinyin import lazy_pinyin
from lxml import etree
from tqdm import tqdm


# ==================================================
# CONFIGURATION
# ==================================================

OFF_LINE = False
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "OSM-Mapper"}

OUTPUT_DIR = Path("osm_cache")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# SAFE HTTP LAYER
# ==================================================

def safe_post(url, data, headers, timeout=180, max_retry=5):
    for attempt in range(1, max_retry + 1):
        try:
            r = requests.post(
                url,
                data=data,
                headers=headers,
                timeout=timeout
            )
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            print(f"    HTTP error (attempt {attempt}/{max_retry}): {e}")
            time.sleep(2 * attempt)
    return None

def safe_overpass_json(query):
    r = safe_post(
        OVERPASS_URL,
        data=query,
        headers=HEADERS,
        timeout=60
    )
    if r is None:
        return None
    try:
        return r.json()
    except Exception as e:
        print(f"    JSON parse error: {e}")
        return None

def safe_overpass_xml(query):
    r = safe_post(
        OVERPASS_URL,
        data=query,
        headers=HEADERS,
        timeout=180
    )
    if r is None:
        return None
    return r.text

def gd_map(addr, timeout=5, max_retry=5):
    if OFF_LINE:
        return []
    raise NotImplementedError("Please override 'gd_map()' to provide a valid API implementation.")
    return []


# ==================================================
# UTILITIES
# ==================================================

def format_admin_pinyin(addr: str) -> str:
    parts = addr.strip().split()
    cleaned = []

    for part in parts:
        if part:
            p = "".join(lazy_pinyin(part))
            cleaned.append(p.capitalize())

    return "_".join(cleaned)

def save_osm(content, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

def load_osm(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read()

def parse_osm(osm_xml):
    return etree.fromstring(osm_xml.encode("utf-8"))


# ==================================================
# OSM DATA ACQUISITION WITH CACHE
# ==================================================

def get_osm_data(lon, lat, radius_m, osm_file, force_update=False):
    if osm_file.exists() and not force_update:
        return load_osm(osm_file)

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

    save_osm(osm_xml, osm_file)
    time.sleep(2)
    return osm_xml


# ==================================================
# EXISTENCE CHECK FROM OSM CACHE
# ==================================================

def infer_node_and_boundary(addr, lon, lat, osm_root=None):

    addr_4 = addr.split()[-1]
    node_id = -1
    boundary_id = -1
    node_lon = node_lat = None

    def match_name(addr_4, tags, exact_match=False):
        if exact_match:
            return (addr_4 == tags.get("name") or
                       "name:zh" in tags and addr_4 == tags.get("name:zh") or
                       "official_name" in tags and addr_4 == tags.get("official_name") or
                       "official_name:zh" in tags and addr_4 == tags.get("official_name:zh") or
                       "alt_name" in tags and addr_4 == tags.get("alt_name") or
                       "alt_name:zh" in tags and addr_4 == tags.get("alt_name:zh") or
                       "old_name" in tags and addr_4 == tags.get("old_name") or
                       "old_name:zh" in tags and addr_4 == tags.get("old_name:zh") or
                       "short_name" in tags and addr_4 == tags.get("short_name") or
                       "short_name:zh" in tags and addr_4 == tags.get("short_name:zh")
                   )
        else:
            return (addr_4 in tags.get("name") or
                       "official_name" in tags and addr_4 in tags.get("official_name") or
                       "alt_name" in tags and addr_4 in tags.get("alt_name") or
                       "old_name" in tags and addr_4 in tags.get("old_name") or
                       "short_name" in tags and addr_4 in tags.get("short_name")
                   )

    if osm_root != None:

        place_cands = {}

        for n in osm_root.xpath("//node"):
            tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
            if "name" not in tags:
                continue
            if ("capital" in tags or "place" in tags or "place:CN" in tags
                or ("amenity" in tags and tags.get("amenity") == "townhall"  )
                or ("office"  in tags and tags.get("office")  == "government")):
                place_cands[n.get("id")] = n
                if node_id < 0 and match_name(addr_4, tags):
                    node_lon = float(n.get("lon"))
                    node_lat = float(n.get("lat"))
                    if "place" in tags:
                        if "capital" in tags or match_name(addr_4, tags, exact_match=True):
                            node_id = int(n.get("id"))
    
        for r in osm_root.xpath("//relation"):
            tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
            if "name" not in tags:
                continue
            if ((tags.get("boundary") == "administrative" or tags.get("boundary") == "historic")
                and match_name(addr_4, tags)):
                boundary_id = int(r.get("id"))
                for m in r.findall("member"):
                    if m.get("type") != "node":
                        continue
                    if m.get("role") not in ("label", "admin_centre"):
                        continue
                    n_id = m.get("ref")
                    node = place_cands.get(n_id)
                    if node is None:
                        continue
                    node_tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
                    node_id = -2
                    node_lon = float(node.get("lon"))
                    node_lat = float(node.get("lat"))
                    if "place" in node_tags and match_name(addr_4, node_tags):
                        if "capital" in node_tags or match_name(addr_4, tags, exact_match=True):
                            node_id = int(node.get("id"))
                            break
                if match_name(addr_4, tags, exact_match=True):
                    break

    else:

        if OFF_LINE:
            return -1, -1, None, None

        radius_m = 15000
        query = f"""
        [out:xml][timeout:60];
        (
          node(around:{radius_m},{lat},{lon})
            ["capital"]
            ["place"]
            ["name"~"{addr_4}"];
          node(around:{radius_m},{lat},{lon})
            ["capital"]
            ["place"]
            ["alt_name"~"{addr_4}"];
          node(around:{radius_m},{lat},{lon})
            ["capital"]
            ["place"]
            ["official_name"~"{addr_4}"];
          node(around:{radius_m},{lat},{lon})
            ["place"]
            ["old_name"~"{addr_4}"];
          node(around:{radius_m},{lat},{lon})
            ["capital"]
            ["place"]
            ["short_name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="administrative"]
            ["name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="administrative"]
            ["alt_name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="administrative"]
            ["official_name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="administrative"]
            ["old_name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="administrative"]
            ["short_name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="historic"]
            ["name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="historic"]
            ["alt_name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="historic"]
            ["official_name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="historic"]
            ["old_name"~"{addr_4}"];
          relation(around:{radius_m},{lat},{lon})
            ["boundary"="historic"]
            ["short_name"~"{addr_4}"];
        );
        out body;
        """
        xml = safe_overpass_xml(query)
        if xml is None:
            return -1, -1, None, None
        root = parse_osm(xml)

        for n in root.xpath("//node"):
            node_id = int(n.get("id"))
            node_lon = float(n.get("lon"))
            node_lat = float(n.get("lat"))
            break

        for r in root.xpath("//relation"):
            boundary_id = int(r.get("id"))
            break

    return node_id, boundary_id, node_lon, node_lat


# ==================================================
# FEATURE COUNTING
# ==================================================

def count_features(osm_root):
    places = set()
    roads = {
        "trunk": 0,
        "primary": 0,
        "secondary": 0,
        "tertiary": 0,
        "res_uncl": 0,
        "bus_stop": 0,
        "parking": 0,
        "fuel": 0
    }
    road_types = set()
    amenities = {
        "gov": 0, 
        "health": 0, 
        "school": 0, 
        "police": 0, 
        "post": 0,
        "bank": 0,
        "shop": 0
    }
    buildings = set()
    landuse_types = set()

    for way in osm_root.xpath("//way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if "highway" in tags:
            h = tags["highway"]
            if h in roads:
                roads[h] += 1
            elif h in ("residential", "unclassified"):
                roads["res_uncl"] += 1
            if h in ("residential", "unclassified", "service", "track", 
                     "cycleway", "pedestrian", "footway", "path", "steps"):
                if not h in road_types:
                    road_types.add(h)
        if "building" in tags:
            buildings.add(way.get("id"))
        if "man_made" in tags:
            buildings.add(way.get("id"))
        if tags.get("amenity") == "townhall" or tags.get("office") == "government":
            amenities["gov"] += 1
        if tags.get("amenity") in ("hospital", "clinic"):
            amenities["health"] += 1
        if tags.get("amenity") == "school" or tags.get("education") == "school":
            amenities["school"] += 1
        if tags.get("amenity") == "police":
            amenities["police"] += 1
        if tags.get("amenity") == "post_office":
            amenities["post"] += 1
        if tags.get("amenity") == "bank":
            amenities["bank"] += 1
        if "shop" in tags or "cuisine" in tags or tags.get("tourism") in ("hotel", "apartment"):
            amenities["shop"] += 1
        if "bus" in tags or tags.get("highway") == "bus_stop" or "public_transport" in tags:
            roads["bus_stop"] += 1
        if tags.get("amenity") == "parking":
            roads["parking"] += 1
        if tags.get("amenity") == "fuel":
            roads["fuel"] += 1
        if "landuse" in tags or "natrual" in tags:
            landuse_types.add(tags["landuse"])
        for h in ("leisure", "tourism", "waterway"):
            if h in tags:
                landuse_types.add(h)

    for node in osm_root.xpath("//node"):
        tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
        if "place" in tags and "name" in tags:
            places.add(node.get("id"))
        if tags.get("amenity") == "townhall" or tags.get("office") == "government":
            amenities["gov"] += 1
        if tags.get("amenity") in ("hospital", "clinic"):
            amenities["health"] += 1
        if tags.get("amenity") == "school" or tags.get("education") == "school":
            amenities["police"] += 1
        if tags.get("amenity") == "police":
            amenities["police"] += 1
        if tags.get("amenity") == "post_office":
            amenities["post"] += 1
        if tags.get("amenity") == "bank":
            amenities["bank"] += 1
        if "shop" in tags or "cuisine" in tags or tags.get("tourism") in ("hotel", "apartment"):
            amenities["shop"] += 1
        if "bus" in tags or tags.get("highway") == "bus_stop" or "public_transport" in tags:
            roads["bus_stop"] += 1
        if tags.get("amenity") == "fuel":
            roads["fuel"] += 1
        if tags.get("amenity") == "parking":
            roads["parking"] += 1
        if "man_made" in tags:
            buildings.add(way.get("id"))
        for h in ("leisure", "tourism", "waterway"):
            if h in tags:
                landuse_types.add(h)

    return roads, amenities, len(places), len(buildings), len(road_types), len(landuse_types)


# ==================================================
# SCORING
# ==================================================

def cap(x, m):
    return min(x, m)

def compute_score(row):
    scores = [0, 0, 0, 0]

    # Nodes & boundary (20)
    if row["node"] > 0:
        scores[0] += 7
    if row["boundary"] > 0:
        scores[0] += 8
        if row["node"] == -1:
            scores[0] += 7
    scores[0] += cap(row["places_total_3km"], 5)

    scores[0] = cap(scores[0], 20)

    # Roads (30)
    if row["road_trunk_3km"] + row["road_primary_3km"] + row["road_secondary_3km"] > 0:
        scores[1] += 5
    scores[1] += cap(row["road_tertiary_3km"], 5)
    scores[1] += row["road_res_uncl_1km"] * 0.3 + row["road_res_uncl_3km"] * 0.2
    scores[1] = cap(scores[1], 20)

    if row["road_bus_stop_3km"] > 0:
        scores[1] += 2
    if row["road_parking_3km"] > 0:
        scores[1] += 2
    if row["road_fuel_3km"] > 0:
        scores[1] += 2
    scores[1] += cap(row["road_types_3km"], 4)

    scores[1] = cap(scores[1], 30)

    # Amenities (30)
    if row["amenity_gov_3km"] > 0:
        scores[2] += 5
    if row["amenity_health_1km"] > 0:
        scores[2] += 5
    if row["amenity_school_1km"] > 0:
        scores[2] += 5
    if row["amenity_police_1km"] > 0:
        scores[2] += 5
    if row["amenity_post_1km"] > 0:
        scores[2] += 2
    if row["amenity_bank_1km"] > 0:
        scores[2] += 2
    scores[2] += cap(row["amenity_shop_1km"], 6)

    scores[2] = cap(scores[2], 30)

    # Landuses & Buildings (20)
    scores[3] += cap((row["buildings_total_1km"] + row["buildings_total_3km"]) * 0.1, 12)
    scores[3] += cap(row["landuse_types_3km"], 8)

    scores[3] = cap(scores[3], 20)

    return scores 


# ==================================================
# MAIN PIPELINE
# ==================================================

def process_places(input_file, output_csv):
    rows = []

    with open(input_file, encoding="utf-8") as f:
        places = [l.strip() for l in f if l.strip()]

    for addr in tqdm(places):

        parts = addr.split()
        if len(parts) != 4 and len(parts) != 6:
            continue

        addr_1, addr_2, addr_3, addr_4 = parts[0:4]
        addr = " ".join([addr_1, addr_2, addr_3, addr_4])

        try:
            if "addr_1" in parts[0]:
                continue
            if len(parts) == 6:
                lon, lat = map(float, parts[4:6])
            elif len(parts) == 4:
                geo = gd_map(addr=addr)
                if "," not in geo:
                    continue
                lon, lat = map(float, geo.split(","))
            else:
                continue
        except Exception as e:
            print(f"Caught an exception: {e}")
            print(f"Skipped ...")
            continue

        pname = format_admin_pinyin(addr)
        row = {
            "addr_1": addr_1,
            "addr_2": addr_2,
            "addr_3": addr_3,
            "addr_4": addr_4,
            "lon": lon,
            "lat": lat
        }

        # ---- Existence inference FROM 3 km OSM ----
        r = 3
        osm_file = OUTPUT_DIR / f"{pname}_{r}km.osm"
        osm_root = None
        if osm_file.is_file():
            xml = get_osm_data(lon, lat, r * 1000, osm_file)
            osm_root = parse_osm(xml)

        force_update = False

        node_id, boundary_id, nlon, nlat = infer_node_and_boundary(addr, lon, lat, osm_root)
        if (not nlon is None and not nlat is None) and (nlon != lon or nlat != lat):
            lon, lat = nlon, nlat
            row["lon"] = lon
            row["lat"] = lat
            force_update = True
        
        print(f"\nProcessing: {addr} {lon} {lat}")

        for r in (3, 1):
            osm_file = OUTPUT_DIR / f"{pname}_{r}km.osm"
#           if node_id == -1 and not OFF_LINE:
#               force_update = True
            xml = get_osm_data(lon, lat, r * 1000, osm_file, force_update)
            if xml is None:
                break

            root = parse_osm(xml)
            roads, amens, pcnt, bcnt, rtnt, ltnt = count_features(root)

            for k, v in roads.items():
                row[f"road_{k}_{r}km"] = v
            for k, v in amens.items():
                row[f"amenity_{k}_{r}km"] = v
            row[f"places_total_{r}km"] = pcnt
            row[f"buildings_total_{r}km"] = bcnt
            row[f"road_types_{r}km"] = rtnt
            row[f"landuse_types_{r}km"] = ltnt

        row["node"] = node_id
        row["boundary"] = boundary_id

        rows.append(row)

    if len(rows) < 1:
        print("\nNo valid records!")
        exit(1)

    df = pd.DataFrame(rows)
    df["u_addr_3"] = df[  "addr_2"].astype(str) + df["addr_3"].astype(str)
    df["u_addr_4"] = df["u_addr_3"].astype(str) + df["addr_4"].astype(str)
    score_cols = df.apply(compute_score, axis=1, result_type="expand")

    # Scoring
    score_cols.columns = ["score_1", "score_2", "score_3", "score_4"]
    df[score_cols.columns] = score_cols
    df["score"] = score_cols.sum(axis=1)

    # Aggregation
    for lvl in ("u_addr_3", "addr_2", "addr_1"):
        df[f"{lvl}_avg_score"] = df.groupby(lvl)["score"].transform("mean")
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
        print("Usage: %s [place_list]" % (os.path.basename(__file__)))
        print("")
        print("Example of record in place_list:")
        print(" 安徽省 合肥市 瑶海区 明光路街道 117.3016267 31.8584716")
        print(" 安徽省 合肥市 瑶海区 胜利路街道 117.2963607 31.8650544")
        print("")
        exit(0)

    process_places(
        input_file=sys.argv[1],
        output_csv="feature_comprehensiveness_statistics.csv"
    )
