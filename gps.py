import os
import cv2
import numpy as np
import exifread
from pyproj import Transformer

# --- EXIF GPS helpers ---


def _to_deg(val):
    parts = val.values
    def f(r): return r.num / r.den
    return f(parts[0]) + f(parts[1])/60.0 + f(parts[2])/3600.0


def read_latlon(path):
    with open(path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        lat = _to_deg(tags['GPS GPSLatitude'])
        lon = _to_deg(tags['GPS GPSLongitude'])
        if str(tags['GPS GPSLatitudeRef']) == 'N':
            lat = -lat
        if str(tags['GPS GPSLongitudeRef']) == 'E':
            lon = -lon
        return lat, lon
    except KeyError:
        return None, None

# --- project lat/lon to local meters ---


def make_transformer(lat0, lon0):
    return Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs",
        always_xy=True
    )


def ll_to_xy(T, lat, lon):
    x, y = T.transform(lon, lat)  # (lon, lat) order when always_xy=True
    return np.array([x, y], np.float32)


def mosaic_gps(folder, gsd_m_per_px=0.10, output="mosaic_gps.jpg"):
    # 1) load images & GPS
    items = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            path = os.path.join(folder, fn)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            lat, lon = read_latlon(path)
            items.append((path, img, lat, lon))

    # need at least one GPS
    anchor = next(((lat, lon)
                  for _, _, lat, lon in items if lat is not None), None)
    if anchor is None:
        raise RuntimeError(
            "No EXIF GPS found; provide GSD + a reference or use the feature-refined script.")

    lat0, lon0 = anchor
    T = make_transformer(lat0, lon0)

    # 2) compute each image top-left translation in pixels from GPS
    #    We’ll pretend each image’s center sits at its GPS lat/lon and place by translation.
    centers_xy = []
    for _, img, lat, lon in items:
        if lat is None:
            centers_xy.append(None)
        else:
            centers_xy.append(ll_to_xy(T, lat, lon))

    # choose first with GPS as origin
    ref_xy = next((xy for xy in centers_xy if xy is not None),
                  np.array([0, 0], np.float32))

    # 3) find output bounds by translating each image’s corners
    corners_world = []
    for (path, img, lat, lon), xy in zip(items, centers_xy):
        h, w = img.shape[:2]
        if xy is None:
            # skip images with no GPS; (you can add a neighbor refinement later)
            continue
        delta_m = xy - ref_xy
        # meters -> pixels; note image y-down, north is +y meters
        tx = -delta_m[0] / gsd_m_per_px
        ty = delta_m[1] / gsd_m_per_px
        # place so that image center goes to (tx, ty)
        H = np.array([[1, 0, tx - w/2],
                      [0, 1, ty - h/2],
                      [0, 0, 1]], np.float32)
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]],
                       np.float32).reshape(-1, 1, 2)
        pts_w = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        corners_world.append(pts_w)

    corners = np.vstack(corners_world)
    min_x, min_y = np.floor(corners.min(0)).astype(int)
    max_x, max_y = np.ceil(corners.max(0)).astype(int)

    shift = np.array([[1, 0, -min_x],
                      [0, 1, -min_y],
                      [0, 0, 1]], np.float32)
    out_w, out_h = int(max_x-min_x), int(max_y-min_y)

    # 4) feather-blend into canvas
    pano = np.zeros((out_h, out_w, 3), np.float32)
    acc = np.zeros((out_h, out_w), np.float32)

    for (path, img, lat, lon), xy in zip(items, centers_xy):
        if xy is None:
            continue
        h, w = img.shape[:2]
        delta_m = xy - ref_xy
        tx = -delta_m[0] / gsd_m_per_px
        ty = delta_m[1] / gsd_m_per_px
        H = shift @ np.array([[1, 0, tx - w/2],
                              [0, 1, ty - h/2],
                              [0, 0, 1]], np.float32)
        warped = cv2.warpPerspective(img, H, (out_w, out_h))
        m = (warped.sum(axis=2) > 0).astype(np.float32)
        pano = (pano * acc[..., None] + warped.astype(np.float32) *
                m[..., None]) / np.clip(acc[..., None]+m[..., None], 1e-6, None)
        acc += m

    out = np.clip(pano, 0, 255).astype(np.uint8)
    cv2.imwrite(output, out)
    print(f"Saved {output} ({out_w}x{out_h}), GSD={gsd_m_per_px} m/px")


mosaic_gps(folder="images/", gsd_m_per_px=0.03, output="mosaic_gps.jpg")
