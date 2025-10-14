import cv2
import numpy as np
import exifread
from PIL import Image
from pyproj import Transformer


def extract_gps(image_path):
    """Extract (lat, lon) from EXIF metadata."""
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f)
    try:
        lat_ref = str(tags["GPS GPSLatitudeRef"])
        lon_ref = str(tags["GPS GPSLongitudeRef"])
        lat = tags["GPS GPSLatitude"].values
        lon = tags["GPS GPSLongitude"].values

        # Convert from degrees/minutes/seconds to decimal
        def dms_to_dd(dms, ref):
            d = float(dms[0].num) / float(dms[0].den)
            m = float(dms[1].num) / float(dms[1].den)
            s = float(dms[2].num) / float(dms[2].den)
            dd = d + m / 60.0 + s / 3600.0
            if ref in ["S", "W"]:
                dd *= -1
            return dd

        return dms_to_dd(lat, lat_ref), dms_to_dd(lon, lon_ref)
    except Exception:
        return None, None


def gps_to_xy(lat, lon, epsg_out="EPSG:32617"):  # UTM Zone 17N for Columbus, OH
    """Convert lat/lon to projected (x, y) coordinates (e.g. UTM)."""
    transformer = Transformer.from_crs("EPSG:4326", epsg_out, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def gps_assisted_stitch(image_paths, canvas_scale=0.1):
    """
    GPS-assisted image stitching.
    - image_paths: list of file paths
    - canvas_scale: meters per pixel (rough scale)
    """
    gps_coords = []
    imgs = []

    # Load images + GPS
    for path in image_paths:
        img = cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_RGB2BGR)
        lat, lon = extract_gps(path)
        if lat is None or lon is None:
            raise ValueError(f"No GPS data found in {path}")
        x, y = gps_to_xy(lat, lon)
        gps_coords.append((x, y))
        imgs.append(img)

    # Normalize GPS coords to start at (0,0)
    min_x = min([c[0] for c in gps_coords])
    min_y = min([c[1] for c in gps_coords])
    gps_coords = [(x - min_x, y - min_y) for (x, y) in gps_coords]

    # Estimate canvas size
    max_x = max([c[0] for c in gps_coords])
    max_y = max([c[1] for c in gps_coords])
    canvas_w = int(max_x / canvas_scale) + 2000
    canvas_h = int(max_y / canvas_scale) + 2000
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place images on canvas based on GPS
    for (img, (x, y)) in zip(imgs, gps_coords):
        h, w = img.shape[:2]
        cx = int(x / canvas_scale)
        cy = int(y / canvas_scale)

        # Rough placement (centered)
        x0 = cx - w // 2
        y0 = cy - h // 2
        x1, y1 = x0 + w, y0 + h

        # Place on canvas (simple overwrite)
        canvas[y0:y1, x0:x1] = img

    # Refine stitching with OpenCV (optional)
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(imgs)

    if status == cv2.Stitcher_OK:
        return stitched
    else:
        return canvas  # fallback to GPS placement


# --- Example usage ---
if __name__ == "__main__":
    images = ["drone1.jpg", "drone2.jpg", "drone3.jpg"]
    result = gps_assisted_stitch(images)
    cv2.imwrite("stitched_output.jpg", result)






#______________________________________________________________ node here






#!/usr/bin/env python3
# ----------------------------------------------------
# Image Stitcher Node (ROS2 + OpenCV in Python)
# ----------------------------------------------------
# - Subscribes to camera images (sensor_msgs/Image)
# - Subscribes to mission state (String)
# - Subscribes to GPS (NavSatFix)
# - Stores images with poses
# - Mosaics using pose-based placement + local overlap refinement
#   (ECC rigid tweak with ORB+RANSAC fallback)
# - Saves the stitched result periodically
# ----------------------------------------------------

import os
import time
import math
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, NavSatFix, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from mavros_msgs.msg import WaypointReached
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import exiftool
# =====================
# ROS2 Node
# =====================

class ImageStitcherNode(Node):
    def __init__(self):
        super().__init__('image_stitcher_node')

        # State variables
        self.transition_in_progress = False
        self.waypoint_current = 0
        self.waypoint_prev = -1
        self.latest_frame = None
        self.latest_gps = None
        self.state = "lap"
        self.reached = False
        self.pic_counter = 0

        # Parameters
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_path', 'field/images')
        self.declare_parameter('crop', True)
        self.declare_parameter('preprocessing', False)
        self.declare_parameter('stitch_interval_sec', 60.0)
        self.declare_parameter('minimum_stitch_distance', 7.0)
        os.makedirs("field/images", exist_ok=True)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value
        self.crop = self.get_parameter('crop').get_parameter_value().bool_value
        self.preprocessing = self.get_parameter('preprocessing').get_parameter_value().bool_value
        self.stitch_interval_sec = self.get_parameter('stitch_interval_sec').get_parameter_value().double_value
        self.minimum_stitch_distance = self.get_parameter('minimum_stitch_distance').get_parameter_value().double_value

        # Subscriptions
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.mission_state_sub = self.create_subscription(String, '/mission_state', self.state_callback, 10)
        self.reached_sub = self.create_subscription(WaypointReached, '/mavros/mission/reached', self.reached_callback, 10)

        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_callback, qos_profile=pose_qos)

        gps_sub_QoS = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        self.gps_sub = self.create_subscription(NavSatFix, '/mavros/global_position/global', self.gps_callback, gps_sub_QoS)

        self.stitch_timer = self.create_timer(self.stitch_interval_sec, self.timer_callback)

        # Helpers
        self.bridge = CvBridge()
        self.received_images = []  # stores (image, pose)

        self.get_logger().info(f"Subscribed to {image_topic}; stitch every {self.stitch_interval_sec}s")

    # -------------------------
    # Callbacks
    # -------------------------
    def rosimg_to_ndarray(self, msg: Image) -> np.ndarray:
        """Convert sensor_msgs/Image to H×W×C NumPy array (BGR8)."""
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return img

    def crop_image(self, img):
        """Square center crop (optional)."""
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        crop_w, crop_h = h, h
        x0 = max(0, cx - crop_w // 2)
        y0 = max(0, cy - crop_h // 2)
        x1 = min(w, x0 + crop_w)
        y1 = min(h, y0 + crop_h)
        return img[y0:y1, x0:x1]

    def deblur_image(self, img):
        # placeholder if you want to add a quick deblur/USM pass later
        return img

    def state_callback(self, msg: String):
        self.state = msg.data

    def reached_callback(self, msg: WaypointReached):
        if self.transition_in_progress:
            return
        idx = msg.wp_seq
        self.get_logger().info(f'Reached waypoint {idx} (state={self.state})')
        self.waypoint_current = idx
        if self.state == "stitching":
            self.reached = True
            self.get_logger().info(f'Reached waypoint in {self.state}')
        else:
            self.reached = False
            self.get_logger().info(f'Reached waypoint, NOT in stitching')

    def gps_callback(self, msg: NavSatFix):
        self.latest_gps = msg

    #THIS SHOULD INSTEAD BE CALLED WHEN THE DRONE IS DONE FLYING
    def timer_callback(self):
        command = "docker run -ti --rm -v .:/datasets opendronemap/odm --project-path /datasets field --skip-3dmodel --force-gps --min-num-features 20000"
        os.system(command)
        self.get_logger().info('callback enter')


    def image_callback(self, msg: Image):
        try:
            frame = self.rosimg_to_ndarray(msg)
            cropped_frame = self.crop_image(frame)
            self.latest_frame = cropped_frame
        except Exception as e:
            self.get_logger().error(f"Failed to convert raw image: {e}")

    def pose_callback(self, msg: PoseStamped):
        time = datetime.now()
        path = os.path.join(self.output_path, f"{time}.jpg")
        cv2.imwrite(path,self.latest_frame)

        lat = self.latest_gps.latitude
        lon = self.latest_gps.longitude
        with exiftool.ExifTool() as et:
            et.execute(
                b"-n",  # tells exiftool to interpret GPS numbers directly
                f"-EXIF:GPSLatitude={lat}".encode(),
                f"-EXIF:GPSLongitude={lon}".encode(),
                f"-EXIF:DateTimeOriginal={time}".encode(),
                b"-overwrite_original",
                path.encode("utf-8")
            )
        # time = datetime.now()
        # path = os.path.join(self.output_path, f"{time}.jpg")
        # cv2.imwrite(path,self.latest_frame)
        # Only append when we have an image to pair
# =====================

def main(args=None):
    rclpy.init(args=args)
    node = ImageStitcherNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

