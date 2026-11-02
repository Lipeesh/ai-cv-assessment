#!/usr/bin/env python3
"""
Task 1: Object Detection, Tracking, and Zone Monitoring
AjnaLens AI/ML & Computer Vision Engineer Assessment

Features:
- People Detection with YOLO11m
- Stable ID tracking using ByteTrack
- Restricted Zone entry detection using bottom-centre foot points
- Camera Motion Compensation: Keeps restricted zone anchored to the physical scene
- Transparent bounding boxes (borders only, no solid fills)
- Explicit Helmet and No Helmet detections and visualizations
- Footstep tracking trails rendering the path of each detected person
- Idle detection (> 5s with movement < threshold)
- PPE Monitoring (Helmet / Unhelmet association)
- Event generation (ZONE_ENTRY, IDLE_ALERT, PPE_VIOLATION) to events.json
- Annotated video output preserving resolution and aspect ratio
"""

import os
import sys
import cv2
import yaml
import json
import time
import logging
import platform
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import torch
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ZoneMonitor")

def get_hardware_info():
    """Detect and return CPU and GPU hardware specifications."""
    info = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    else:
        info["gpu_name"] = "CPU only"
    return info

def is_point_in_polygon(point, polygon_pts):
    """
    Check if a point (x, y) is inside the polygon using cv2.pointPolygonTest.
    Returns True if inside or on boundary.
    """
    pt = (float(point[0]), float(point[1]))
    return cv2.pointPolygonTest(polygon_pts, pt, False) >= 0

def compute_box_iou(box1, box2):
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

class ZoneMonitor:
    def __init__(self, config_path=None, video_path=None):
        if config_path is None:
            script_dir = Path(__file__).resolve().parent
            candidates = [
                Path("task1/config.yaml"),
                script_dir.parent / "config.yaml",
                Path("/home/ub/Desktop/COMBINATION/task1/config.yaml")
            ]
            for c in candidates:
                if c.exists():
                    config_path = str(c)
                    break
            if config_path is None:
                config_path = "task1/config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.people_model_path = self.config["people_model"]
        self.helmet_model_path = self.config["helmet_model"]
        self.input_video_path = video_path if video_path else self.config["input_video"]
        self.output_video_path = self.config["output_video"]
        self.events_json_path = self.config["events_json"]
        
        # Ensure output directories exist
        Path(self.output_video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.events_json_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Hardware
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.hw_info = get_hardware_info()
        logger.info(f"Hardware: {self.hw_info['gpu_name']} | CUDA: {self.hw_info['cuda_available']}")
        
        # Load Models
        logger.info(f"Loading People Model: {self.people_model_path}")
        self.people_model = YOLO(self.people_model_path)
        logger.info(f"People classes: {self.people_model.names}")
        
        logger.info(f"Loading Helmet Model: {self.helmet_model_path}")
        self.helmet_model = YOLO(self.helmet_model_path)
        logger.info(f"Helmet classes: {self.helmet_model.names}")
        
        # Map class names
        self.helmet_names = self.helmet_model.names
        self.helmet_cls_id = None
        self.unhelmet_cls_id = None
        for cid, name in self.helmet_names.items():
            name_lower = str(name).lower()
            if "unhelmet" in name_lower or "no_helmet" in name_lower or "head" in name_lower:
                self.unhelmet_cls_id = cid
            elif "helmet" in name_lower or "hard_hat" in name_lower:
                self.helmet_cls_id = cid
                
        logger.info(f"Class mapping: Helmet ID={self.helmet_cls_id}, Unhelmet ID={self.unhelmet_cls_id}")
        
        # Restricted Zone polygon
        zone_coords = self.config["restricted_zone"]
        self.initial_zone_polygon = np.array(zone_coords, dtype=np.float32)
        self.curr_zone_polygon = self.initial_zone_polygon.copy()
        self.anchor_zone = self.config.get("anchor_zone_to_scene", True)
        self.prev_gray = None
        
        # Tracking & State history
        self.in_zone_state = {}          # track_id -> bool
        self.foot_history = defaultdict(deque)  # track_id -> deque of (timestamp, x, y)
        self.footstep_trails = defaultdict(lambda: deque(maxlen=45))  # track_id -> list of foot points
        self.idle_alerted = {}          # track_id -> bool
        self.ppe_status = {}            # track_id -> "COMPLIANT" | "VIOLATION" | "UNKNOWN"
        self.ppe_violation_count = defaultdict(int) # track_id -> consecutive violation frames
        self.ppe_alerted = {}           # track_id -> bool
        
        # Output Events
        self.events = []

    def update_zone_tracking(self, frame_gray):
        """
        Camera Motion Compensation (CMC):
        Estimate background affine motion to keep the restricted zone polygon
        anchored to the physical scene as the camera moves or pans.
        """
        if not self.anchor_zone or self.prev_gray is None:
            self.prev_gray = frame_gray
            return

        p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=250, qualityLevel=0.01, minDistance=10)
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, p0, None)
            valid0 = p0[st == 1]
            valid1 = p1[st == 1]
            if len(valid0) >= 12:
                M, inliers = cv2.estimateAffinePartial2D(valid0, valid1)
                if M is not None:
                    pts = np.hstack([self.curr_zone_polygon, np.ones((len(self.curr_zone_polygon), 1), dtype=np.float32)])
                    new_pts = (M @ pts.T).T
                    # Boundary sanity check
                    h, w = frame_gray.shape[:2]
                    if -w < np.mean(new_pts[:, 0]) < 2 * w and -h < np.mean(new_pts[:, 1]) < 2 * h:
                        self.curr_zone_polygon = new_pts

        self.prev_gray = frame_gray

    def check_zone_entry(self, track_id, foot_pt, box, timestamp, frame_num):
        """
        Check if bottom-centre foot point enters restricted zone.
        Raises ZONE_ENTRY once per entry. Only re-triggers if exited and re-entered.
        """
        poly_int = self.curr_zone_polygon.astype(np.int32).reshape((-1, 1, 2))
        currently_in_zone = is_point_in_polygon(foot_pt, poly_int)
        previously_in_zone = self.in_zone_state.get(track_id, False)
        
        if currently_in_zone and not previously_in_zone:
            event = {
                "event_type": "ZONE_ENTRY",
                "track_id": int(track_id),
                "timestamp": round(float(timestamp), 2),
                "frame_number": int(frame_num),
                "bounding_box": [int(b) for b in box]
            }
            self.events.append(event)
            logger.info(f"Event: ZONE_ENTRY | Track: {track_id} | Time: {timestamp:.2f}s | Box: {box}")
            
        self.in_zone_state[track_id] = currently_in_zone
        return currently_in_zone

    def check_idle(self, track_id, foot_pt, box, timestamp, frame_num):
        """
        Detect idle person when movement is below threshold over 5 seconds.
        """
        if not self.config["idle"]["enabled"]:
            return False
            
        threshold_sec = float(self.config["idle"]["threshold_seconds"])
        threshold_px = float(self.config["idle"]["movement_threshold_pixels"])
        
        hist = self.foot_history[track_id]
        hist.append((timestamp, foot_pt[0], foot_pt[1]))
        
        while hist and (timestamp - hist[0][0]) > threshold_sec:
            hist.popleft()
            
        is_idle = False
        if len(hist) >= 2 and (hist[-1][0] - hist[0][0]) >= (threshold_sec * 0.9):
            anchor_x, anchor_y = hist[0][1], hist[0][2]
            max_disp = max(np.sqrt((x - anchor_x)**2 + (y - anchor_y)**2) for _, x, y in hist)
            
            if max_disp < threshold_px:
                is_idle = True
                if not self.idle_alerted.get(track_id, False):
                    event = {
                        "event_type": "IDLE_ALERT",
                        "track_id": int(track_id),
                        "timestamp": round(float(timestamp), 2),
                        "frame_number": int(frame_num),
                        "bounding_box": [int(b) for b in box]
                    }
                    self.events.append(event)
                    self.idle_alerted[track_id] = True
                    logger.info(f"Event: IDLE_ALERT | Track: {track_id} | Time: {timestamp:.2f}s | Disp: {max_disp:.1f}px")
            else:
                self.idle_alerted[track_id] = False
        return is_idle

    def associate_ppe(self, person_box, helmet_boxes, helmet_classes):
        """
        Associate helmet/unhelmet detections with a person.
        Focuses on the head/upper body region (top 45% of person box).
        Returns:
            match_type: "HELMET" | "NO_HELMET" | None
            match_box: [hx1, hy1, hx2, hy2] or None
        """
        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1
        
        # Upper body / head region
        head_region = [px1 - 0.15 * pw, py1 - 0.15 * ph, px2 + 0.15 * pw, py1 + 0.45 * ph]
        
        best_match_cls = None
        best_match_box = None
        max_overlap = 0.0
        
        for h_box, h_cls in zip(helmet_boxes, helmet_classes):
            hx1, hy1, hx2, hy2 = h_box
            hcx = (hx1 + hx2) / 2.0
            hcy = (hy1 + hy2) / 2.0
            
            # Check if helmet center is within expanded head region
            if (head_region[0] <= hcx <= head_region[2]) and (head_region[1] <= hcy <= head_region[3]):
                iou = compute_box_iou(head_region, h_box)
                if iou >= max_overlap:
                    max_overlap = iou
                    best_match_cls = h_cls
                    best_match_box = h_box
                    
        if best_match_cls == self.helmet_cls_id:
            return "HELMET", best_match_box
        elif best_match_cls == self.unhelmet_cls_id:
            return "NO_HELMET", best_match_box
        return None, None

    def check_ppe_violation(self, track_id, person_box, helmet_boxes, helmet_classes, timestamp, frame_num):
        """
        Determine PPE status and raise PPE_VIOLATION if person is without helmet.
        """
        if not self.config["ppe"]["enabled"]:
            return "UNKNOWN", None
            
        confirmation_frames = int(self.config["ppe"]["violation_confirmation_frames"])
        detected_status, match_box = self.associate_ppe(person_box, helmet_boxes, helmet_classes)
        
        status = "COMPLIANT"
        if detected_status == "NO_HELMET" or detected_status is None:
            self.ppe_violation_count[track_id] += 1
            if self.ppe_violation_count[track_id] >= confirmation_frames:
                status = "VIOLATION"
                if not self.ppe_alerted.get(track_id, False):
                    event = {
                        "event_type": "PPE_VIOLATION",
                        "track_id": int(track_id),
                        "timestamp": round(float(timestamp), 2),
                        "frame_number": int(frame_num),
                        "bounding_box": [int(b) for b in person_box]
                    }
                    self.events.append(event)
                    self.ppe_alerted[track_id] = True
                    logger.info(f"Event: PPE_VIOLATION | Track: {track_id} | Time: {timestamp:.2f}s | Box: {person_box}")
        else:
            self.ppe_violation_count[track_id] = 0
            self.ppe_alerted[track_id] = False
            status = "COMPLIANT"
            
        self.ppe_status[track_id] = status
        return status, match_box

    def draw_ui(self, frame, tracks_data, helmet_detections, active_events_summary, fps_val):
        """
        Draw professional UI annotations:
        - Transparent bounding boxes (only clean border lines visible)
        - Restricted zone outline anchored to the physical scene (no fill)
        - Footstep trails and footstep markers for every tracked person
        - Visual detection boxes for both Helmet (Green) and No Helmet (Red)
        - Top event HUD banner
        """
        h, w = frame.shape[:2]
        vis = frame.copy()
        
        # 1. Draw Restricted Zone (ONLY BORDER, transparent inside)
        if self.config.get("show_zone", True):
            poly_int = self.curr_zone_polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [poly_int], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            for pt in poly_int:
                cv2.circle(vis, (pt[0][0], pt[0][1]), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(vis, "RESTRICTED ZONE", (int(poly_int[0][0][0]) + 10, int(poly_int[0][0][1]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # 2. Draw Footstep Trails
        for item in tracks_data:
            tid = item["track_id"]
            trail = list(self.footstep_trails[tid])
            if len(trail) > 1:
                in_z = item["in_zone"]
                ppe = item["ppe"]
                base_color = (0, 0, 255) if (in_z or ppe == "VIOLATION") else (0, 255, 0)
                
                for i in range(1, len(trail)):
                    p1 = trail[i - 1]
                    p2 = trail[i]
                    # Connecting trail line
                    cv2.line(vis, p1, p2, base_color, 2, cv2.LINE_AA)
                    # Footstep dots every 2nd step
                    if i % 2 == 0:
                        cv2.circle(vis, p2, 3, (0, 255, 255), -1, cv2.LINE_AA)

        # 3. Draw Tracked People (TRANSPARENT BOXES - BORDERS ONLY)
        for item in tracks_data:
            tid = item["track_id"]
            box = item["box"]
            foot = item["foot"]
            in_zone = item["in_zone"]
            is_idle = item["is_idle"]
            ppe = item["ppe"]
            
            x1, y1, x2, y2 = box
            
            # Determine person box border color
            if ppe == "VIOLATION" or in_zone:
                box_color = (0, 0, 255) # Red for violation / zone entry
            elif is_idle:
                box_color = (0, 255, 255) # Yellow for idle
            else:
                box_color = (0, 255, 0) # Green for compliant
                
            # Draw ONLY the border outline (completely transparent interior)
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
            
            # Footstep current position marker (dual circle)
            fx, fy = int(foot[0]), int(foot[1])
            cv2.circle(vis, (fx, fy), 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, (fx, fy), 7, box_color, 2, cv2.LINE_AA)
            
            # Person status label (Clean text with shadow, NO solid rectangle background)
            if self.config.get("show_labels", True):
                status_txt = "HELMET" if ppe == "COMPLIANT" else "NO HELMET"
                label_txt = f"ID:{tid} | {status_txt}"
                if in_zone:
                    label_txt += " | ZONE"
                if is_idle:
                    label_txt += " | IDLE"
                    
                # Black outline + colored text for perfect contrast without blocking video
                cv2.putText(vis, label_txt, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, label_txt, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)

        # 4. Draw Helmet / No Helmet Detections (TRANSPARENT BOXES - BORDER ONLY)
        for h_det in helmet_detections:
            hx1, hy1, hx2, hy2 = [int(v) for v in h_det["box"]]
            h_cls = h_det["cls"]
            h_conf = h_det["conf"]
            
            if h_cls == self.helmet_cls_id:
                h_color = (0, 255, 0) # Green for Helmet
                h_label = f"HELMET {h_conf:.2f}"
            else:
                h_color = (0, 0, 255) # Red for No Helmet
                h_label = f"NO HELMET {h_conf:.2f}"
                
            # Transparent box (only border)
            cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), h_color, 2, cv2.LINE_AA)
            cv2.putText(vis, h_label, (hx1, max(12, hy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, h_label, (hx1, max(12, hy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, h_color, 1, cv2.LINE_AA)

        # 5. Draw Top Event HUD Banner
        if self.config.get("show_event_banner", True):
            banner_h = 42
            banner_overlay = vis.copy()
            cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
            cv2.addWeighted(banner_overlay, 0.65, vis, 0.35, 0, vis)
            
            hud_text = (
                f"FPS: {fps_val:.1f} | Active Tracks: {len(tracks_data)} | "
                f"Zone Entries: {active_events_summary['ZONE_ENTRY']} | "
                f"PPE Violations: {active_events_summary['PPE_VIOLATION']} | "
                f"Idle Alerts: {active_events_summary['IDLE_ALERT']}"
            )
            cv2.putText(vis, hud_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            
        return vis

    def process_video(self):
        """Main processing pipeline for Task 1."""
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {self.input_video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video input: {width}x{height} @ {fps:.1f} FPS | Total frames: {total_frames}")
        
        # Prepare Video Writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        start_time = time.time()
        fps_tracker = deque(maxlen=30)
        
        people_conf = float(self.config.get("people_confidence", 0.45))
        helmet_conf = float(self.config.get("helmet_confidence", 0.40))
        tracker_cfg = self.config.get("tracker", "bytetrack.yaml")
        
        active_summary = {"ZONE_ENTRY": 0, "PPE_VIOLATION": 0, "IDLE_ALERT": 0}
        
        logger.info("Starting frame processing...")
        while cap.isOpened():
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            timestamp = frame_idx / fps
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Camera Motion Compensation for Restricted Zone
            self.update_zone_tracking(frame_gray)
            
            # 2. Run People Detection & ByteTrack Tracking
            track_res = self.people_model.track(
                source=frame,
                persist=True,
                tracker=tracker_cfg,
                conf=people_conf,
                device=self.device,
                verbose=False
            )
            
            # 3. Run Helmet Detection on frame
            helmet_res = self.helmet_model.predict(
                source=frame,
                conf=helmet_conf,
                device=self.device,
                verbose=False
            )
            
            # Extract helmet detections
            helmet_boxes = []
            helmet_classes = []
            helmet_detections = []
            if len(helmet_res) > 0 and helmet_res[0].boxes is not None:
                h_b = helmet_res[0].boxes
                helmet_boxes = h_b.xyxy.cpu().numpy()
                helmet_classes = h_b.cls.cpu().numpy().astype(int)
                helmet_confs = h_b.conf.cpu().numpy()
                for h_box, h_cls, h_conf in zip(helmet_boxes, helmet_classes, helmet_confs):
                    helmet_detections.append({
                        "box": h_box,
                        "cls": int(h_cls),
                        "conf": float(h_conf)
                    })
                
            # 4. Process Tracked People & Footsteps
            tracks_data = []
            if len(track_res) > 0 and track_res[0].boxes is not None and track_res[0].boxes.id is not None:
                p_boxes = track_res[0].boxes.xyxy.cpu().numpy()
                p_ids = track_res[0].boxes.id.cpu().numpy().astype(int)
                
                for tid, pbox in zip(p_ids, p_boxes):
                    x1, y1, x2, y2 = [int(v) for v in pbox]
                    foot_pt = ((x1 + x2) / 2.0, float(y2))
                    
                    # Record footsteps trajectory
                    self.footstep_trails[tid].append((int(foot_pt[0]), int(foot_pt[1])))
                    
                    # Zone Entry check (against scene-anchored polygon)
                    in_zone = self.check_zone_entry(tid, foot_pt, [x1, y1, x2, y2], timestamp, frame_idx)
                    
                    # Idle check
                    is_idle = self.check_idle(tid, foot_pt, [x1, y1, x2, y2], timestamp, frame_idx)
                    
                    # PPE check
                    ppe_stat, _ = self.check_ppe_violation(tid, [x1, y1, x2, y2], helmet_boxes, helmet_classes, timestamp, frame_idx)
                    
                    tracks_data.append({
                        "track_id": tid,
                        "box": [x1, y1, x2, y2],
                        "foot": foot_pt,
                        "in_zone": in_zone,
                        "is_idle": is_idle,
                        "ppe": ppe_stat
                    })
                    
            # Update counts
            active_summary["ZONE_ENTRY"] = sum(1 for e in self.events if e["event_type"] == "ZONE_ENTRY")
            active_summary["PPE_VIOLATION"] = sum(1 for e in self.events if e["event_type"] == "PPE_VIOLATION")
            active_summary["IDLE_ALERT"] = sum(1 for e in self.events if e["event_type"] == "IDLE_ALERT")
            
            # Compute FPS
            frame_elapsed = time.time() - t_frame_start
            current_fps = 1.0 / max(frame_elapsed, 1e-4)
            fps_tracker.append(current_fps)
            avg_fps = sum(fps_tracker) / len(fps_tracker)
            
            # 5. Draw Annotations with footsteps & transparent boxes
            annotated_frame = self.draw_ui(frame, tracks_data, helmet_detections, active_summary, avg_fps)
            writer.write(annotated_frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                logger.info(f"Processed frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | FPS: {avg_fps:.1f}")
                
        cap.release()
        writer.release()
        
        total_time = time.time() - start_time
        final_fps = frame_idx / max(total_time, 1e-4)
        
        # Save events.json
        with open(self.events_json_path, "w") as f:
            json.dump(self.events, f, indent=4)
            
        logger.info("=" * 60)
        logger.info("TASK 1 COMPLETED SUCCESSFULLY")
        logger.info(f"Total Frames Processed: {frame_idx}")
        logger.info(f"Total Processing Time: {total_time:.2f} seconds")
        logger.info(f"Average Processing Speed: {final_fps:.2f} FPS")
        logger.info(f"Total Events Recorded: {len(self.events)}")
        logger.info(f"  - ZONE_ENTRY: {active_summary['ZONE_ENTRY']}")
        logger.info(f"  - PPE_VIOLATION: {active_summary['PPE_VIOLATION']}")
        logger.info(f"  - IDLE_ALERT: {active_summary['IDLE_ALERT']}")
        logger.info(f"Output Video saved to: {self.output_video_path}")
        logger.info(f"Events JSON saved to: {self.events_json_path}")
        logger.info("=" * 60)
        
        return {
            "total_frames": frame_idx,
            "total_time_sec": total_time,
            "average_fps": final_fps,
            "total_events": len(self.events),
            "events_breakdown": active_summary,
            "hardware": self.hw_info
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 1: Zone Monitoring & PPE Detection")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: auto-detected task1/config.yaml)")
    parser.add_argument("--video", type=str, default=None, help="Path to input test video (overrides config.yaml)")
    args = parser.parse_args()

    monitor = ZoneMonitor(config_path=args.config, video_path=args.video)
    monitor.process_video()
