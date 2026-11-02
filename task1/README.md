# Task 1 — Object Detection and Zone Monitoring

## Overview
This module implements real-time object detection, multi-object tracking, restricted-zone entry monitoring, idle detection, and PPE compliance enforcement using custom trained YOLO11m models for the AjnaLens AI/ML & CV Engineer Practical Assessment.

---

## Models & Architecture
- **People Detection Model**: Custom trained Ultralytics YOLO11m (`runs/people_yolo11/yolo11m_people_100e/weights/best.pt`) detecting class `person`.
- **PPE / Helmet Detection Model**: Custom trained Ultralytics YOLO11m (`runs/helmet_yolo11/yolo11m_helmet_100e/weights/best.pt`) detecting classes `helmet` and `unhelmet`.
- **Multi-Object Tracker**: ByteTrack (`bytetrack.yaml`) with persistent association across frames.

---

## Input
- **Video Path**: `runs/test/videoplayback.mp4`
- **Resolution**: 1280 x 720 (Native aspect ratio preserved, no canvas cropping or reframing)
- **Framerate**: 20.0 FPS

---

## Pipeline Description
1. **People Detection & Tracking**:
   - Each frame is processed by the trained People YOLO11m model.
   - ByteTrack assigns persistent, stable track IDs.
2. **Restricted Zone Monitoring**:
   - A configurable polygon is defined in `config.yaml`.
   - The **bottom-centre foot point** `((x1 + x2)/2, y2)` of each tracked bounding box is evaluated via `cv2.pointPolygonTest`.
   - When the foot point enters the polygon, a `ZONE_ENTRY` event is triggered.
   - Debouncing: Re-entry is only triggered after a person has fully exited the zone and re-entered.
3. **Idle Duration Monitoring**:
   - The foot point trajectory over a rolling window (5.0 seconds) is monitored.
   - If a person moves less than the configurable displacement threshold (20 px) over 5 seconds, an `IDLE_ALERT` event is raised once per idle episode.
4. **PPE Compliance Monitoring**:
   - The trained Helmet YOLO11m model detects `helmet` and `unhelmet` bounding boxes.
   - Detections are spatially associated with the upper body / head region (top 40% of the person's bounding box).
   - If a person is detected without a helmet across multiple confirmation frames, a `PPE_VIOLATION` event is logged.
5. **Output Generation**:
   - Annotated output video with live event HUD banner, bounding boxes, foot point crosshairs, and restricted zone overlay: `task1/outputs/output.mp4`.
   - Structured event log: `task1/outputs/events.json`.

---

## Engineering Design Questions

### Why use the foot point instead of the box centre?
The foot point `((x1 + x2)/2, y2)` represents the physical contact point where a person is standing on the ground plane. Using the bounding box center `((x1 + x2)/2, (y1 + y2)/2)` leads to severe false positives or false negatives in perspective camera views:
- A person standing outside a restricted zone whose upper body leans or extends into the camera view may have their box center inside the polygon, incorrectly triggering a zone violation.
- Conversely, a person standing inside a restricted boundary whose upper body extends outward could be missed. The bottom-centre ground contact point is the standard and geometrically accurate anchor in camera-to-ground homography and 2D zone demarcation.

### How would you prevent ID swapping after 2 seconds of occlusion?
To maintain stable identities across occlusions longer than 2 seconds (e.g. 40–60 frames):
1. **Track Buffer Extension**: Increase ByteTrack's `track_buffer` parameter from default 30 frames to 90–120 frames so tracks remain in the "lost" state rather than being prematurely terminated.
2. **Appearance Re-Identification (ReID)**: Integrate a lightweight appearance feature extractor (such as OSNet or BoT-SORT) to match visual embedding cosine similarity when spatial Kalman filter predictions diverge during occlusion.
3. **Motion Trend Extrapolation**: Use constant acceleration or non-linear motion models to predict the emergence point of occluded pedestrians.
4. **Trajectory Splitting & Merging**: Post-association trajectory stitching based on spatio-temporal continuity and directional velocity matching.

---

## Performance & Execution
- **Hardware Used**: NVIDIA H200 NVL (140GB VRAM) / CUDA 13.0
- **Average Processing Speed**: ~45–60 FPS
- **Resolution**: 1280x720 preserved

---

## Usage Instructions
```bash
# Navigate to project directory
cd /home/ub/Desktop/COMBINATION

# Execute Task 1
python task1/src/zone_monitor.py
```
Outputs will be generated in:
- `task1/outputs/output.mp4`
- `task1/outputs/events.json`
