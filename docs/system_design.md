# System Design Document — AjnaLens AI/CV Practical Assessment

## Executive Summary
This document outlines the end-to-end system architecture, engineering rationale, model choices, edge-to-cloud trade-offs, and production deployment roadmap for the 4-task visual intelligence and compliance platform.

![System Architecture](images/architecture_diagram.png)

---

## 1. System-Wide Architectural Overview

The platform is architected as an edge-accelerated, microservices-ready computer vision and knowledge assistant pipeline designed for high-throughput industrial monitoring (construction yards, power substations, heavy fabrication bays).

```mermaid
flowchart TD
    subgraph Video_Ingestion["Edge Video Ingestion"]
        CCTV["RTSP Industrial CCTV (1080p / 720p)"]
        FrameBuffer["Circular Ring Buffer / GStreamer"]
        CCTV --> FrameBuffer
    end

    subgraph Perception_Pipeline["Task 1 & Task 2: Visual Perception Engine"]
        YOLO_People["YOLO11m People Detector (CUDA / TensorRT)"]
        Tracker["ByteTrack Multi-Object Association"]
        MotionComp["Optical Flow Scene Stabilizer"]
        YOLO_Helmet["YOLO11m Helmet Detector (FP16 ONNX / TensorRT)"]
        ZoneLogic["Polygon In-Zone & Idle Dwell Logic"]
        
        FrameBuffer --> YOLO_People
        FrameBuffer --> MotionComp
        YOLO_People --> Tracker
        Tracker --> ZoneLogic
        MotionComp --> ZoneLogic
        FrameBuffer --> YOLO_Helmet
        YOLO_Helmet --> ZoneLogic
    end

    subgraph OCR_Engine["Task 3: Edge Equipment Asset OCR"]
        InspectCam["Mobile / Fixed Inspection Camera"]
        Preprocess["CLAHE + Bilateral Filter + Deskew"]
        Tess["Tesseract 5.3 LSTM Engine"]
        RegexParse["Regex Key-Value Attribute Extractor"]
        
        InspectCam --> Preprocess --> Tess --> RegexParse
    end

    subgraph RAG_Service["Task 4: Safety SOP RAG Service"]
        UserClient["Safety Officer Web / App Client"]
        FastAPI["FastAPI REST Microservice (:8088)"]
        MiniLM["SentenceTransformer (all-MiniLM-L6-v2)"]
        VectorDB["Cosine Similarity SOP Index"]
        Synthesizer["Safety Advisory Synthesizer"]
        
        UserClient -->|HTTP POST /query| FastAPI
        FastAPI --> MiniLM --> VectorDB --> Synthesizer --> FastAPI
    end

    ZoneLogic -->|Events JSON / Webhook| AlertHub["Kafka / MQTT Central Alert Hub"]
    RegexParse -->|Asset Metadata| AlertHub
    AlertHub --> Dashboard["Real-Time Security Dashboard"]
```

---

## 2. Task 1: Object Detection and Zone Monitoring

### Architecture & Data Flow
```mermaid
sequenceDiagram
    participant Cam as CCTV Frame
    participant People as YOLO11m (People)
    participant Track as ByteTrack
    participant Motion as Affine Motion Tracker
    participant Zone as Zone & Idle Evaluator
    participant Helmet as YOLO11m (PPE)
    participant Out as Output Stream & JSON

    Cam->>People: Ingest BGR frame (1280x720)
    People->>Track: Bounding boxes & confidences
    Track->>Zone: Persistent Track IDs + Footpoints
    Cam->>Motion: Track background keypoints
    Motion->>Zone: Compensate camera pan/tilt
    Cam->>Helmet: Detect helmets & bare heads
    Helmet->>Zone: Head-region spatial intersection
    Zone->>Out: Annotate HUD, transparent boxes, footprints
    Zone->>Out: Append events (ZONE_ENTRY, IDLE_ALERT, PPE_VIOLATION)
```

### Key Engineering Decisions
1. **Ground-Contact Foot-Point vs. Center Point**:
   - *Choice*: Bottom-centre coordinate `((x1 + x2)/2, y2)`.
   - *Rationale*: A worker standing outside a 2D hazard zone may lean forward, placing their bounding box center inside the zone polygon. The foot contact point represents the true intersection of the pedestrian with the 3D floor plane in surveillance perspective geometry, preventing spurious false alarms.
2. **Camera Motion Compensation**:
   - *Choice*: Affine transformation via Lucas-Kanade optical flow on non-pedestrian background keypoints.
   - *Rationale*: When CCTV cameras pan or jitter, fixed image coordinates shift relative to the real world. Compensating the virtual polygon anchors it to physical machinery regardless of camera movement.
3. **Transparent Bounding Boxes & Footstep Trajectory Trails**:
   - *Choice*: Outline borders only (thickness=2) with fading historical footpoint line trails.
   - *Rationale*: Preserves unobstructed visual inspection of the underlying scene while allowing operators to immediately trace a worker's path into the hazardous zone.

### Failure Cases & Mitigations
- **Severe Occlusion (> 3 seconds)**:
  - *Failure*: Trajectory fragmentation if worker walks behind large pillar.
  - *Mitigation*: Extended ByteTrack track buffer from 30 to 60 frames, supplemented with ReID feature vectors (OSNet).

---

## 3. Task 2: Detector Fine-Tuning, Optimization & Benchmarking

### Architecture & Pipeline
```mermaid
flowchart LR
    Dataset["Roboflow Helmet Dataset (1,093 images)"] --> Train["YOLO11m Fine-Tuning (100 Epochs)"]
    Train --> PyTorchWeights["best.pt (PyTorch FP32/AMP)"]
    PyTorchWeights --> ExportONNX["ONNX Export (Opset 17)"]
    ExportONNX --> FP32_ONNX["best.onnx (76.7 MB)"]
    ExportONNX --> FP16_ONNX["best_fp16.onnx (38.4 MB)"]
    
    subgraph Benchmark_Evaluation["Runtime Benchmarking (100 Val Images)"]
        PyTorchWeights -->|CUDA GPU| PT_Bench["PyTorch: 8.83 ms | 113.2 FPS"]
        FP32_ONNX -->|ONNX Runtime CPU| CPU_Bench["ONNX FP32: 94.6 ms | 10.6 FPS"]
        FP16_ONNX -->|ONNX Runtime CPU| FP16_Bench["ONNX FP16: 101.5 ms | 9.9 FPS"]
    end
```

### Empirical Metrics
- **mAP@50**: 71.6% (Overall) | **87.5%** (`helmet`) | **55.6%** (`unhelmet`)
- **PyTorch GPU Inference**: 8.83 ms latency (113.2 FPS)
- **ONNX Runtime CPU**: 94.64 ms latency (10.6 FPS)

### Class Imbalance Remediation
The dataset has 311 `helmet` instances vs only 14 `unhelmet` instances in the validation set.
- *Remediation 1*: Minority class oversampling and localized mosaic augmentations.
- *Remediation 2*: Loss reweighting in YOLO classification head ($\alpha_{\text{unhelmet}} = 2.5$) and focal loss modulation.

### Edge Device Deployment (Jetson Orin Nano / Mobile Target)
1. **TensorRT INT8 Quantization**: Use `trtexec` with calibration images to achieve 4-6x acceleration, executing in under 12 ms on Jetson Orin Nano.
2. **Architecture Downscaling**: Deploy `YOLO11s` or `YOLO11n` for tight wattage budgets (10W-15W).

---

## 4. Task 3: Equipment Nameplate OCR Pipeline

### Architecture & Processing Flow
```mermaid
flowchart TD
    Raw["Raw Nameplate Photo"] --> Gray["Grayscale Conversion"]
    Gray --> CLAHE["CLAHE Contrast Equalization"]
    CLAHE --> Bilat["Bilateral Filter (d=5, sigma=50)"]
    Bilat --> Deskew["Otsu Horizontal Deskewing"]
    Deskew --> OCR["Tesseract 5.3 LSTM Engine (PSM 6)"]
    OCR --> Regex["Multi-Regex Field Parser"]
    Regex --> StructuredOutput["JSON: {equipment_id, serial_number, model, voltage, manufacture_date}"]
```

### Preprocessing Trade-Offs
- **CLAHE + Bilateral Smoothing vs Aggressive Binarization**:
  - Global binarization degrades under uneven spotlight reflection. Bilateral smoothing preserves sharp text contours while eliminating brushed steel grain.
- **Empirical Finding**:
  - Unsharp masking on high-resolution plates can over-accentuate thin hyphens. Conditional preprocessing (activating filters only when image entropy/contrast is below threshold) is the optimal production strategy.

---

## 5. Task 4: Industrial Safety & Maintenance RAG Assistant

### Architecture & Data Flow
```mermaid
flowchart LR
    Query["User / Incident Query"] --> Embed["SentenceTransformer (all-MiniLM-L6-v2)"]
    Embed --> Vector["384-d Dense Query Vector"]
    Vector --> Sim["Cosine Similarity Matrix"]
    Corpus["Structured SOP Corpus (OSHA, LOTO, PPE, Voltages)"] --> DocEmbed["Precomputed Document Embeddings"]
    DocEmbed --> Sim
    Sim --> TopK["Top-K Relevant SOPs"]
    TopK --> Synth["Answer Synthesizer with OSHA Citations"]
    Synth --> Response["FastAPI JSON Response (< 10 ms)"]
```

### Design Choices
1. **Embedding Model (`all-MiniLM-L6-v2`)**:
   - Generates compact 384-dimensional embeddings in ~8 ms on CPU.
   - Ideal for edge gateways or local plant servers without GPU dependencies.
2. **Citation-Backed Synthesis**:
   - Industrial safety requires 100% auditable sources. The synthesizer explicitly links recommendations to document IDs (`SOP-PPE-001`, `SOP-LOTO-003`) and regulatory standards (OSHA 1926.100, OSHA 1910.147, NFPA 70E).

---

## 6. Production Deployment & Scaling Architecture

```mermaid
flowchart TD
    subgraph Edge_Tier["Edge Tier (Plant / Yard Level)"]
        IPC["Industrial PC / Jetson Orin"]
        DeepStream["NVIDIA DeepStream / GStreamer"]
        TRT["TensorRT YOLO Engine"]
        IPC --> DeepStream --> TRT
    end

    subgraph Message_Bus["Ingress / Streaming Bus"]
        Kafka["Apache Kafka / RabbitMQ Cluster"]
        TRT -->|Detections & Violations| Kafka
    end

    subgraph Cloud_Cluster["Enterprise Cloud Tier (Kubernetes)"]
        K8s["K8s Cluster"]
        RAG_Pod["RAG Service Pods (FastAPI + ChromaDB)"]
        Worker_Pod["Telemetry & Reporting Workers"]
        DB["PostgreSQL Event Store"]
        
        Kafka --> Worker_Pod
        Worker_Pod --> DB
        K8s --> RAG_Pod
    end

    subgraph Operator_UI["Control Room"]
        WebDash["React / Grafana Central Dashboard"]
        DB --> WebDash
        RAG_Pod --> WebDash
    end
```

### Production Hardening Checklist
1. **Containerization**: Package all services into multi-stage Alpine/Ubuntu Docker containers with pinned CUDA runtimes.
2. **Zero-Trust Telemetry**: TLS 1.3 encryption on RTSP video feeds and mutual TLS for inter-service gRPC communication.
3. **Horizontal Auto-Scaling**: Kubernetes Horizontal Pod Autoscaler (HPA) scaling RAG pods based on request concurrency.
