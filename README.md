# PIV Project 2026 - Part 1: Planar Tracking & Homography Estimation

Implementation of **Part 1** of the Computer Vision (PIV - *Processamento de Imagem e Visão*) project at Instituto Superior Técnico (IST).

---

## 📌 Project Overview (Part 1)

The objective is to build a robust homography estimation and tracking pipeline using only RGB data. For every frame $i$, the pipeline calculates two $3 \times 3$ homography matrices:
1. **$H$**: Maps frame $i$ onto the reference image (`templateimg.jpg`, typically the first frame) for video stabilization and drift-free tracking.
2. **$H_c$**: Maps frame $i$ onto the synthetic **court model** in metric units (meters) on the court plane ($x_{\text{court}} \sim H_c x_i$).

Outputs are stored as `homography_NNNN.mat` containing `{"H": H, "Hc": Hc}` for every frame in the sequence, including cuts, close-ups, and occluded frames.

---

## ⚖️ Strict Library & Architecture Constraints

As required by the assignment specification:
* **`main1.py`**:
  * May import `cv2`.
  * Must import `part1`.
  * Extracts SIFT features from all frames and saves them in `.mat` format matching `pivist/features`.
  * Calls `part1.part1(path_to_refdir, path_images_dir, path_feature_dir, path_output_dir)`.
* **`part1.py`**:
  * **CANNOT import `cv2`** (strictly enforced).
  * Uses only `numpy` and `scipy`.
  * Implements **Hartley-normalized Direct Linear Transformation (DLT)** from scratch.
  * Implements **RANSAC outlier rejection** from scratch (minimal 4-point sample, collinearity rejection, consensus scoring).
  * Implements SIFT feature matching via Lowe's ratio test and mutual consistency using `scipy.spatial.cKDTree`.
  * Implements hybrid tracking (direct template matching + sequential composition + cut/occlusion fallback).

---

## 📁 Repository Structure

```
.
├── main1.py                  # Part 1 runner: extracts SIFT features and invokes part1()
├── part1.py                  # Core algorithm: DLT, RANSAC, tracker, court mapping (NO cv2)
├── court_model.py            # Generates official ITF Tennis Court model (courtmodel.mat)
├── annotate_anchor.py        # Interactive tool to anchor templateimg.jpg to courtmodel.mat
├── extract_frames.py         # Utility to extract frame sequences from video (e.g. .mp4/.mov)
├── visualize_homography.py   # Overlay court wireframe onto frames & evaluate Appendix A metrics
├── tests/
│   └── test_part1.py         # Automated unit and integration tests
├── Projecet_v1.pdf           # Project assignment description
└── README.md
```

---

## 🚀 How to Run

### 1. Generate Court Model (`courtmodel.mat`)
Generate the official ITF tennis court geometry in meters:
```bash
python court_model.py --output data/ref/courtmodel.mat
```

### 2. Extract Frames from Video (Optional)
If starting from a video file:
```bash
python extract_frames.py path/to/video.mov --images_dir data/images --ref_dir data/ref --max_frames 150 --step 1
```
This extracts frames named `rally_NNNN.jpg` and writes `templateimg.jpg` into `data/ref/`.

### 3. Anchor Court Model (Interactive)
Associate landmarks between `templateimg.jpg` and `courtmodel.mat` to calculate the reference court anchor $H_{\text{ref}\to\text{court}}$:
```bash
python annotate_anchor.py data/ref
```
Click 4+ court corners/Ts on the window, press `c` to save `data/ref/anchor.mat`.

### 4. Run Part 1 Pipeline
Execute the official project command:
```bash
python main1.py path_to_refdir path_images_dir path_feature_dir path_output_dir
```
**Example:**
```bash
python main1.py data/ref data/images data/features data/output
```
This produces:
- SIFT feature `.mat` files in `data/features/` (`combined` array of shape `(130, N)`).
- `homography_NNNN.mat` in `data/output/` containing `{"H": H, "Hc": Hc}`.

---

## 🔍 Visual Verification & Evaluation (Appendix A)

To overlay the court wireframe on the original frames and measure temporal consistency:
```bash
python visualize_homography.py data/ref data/images data/output --save_video overlay.mp4
```

---

## 🧪 Running Tests

Run the test suite verifying Hartley DLT, RANSAC outlier rejection, SIFT matching, and adherence to library constraints:
```bash
python tests/test_part1.py
```
Or with pytest:
```bash
pytest tests/
```
