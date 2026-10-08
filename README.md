# Tennis Court Tracking (PIV Project)

This repository contains the implementation for tracking tennis court lines across a video sequence using homography estimation, part of the PIV course project.

## Requirements

Ensure you have installed the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup & Data Preparation

Due to the size of the model weights and datasets, they are not included in this repository. You must set them up manually before running the tracking algorithms.

### 1. Download the Dataset
You can download the rally video datasets and their associated data from this [Google Drive folder](https://drive.google.com/drive/folders/1_FGsEY-lHNNxEBOCPWyPXaoLSU2AD__X?usp=sharing). Once downloaded, place the rallies in your desired local directory (e.g., `datasets/rally_04/`, etc.).

### 2. Download and Setup TennisCourtDetector
This project relies on the [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) to extract the initial reference court keypoints (`court_base_<name>.mat`) from the first frame of the rally.

1. Clone the `TennisCourtDetector` repository inside this project directory (or elsewhere):
   ```bash
   git clone https://github.com/yastrebksv/TennisCourtDetector.git TennisCourtDetector
   ```
2. Download the pre-trained PyTorch weights as specified in their repository and place them in the correct folder (e.g., `TennisCourtDetector/models/`).
3. Follow their installation instructions (installing `fastai`, `torch`, etc.).

### 3. Extract Reference Images and Keypoints
For each rally you want to evaluate, you need to extract the first frame (as `templateimg.jpg`) and compute the initial 14 court keypoints (`court_base_templateimg.mat`).

1. Extract the first frame of the rally and save it. For instance, save it as `templateimg.jpg`.
2. Run the `TennisCourtDetector` inference on that image to extract the keypoints:
   ```bash
   cd TennisCourtDetector
   python infer_in_image.py --image_path ../templateimg.jpg --use_homography --use_refine_kps
   ```
3. This will generate a file named `court_base_templateimg.mat`. 
4. Place **both** `templateimg.jpg` and `court_base_templateimg.mat` into your reference directory (e.g., `ref_tennis/` or a rally-specific directory). The evaluation scripts will look for `court_base_*.mat` in either the `images_dir` or the `ref_dir`.

> **Note:** The universal physical dimensions of the ITF tennis court are provided in `ref_tennis/courtmodel.mat`. This file is already tracked in the repository and you do not need to generate it.

## Running the Tracking Pipeline

Once the data is prepared, you can run the evaluation script:

```bash
python main1.py <path_to_refdir> <path_images_dir> <path_feature_dir> <path_output_dir>
```

Example:
```bash
python main1.py ref_tennis/ datasets/rally_04/images/ features/rally_04/ results/rally_04/
```
