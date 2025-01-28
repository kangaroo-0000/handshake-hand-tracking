# Handshake Hand Tracking with SAM 2 HW

This project demonstrates how to:
1. Detect hands in video frames with **MediaPipe**.
2. Use bounding boxes as prompts for **SAM 2** to segment hands.
3. Produce a video with per-frame masked hands.

## Setup

1. Clone this repository.
2. Install dependencies:
   ```bash
   conda create -n sam2 python=3.12
   conda activate sam2
   pip install "git+https://github.com/facebookresearch/sam2.git"
   pip install -r requirements.txt
3. Run ```sh download_model.sh``` to download required .pt file for inference
4. Run ```python main.py``` and look at results in test_output.mp4
