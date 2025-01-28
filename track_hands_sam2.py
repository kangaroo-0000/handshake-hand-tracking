import os
import cv2
import numpy as np
import torch
import tempfile
import shutil

from sam2.build_sam import build_sam2_video_predictor

def generate_video_masks_sam2(
    video_path: str,
    output_path: str,
    hand_info: list,
    sam2_checkpoint: str = "sam2.1_hiera_large.pt",
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cpu",
    keep_temp_frames: bool = False,
):
    """
    Generate hand masks for an entire video using SAM 2 and bounding-box prompts.

    Args:
        video_path (str): Path to the input .mp4 video.
        output_path (str): Where to save the output masked video (.mp4).
        hand_info (list): A list of dicts from Part 1; each has "bbox": (x_min, y_min, x_max, y_max).
        sam2_checkpoint (str): Path to the SAM 2 model weights (.pt).
        sam2_config (str): Path to the SAM 2 config YAML file.
        device (str): "cpu", "mps" (Apple GPU), or "cuda" (NVIDIA). Default is "cpu".
        keep_temp_frames (bool): If True, keep the extracted frames folder for debugging.

    Returns:
        None; writes out a masked video at `output_path`.
    """
    # 1. Extract frames from the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Make a temp directory to store all JPEG frames
    temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    print(f"[INFO] Extracting {frame_count} frames to: {temp_dir}")

    idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_path = os.path.join(temp_dir, f"{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()

    try:
        # 2. Build SAM 2 predictor
        print("[INFO] Building SAM 2 predictor...")
        predictor = build_sam2_video_predictor(
            sam2_config,
            sam2_checkpoint,
            device=torch.device(device),
        )

        # 3. Initialize predictor state on the extracted frames directory
        print("[INFO] Initializing SAM 2 on frames directory...")
        inference_state = predictor.init_state(video_path=temp_dir)

        # 4. Reset any previous state
        predictor.reset_state(inference_state)

        # 5. Add bounding box prompts on the first frame (frame index=0)
        ann_frame_idx = 0
        obj_id = 1
        for hand_dict in hand_info:
            (x_min, y_min, x_max, y_max) = hand_dict["bbox"]
            box_prompt = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                box=box_prompt
            )
            obj_id += 1

        # 6. Propagate through the entire video
        video_segments = {}
        print("[INFO] Running propagate_in_video across all frames...")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # out_mask_logits is shape [N, H, W] for N objects
            mask_dict = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                # Binarize the logit mask
                bin_mask = (out_mask_logits[i] > 0.0).cpu().numpy()  # shape: (H, W) ideally
                mask_dict[out_obj_id] = bin_mask
            video_segments[out_frame_idx] = mask_dict

        # 7. Re-encode frames with colorized masks
        print(f"[INFO] Writing final masked video to: {output_path}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

        frame_files = sorted(os.listdir(temp_dir))
        for idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(temp_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            if idx in video_segments:
                # We'll accumulate all object masks in one color_mask
                color_mask = np.zeros_like(frame, dtype=np.uint8)
                for object_id, mask in video_segments[idx].items():
                    # Squeeze to ensure shape (H, W)
                    mask_2d = np.squeeze(mask)
                    # Choose a color for each hand/object
                    color = (0, 255, 255)  # or pick different colors per object_id

                    # Use 2D boolean indexing to set (R,G,B) for each "True" pixel
                    color_mask[mask_2d == 1] = color

                # Blend color_mask with the original frame
                alpha = 0.5
                frame = cv2.addWeighted(frame, 1.0, color_mask, alpha, 0)

            out_writer.write(frame)

        out_writer.release()
        print("[INFO] Masked video saved successfully!")

    finally:
        # Clean up temporary frames unless debug
        if not keep_temp_frames:
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"[DEBUG] Temporary frames kept at: {temp_dir}")
