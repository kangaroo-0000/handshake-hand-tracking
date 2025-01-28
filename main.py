from detect_hands import detect_hands_in_first_frame
from track_hands_sam2 import generate_video_masks_sam2

def main():
    input_video = "test.mp4"
    output_video = "test_output.mp4"

    # 1) Detect hands (Part 1)
    hands_info = detect_hands_in_first_frame(input_video)
    print("[INFO] Detected bounding boxes:", [h["bbox"] for h in hands_info])

    # 2) Segment & track (Part 2)
    #    Adjust sam2_checkpoint / config paths as needed
    generate_video_masks_sam2(
        video_path=input_video,
        output_path=output_video,
        hand_info=hands_info,
        sam2_checkpoint="sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        device="cuda",
    )

if __name__ == "__main__":
    main()
