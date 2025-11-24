import os
import shutil
import argparse


def extract_vocals(demucs_root: str, output_dir: str):
    """
    Extract vocals.wav from each Demucs output folder and rename it:

        <folder_name>_demucs.wav

    Example:
        demucs_root/htdemucs/librispeech_1_noise0.0050/vocals.wav
        -> output_dir/librispeech_1_noise0.0050_demucs.wav
    """

    os.makedirs(output_dir, exist_ok=True)

    # demucs_root should contain exactly one model directory, e.g., 'htdemucs'
    for model_name in os.listdir(demucs_root):
        model_dir = os.path.join(demucs_root, model_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"[INFO] Found Demucs model directory: {model_name}")

        # iterate each track subfolder
        for track_name in os.listdir(model_dir):
            track_dir = os.path.join(model_dir, track_name)
            if not os.path.isdir(track_dir):
                continue

            # Expect vocals.wav here
            src_vocals = os.path.join(track_dir, "vocals.wav")

            if not os.path.isfile(src_vocals):
                print(f"[WARN] No vocals.wav found in {track_dir}, skipping.")
                continue

            # new name: <track_name>_demucs.wav
            out_name = f"{track_name}_demucs.wav"
            out_path = os.path.join(output_dir, out_name)

            shutil.copy2(src_vocals, out_path)
            print(f"[OK] {src_vocals} -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract vocals.wav from Demucs output and flatten structure."
    )
    parser.add_argument(
        "--demucs_root",
        type=str,
        required=True,
        help="Path to demucs_raw_out folder."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save extracted & renamed vocals.wav files."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_vocals(args.demucs_root, args.output_dir)


'''
python flatten_demucs_outputs.py \
  --demucs_root demucs_raw_out \
  --output_dir defended_whitenoise_demucs

'''