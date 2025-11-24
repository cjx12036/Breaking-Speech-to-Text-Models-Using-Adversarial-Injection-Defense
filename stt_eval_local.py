#!/usr/bin/env python3
import os
import argparse
import csv

import glob
import whisper
from jiwer import wer


def find_audio_files(directory):
    """
    Find .wav and .flac (upper- and lower-case), recursively.
    """
    exts = (".wav", ".WAV", ".flac", ".FLAC")
    paths = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith(exts):
                paths.append(os.path.join(root, name))
    return paths


def index_files_by_stem(paths):
    """
    Map file stem (name without extension) -> full path.

    Example:
        /path/to/clip01.wav -> key "clip01"
    """
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def transcribe_files_whisper(model, files_dict, language="en"):
    """
    Run Whisper STT on all files in files_dict (id -> path).
    Returns a dict: id -> transcript string.
    """
    transcripts = {}
    items = sorted(files_dict.items())
    total = len(items)

    for idx, (file_id, path) in enumerate(items, start=1):
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"[Whisper] Transcribing file {idx}/{total}: {path}")

        result = model.transcribe(path, language=language, fp16=False)
        text = result.get("text", "").strip()
        transcripts[file_id] = text

    return transcripts


def match_adv_to_clean_by_name(clean_transcripts, adv_transcripts):
    """
    For each adversarial file, find the clean file whose ID appears
    as a substring in the adversarial filename (case-insensitive).

    Example:
        clean_id: "clip01"
        adv_id:   "clip01_15kHz_0.05"
    """
    rows = []
    skipped = 0

    clean_ids = list(clean_transcripts.keys())
    print(f"\nMatching adversarial files to clean files by name substring...")
    print(f"Clean IDs: {clean_ids}")

    for adv_id, hyp in adv_transcripts.items():
        adv_lower = adv_id.lower()
        candidates = [cid for cid in clean_ids if cid.lower() in adv_lower]

        if not candidates or not hyp.strip():
            skipped += 1
            continue

        # If multiple candidates match, pick the longest name (most specific)
        best_clean_id = max(candidates, key=len)
        ref = clean_transcripts[best_clean_id]
        score = wer(ref, hyp)

        rows.append(
            {
                "file_id": adv_id,
                "clean_id": best_clean_id,
                "reference": ref,
                "hypothesis": hyp,
                "wer": score,
                "wer_percent": score * 100.0,
                "condition": "adversarial",
            }
        )

    print(f"Adversarial files matched to some clean file: {len(rows)}")
    print(f"Adversarial files skipped (no match or empty transcript): {skipped}")
    return rows


def run_evaluation(clean_dir, adv_dir, whisper_model="base", language="en", output_csv="stt_evaluation_results.csv"):
    """
    Main evaluation routine:
      - load Whisper model
      - transcribe clean + adversarial sets
      - compute WER
      - save detailed CSV
    """
    if not os.path.isdir(clean_dir):
        print(f"Error: clean_dir does not exist or is not a directory: {clean_dir}")
        return
    if not os.path.isdir(adv_dir):
        print(f"Error: adv_dir does not exist or is not a directory: {adv_dir}")
        return

    print(f"Loading Whisper model '{whisper_model}' ...")
    model = whisper.load_model(whisper_model)

    print(f"Finding clean audio files in: {clean_dir}")
    clean_paths = find_audio_files(clean_dir)
    print(f"Found {len(clean_paths)} clean files.")

    print(f"Finding adversarial audio files in: {adv_dir}")
    adv_paths = find_audio_files(adv_dir)
    print(f"Found {len(adv_paths)} adversarial files.")

    if not clean_paths:
        print("Error: no clean audio files found.")
        return
    if not adv_paths:
        print("Error: no adversarial audio files found.")
        return

    clean_files = index_files_by_stem(clean_paths)
    adv_files = index_files_by_stem(adv_paths)

    print("\nTranscribing clean audio (reference transcripts)...")
    clean_transcripts = transcribe_files_whisper(model, clean_files, language=language)

    print("\nTranscribing adversarial audio (attack transcripts)...")
    adv_transcripts = transcribe_files_whisper(model, adv_files, language=language)

    # "Clean condition" rows: ref vs ref (WER 0 by construction)
    clean_rows = []
    for clean_id, ref_text in clean_transcripts.items():
        clean_rows.append(
            {
                "file_id": clean_id,
                "clean_id": clean_id,
                "reference": ref_text,
                "hypothesis": ref_text,
                "wer": 0.0,
                "wer_percent": 0.0,
                "condition": "clean",
            }
        )

    # Adversarial rows: match to clean by filename substring and compute WER
    adv_rows = match_adv_to_clean_by_name(clean_transcripts, adv_transcripts)

    all_rows = clean_rows + adv_rows

    # Average WER per condition
    avg_by_condition = {}
    for row in all_rows:
        cond = row["condition"]
        avg_by_condition.setdefault(cond, []).append(row["wer"])

    print("\nAverage WER by condition:")
    for cond, scores in avg_by_condition.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {cond}: {avg:.3f} ({avg*100:.1f}%)")
        else:
            print(f"  {cond}: no data")

    # Save CSV
    fieldnames = ["file_id", "clean_id", "condition", "wer", "wer_percent", "reference", "hypothesis"]
    print(f"\nSaving detailed results to: {output_csv}")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print("Done.")
    print("\nCSV files in current working directory:")
    print(glob.glob("*.csv"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper WER on clean vs adversarial audio sets."
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        required=True,
        help="Directory containing clean reference audio files.",
    )
    parser.add_argument(
        "--adv_dir",
        type=str,
        required=True,
        help="Directory containing adversarial (or defended) audio files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="Whisper model name (tiny, base, small, medium, large). Default: base",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for Whisper (default: en).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="stt_evaluation_results.csv",
        help="Path to output CSV file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        clean_dir=args.clean_dir,
        adv_dir=args.adv_dir,
        whisper_model=args.model,
        language=args.language,
        output_csv=args.output_csv,
    )

"""
python stt_eval_local.py \
  --clean_dir "./data/clean_audio" \
  --adv_dir "./adversarial_audio_highfreq_4.5_6.5kHz" \
  --model base \
  --output_csv "stt_highfreq.csv"

"""