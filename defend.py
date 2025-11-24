import os
import argparse
import re

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt


# Expected pattern:
#   librispeech_<utt_id>_freq<freq>_amp<amp>.wav
# Example:
#   librispeech_1_freq12000_amp0.010.wav
FILENAME_RE = re.compile(r"librispeech_(\d+)_freq(\d+)_amp([\d.]+)\.wav")


def parse_filename(fname: str):
    """
    Parse utt_id, frequency (Hz), and amplitude from a filename.

    Example:
        librispeech_1_freq12000_amp0.010.wav
        -> utt_id = 1, freq_hz = 12000, amp_str = "0.010"
    """
    m = FILENAME_RE.match(fname)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {fname}")
    utt_id = int(m.group(1))
    freq_hz = int(m.group(2))
    amp_str = m.group(3)  # Keep as string to preserve formatting
    return utt_id, freq_hz, amp_str


def load_audio(path: str, target_sr: int = 16000):
    """
    Load audio as mono and resample to target_sr (default: 16 kHz).
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr


def save_audio(path: str, y: np.ndarray, sr: int):
    """
    Save waveform as 16-bit PCM WAV.
    """
    sf.write(path, y, sr, subtype="PCM_16")


def lowpass_filter(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 6):
    """
    Apply a Butterworth low-pass filter that keeps frequencies below cutoff_hz.
    """
    nyquist = 0.5 * sr

    # Safety: cutoff must be strictly less than Nyquist
    if cutoff_hz >= nyquist:
        print(
            f"[WARN] cutoff_hz={cutoff_hz} >= Nyquist={nyquist}. "
            f"Clamping cutoff to 0.99 * Nyquist."
        )
        cutoff_hz = 0.99 * nyquist

    norm_cutoff = cutoff_hz / nyquist
    b, a = butter(order, norm_cutoff, btype="low")
    y_lp = filtfilt(b, a, y)
    return y_lp


# --------- NEW: Spectral Gating Denoiser --------- #

def spectral_gate(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    noise_floor_percentile: float = 20.0,
    gate_threshold: float = 1.5,
    reduction_db: float = 15.0,
):
    """
    Simple spectral gating / frequency-domain noise reduction.

    1. Compute STFT.
    2. Estimate a noise floor per frequency bin using a low percentile.
    3. Build a gate (mask) that attenuates bins close to the noise floor.
    4. Apply the mask and invert STFT.

    This is intentionally simple but good enough for demonstrating
    spectral-domain denoising in your project.
    """
    # Step 1: STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S), np.angle(S)

    # Step 2: estimate noise floor (per frequency bin)
    # Using a low percentile over time as an approximation of noise level.
    noise_floor = np.percentile(magnitude, noise_floor_percentile, axis=1, keepdims=True)

    # Step 3: build a gate (mask)
    # If magnitude is not much higher than noise_floor * gate_threshold,
    # we treat it as (mostly) noise and attenuate it.
    eps = 1e-8
    gate = magnitude >= (noise_floor * gate_threshold)

    # Convert reduction in dB to a linear gain factor.
    # For example, 15 dB reduction -> gain ≈ 0.178
    noise_gain = 10.0 ** (-reduction_db / 20.0)

    gain = np.where(gate, 1.0, noise_gain)

    # Optional: you could smooth gain over time/frequency here if desired.

    # Step 4: apply mask
    S_denoised = S * gain

    # Step 5: inverse STFT
    # Use length=len(y) to keep the same number of samples as input.
    y_denoised = librosa.istft(S_denoised, hop_length=hop_length, length=len(y))

    return y_denoised


def denoise_signal(
    y: np.ndarray,
    sr: int,
):
    """
    Wrapper for spectral gating denoising.

    Replaces the previous time-domain Wiener filter with a simple
    frequency-domain spectral gating method.
    """
    return spectral_gate(y, sr)


# ------------- Defense Pipeline ------------- #

def defend_waveform(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 12000.0,
    use_denoise: bool = True,
):
    """
    Full defense pipeline for a single waveform:
    1. Low-pass filter to remove high-frequency adversarial components.
    2. Optional spectral-gating denoising.
    3. Normalize to [-1, 1].
    """
    # Step 1: low-pass filtering
    y_f = lowpass_filter(y, sr, cutoff_hz=cutoff_hz, order=6)

    # Step 2: optional denoising (now spectral gating)
    if use_denoise:
        y_f = denoise_signal(y_f, sr)

    # Step 3: normalization to [-1, 1]
    max_val = np.max(np.abs(y_f)) + 1e-9
    y_f = y_f / max_val

    return y_f


def defend_folder(
    input_dir: str,
    output_dir: str,
    cutoff_hz: float = 12000.0,
    use_denoise: bool = True,
    target_sr: int = 16000,
):
    """
    Apply the defense pipeline to all adversarial WAV files in input_dir.

    Expected filenames match:
        librispeech_<utt_id>_freq<freq>_amp<amp>.wav

    Output filenames will be:
        librispeech_<utt_id>_freq<freq>_amp<amp>_lp<cutoff_in_kHz>_<dn|nodn>.wav
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".wav"):
            continue

        in_path = os.path.join(input_dir, fname)

        try:
            utt_id, freq_hz, amp_str = parse_filename(fname)
        except ValueError as e:
            # If there are other WAV files that don't match the naming scheme,
            # we just warn and skip them.
            print(f"[WARN] {e}")
            continue

        # 1. Load adversarial audio
        y, sr = load_audio(in_path, target_sr=target_sr)

        # 2. Apply defense
        y_def = defend_waveform(
            y, sr, cutoff_hz=cutoff_hz, use_denoise=use_denoise
        )

        # 3. Construct output filename
        cutoff_tag = f"lp{int(cutoff_hz/1000)}k"
        denoise_tag = "dn" if use_denoise else "nodn"

        out_fname = (
            f"librispeech_{utt_id}_freq{freq_hz}_amp{amp_str}_"
            f"{cutoff_tag}_{denoise_tag}.wav"
        )
        out_path = os.path.join(output_dir, out_fname)

        # 4. Save defended audio
        save_audio(out_path, y_def, sr)

        print(
            f"[OK] utt={utt_id:02d}, freq={freq_hz}, amp={amp_str}  "
            f"-> {out_path}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Defense pipeline: low-pass filtering + optional spectral-gating "
            "denoising for a folder of adversarial audios named like "
            'librispeech_<id>_freq<freq>_amp<amp>.wav'
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing adversarial .wav files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save defended .wav files.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=8000.0,
        help="Low-pass cutoff frequency in Hz (e.g., 7000, 8000, 12000).",
    )
    parser.add_argument(
        "--no_denoise",
        action="store_true",
        help="If set, skip denoising and only apply low-pass filtering.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sampling rate for loading audio (default: 16000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    defend_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cutoff_hz=args.cutoff,
        use_denoise=not args.no_denoise,
        target_sr=args.sr,
    )

"""
python defend_folder.py \
  --input_dir . \
  --output_dir defended_lp8_dn \
  --cutoff 8000
  --no_denoise  # optional
"""