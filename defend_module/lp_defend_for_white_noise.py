import os
import argparse
import re

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

# Pattern for white-noise files:
#   librispeech_<utt_id>_noise<level>.wav
# Example:
#   librispeech_1_noise0.0050.wav
NOISE_FILENAME_RE = re.compile(r"librispeech_(\d+)_noise([\d.]+)\.wav")


def parse_noise_filename(fname: str):
    """
    Parse utt_id and noise level from filenames like:
        librispeech_1_noise0.0050.wav
    """
    m = NOISE_FILENAME_RE.match(fname)
    if not m:
        raise ValueError(f"Filename does not match noise pattern: {fname}")
    utt_id = int(m.group(1))
    noise_level_str = m.group(2)
    return utt_id, noise_level_str


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


# ---------- Spectral Gating Denoiser ---------- #

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

    1. STFT
    2. Estimate noise floor (per frequency bin) using a low percentile.
    3. Build a gate mask that attenuates bins near the noise floor.
    4. Apply the mask and invert STFT.
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(S), np.angle(S)

    # Estimate noise floor per frequency bin.
    noise_floor = np.percentile(magnitude, noise_floor_percentile, axis=1, keepdims=True)

    eps = 1e-8
    gate = magnitude >= (noise_floor * gate_threshold)

    # Convert dB reduction to linear gain
    noise_gain = 10.0 ** (-reduction_db / 20.0)
    gain = np.where(gate, 1.0, noise_gain)

    S_denoised = S * gain
    y_denoised = librosa.istft(S_denoised, hop_length=hop_length, length=len(y))
    return y_denoised


def defend_waveform(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 7000.0,
    use_denoise: bool = True,
):
    """
    Defense pipeline for a single waveform:
    1. Low-pass filter.
    2. Optional spectral-gating denoising.
    3. Normalize to [-1, 1].
    """
    y_f = lowpass_filter(y, sr, cutoff_hz=cutoff_hz, order=6)

    if use_denoise:
        y_f = spectral_gate(y_f, sr)

    max_val = np.max(np.abs(y_f)) + 1e-9
    y_f = y_f / max_val
    return y_f


def defend_folder(
    input_dir: str,
    output_dir: str,
    cutoff_hz: float = 7000.0,
    use_denoise: bool = True,
    target_sr: int = 16000,
):
    """
    Apply the defense pipeline to all white-noise WAV files in input_dir.

    Expected filenames:
        librispeech_<utt_id>_noise<level>.wav

    Output filenames:
        librispeech_<utt_id>_noise<level>_lp<cutoff_kHz>_<dn|nodn>.wav
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".wav"):
            continue

        in_path = os.path.join(input_dir, fname)

        try:
            utt_id, noise_level = parse_noise_filename(fname)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        y, sr = load_audio(in_path, target_sr=target_sr)

        y_def = defend_waveform(
            y, sr, cutoff_hz=cutoff_hz, use_denoise=use_denoise
        )

        cutoff_tag = f"lp{int(cutoff_hz/1000)}k"
        denoise_tag = "dn" if use_denoise else "nodn"

        out_fname = (
            f"librispeech_{utt_id}_noise{noise_level}_"
            f"{cutoff_tag}_{denoise_tag}.wav"
        )
        out_path = os.path.join(output_dir, out_fname)

        save_audio(out_path, y_def, sr)

        print(
            f"[OK] utt={utt_id:02d}, noise={noise_level} "
            f"-> {out_path}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Defense pipeline (LPF + spectral gating) for white-noise-"
            "corrupted audios named librispeech_<id>_noise<level>.wav"
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing noisy .wav files.",
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
        default=7000.0,
        help="Low-pass cutoff frequency in Hz (e.g., 6000, 7000, 8000).",
    )
    parser.add_argument(
        "--no_denoise",
        action="store_true",
        help="If set, skip spectral-gating denoising.",
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
