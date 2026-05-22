"""Live audio delay that follows the video pipeline latency.

Captures from the system default input (your mic) and emits to a chosen output
device (e.g. BlackHole 2ch) with a target delay in milliseconds. The target can
be updated at any time; we drift the *applied* delay toward the target by a
fraction of a block per callback so changes don't cause audible clicks.

Used by live_remote.py so the receiver in Zoom/Meet sees video and audio
arriving in sync without anyone editing OBS sync offsets by hand.
"""

from __future__ import annotations

import sys
import threading
from typing import Optional, Union

import numpy as np


class AudioSyncDelay:
    def __init__(
        self,
        input_device: Optional[Union[int, str]] = None,
        output_device: Union[int, str] = "BlackHole 2ch",
        samplerate: int = 48000,
        blocksize: int = 512,
        max_delay_s: float = 2.0,
        initial_delay_ms: float = 300.0,
        in_channels: int = 1,
        out_channels: int = 2,
    ) -> None:
        import sounddevice as sd  # imported lazily so live_remote starts without it
        self._sd = sd

        self.samplerate = samplerate
        self.blocksize = blocksize
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_device = input_device
        self.output_device = output_device

        self._buf_len = int(samplerate * max_delay_s)
        # Mono internal buffer; we mix down on input, duplicate on output.
        self._buf = np.zeros(self._buf_len, dtype=np.float32)
        self._write_idx = 0

        self._lock = threading.Lock()
        self._target_delay_samples = int(initial_delay_ms / 1000.0 * samplerate)
        self._current_delay_samples = self._target_delay_samples

        self.stream: Optional["sd.Stream"] = None
        self.last_error: Optional[str] = None

    # ---- public knob ----------------------------------------------------
    def set_target_delay_ms(self, ms: float) -> None:
        ms = max(0.0, min(ms, 1000.0 * self._buf_len / self.samplerate - 50.0))
        with self._lock:
            self._target_delay_samples = int(ms / 1000.0 * self.samplerate)

    @property
    def current_delay_ms(self) -> float:
        return 1000.0 * self._current_delay_samples / self.samplerate

    # ---- audio callback -------------------------------------------------
    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            # under/overflow — annoying but not fatal; the smoothing helps recover.
            pass

        # Down-mix input to mono if needed.
        if indata.shape[1] == 1:
            mono = indata[:, 0]
        else:
            mono = indata.mean(axis=1)

        end = self._write_idx + frames
        if end <= self._buf_len:
            self._buf[self._write_idx:end] = mono
        else:
            n1 = self._buf_len - self._write_idx
            self._buf[self._write_idx:] = mono[:n1]
            self._buf[:frames - n1] = mono[n1:]
        self._write_idx = end % self._buf_len

        # Drift current toward target very slowly so audio doesn't pitch-bend
        # audibly. 1 sample per block at 48k blocksize 512 ≈ 0.2% time-stretch,
        # which is imperceptible. Convergence from a 100ms mismatch takes ~50s,
        # but typical use-case (steady-state latency) barely needs to adjust.
        with self._lock:
            target = self._target_delay_samples
        diff = target - self._current_delay_samples
        if diff != 0:
            step = 1 if diff > 0 else -1
            self._current_delay_samples += step

        # Read `frames` samples ending `current_delay` samples before write_idx.
        start = (self._write_idx - self._current_delay_samples - frames) % self._buf_len
        if start + frames <= self._buf_len:
            mono_out = self._buf[start:start + frames]
        else:
            n1 = self._buf_len - start
            mono_out = np.concatenate((self._buf[start:], self._buf[:frames - n1]))

        if self.out_channels == 1:
            outdata[:, 0] = mono_out
        else:
            for ch in range(self.out_channels):
                outdata[:, ch] = mono_out

    # ---- lifecycle ------------------------------------------------------
    def start(self) -> bool:
        try:
            self.stream = self._sd.Stream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=(self.in_channels, self.out_channels),
                dtype="float32",
                device=(self.input_device, self.output_device),
                callback=self._callback,
            )
            self.stream.start()
            return True
        except Exception as e:
            self.last_error = str(e)
            self.stream = None
            return False

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


def find_device_index(name_substring: str, kind: str) -> Optional[int]:
    """Return the first device whose name contains `name_substring` and has the
    requested kind (`input` or `output`) channels.
    """
    import sounddevice as sd
    needle = name_substring.lower()
    for i, d in enumerate(sd.query_devices()):
        if needle not in d["name"].lower():
            continue
        if kind == "input" and d["max_input_channels"] > 0:
            return i
        if kind == "output" and d["max_output_channels"] > 0:
            return i
    return None


if __name__ == "__main__":
    # Quick standalone smoke test: mic -> BlackHole with 300ms delay.
    import time
    out_idx = find_device_index("BlackHole", "output")
    if out_idx is None:
        print("BlackHole 2ch not found; install with: brew install blackhole-2ch",
              file=sys.stderr)
        sys.exit(1)
    a = AudioSyncDelay(output_device=out_idx, initial_delay_ms=300)
    if not a.start():
        print(f"failed: {a.last_error}", file=sys.stderr)
        sys.exit(1)
    print(f"mic -> BlackHole at {a.current_delay_ms:.0f}ms. ctrl-c to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        a.stop()
