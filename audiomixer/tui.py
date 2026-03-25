#!/usr/bin/env python3
"""
TUI for selecting USB microphones, monitoring their live volume,
recording a mixed output, and browsing/playing back recordings.

Usage:
  source .venv/bin/activate && python tui.py
"""

import glob
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_backend import AudioBackend, AudioDevice
from web_server import start_web_server
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Static,
)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCKSIZE = 1024
BAR_WIDTH = 50
NOISE_GATE_RMS = 0.005
DEAD_MIC_BLOCKS = 30
AUTO_SELECT_KEYWORD = "fifine"
# Dominant mic must drop below this to allow switching (slightly above noise gate)
DOMINANT_RELEASE_RMS = 0.008
# Minimum hold time even after dominant mic drops (prevents rapid toggling)
DOMINANT_MIN_HOLD = 0.5
# New mic must be this many times louder to interrupt (only after release)
DOMINANT_SWITCH_RATIO = 1.5


# ── Volume meter widget ──────────────────────────────────────────────

class VolumeMeter(Widget):
    """Displays a live volume bar for one microphone."""

    DEFAULT_CSS = """
    VolumeMeter {
        height: 3;
        padding: 0 1;
    }
    """

    level: reactive[float] = reactive(0.0)
    peak: reactive[float] = reactive(0.0)
    is_dominant: reactive[bool] = reactive(False)
    is_muted: reactive[bool] = reactive(False)
    is_dead: reactive[bool] = reactive(False)
    below_gate: reactive[bool] = reactive(False)

    def __init__(self, device_index: int, device_name: str, mic_number: int) -> None:
        super().__init__()
        self.device_index = device_index
        self.device_name = device_name
        self.mic_number = mic_number

    def render(self) -> str:
        bar_len = int(self.level * BAR_WIDTH)
        peak_pos = int(self.peak * BAR_WIDTH)
        bar_len = min(bar_len, BAR_WIDTH)
        peak_pos = min(peak_pos, BAR_WIDTH)

        gate_pos = int(NOISE_GATE_RMS * 10 * BAR_WIDTH)
        gate_pos = min(gate_pos, BAR_WIDTH - 1)

        bar = ""
        for i in range(BAR_WIDTH):
            if i < bar_len:
                if self.is_dead:
                    bar += "[#880000]█[/]"
                elif self.is_muted or self.below_gate:
                    bar += "[#555555]█[/]"
                elif self.level > 0.85:
                    bar += "[red]█[/]"
                elif self.level > 0.6:
                    bar += "[yellow]█[/]"
                else:
                    bar += "[green]█[/]"
            elif i == gate_pos:
                bar += "[#666600]┊[/]"
            elif i == peak_pos and self.peak > 0.01:
                bar += "[red]│[/]"
            else:
                bar += "[#333333]░[/]"

        db = 20 * np.log10(max(self.level, 1e-10))
        db_str = f"{db:+.0f} dB" if self.level > 0.001 else "  -∞ dB"

        if self.is_dead:
            label = f"[bold red]✖ Mic {self.mic_number}[/] [{self.device_index}] {self.device_name} [bold red]NO SIGNAL[/]"
        elif self.is_dominant:
            label = f"[bold cyan]● Mic {self.mic_number}[/] [{self.device_index}] {self.device_name} [bold cyan]ACTIVE[/]"
        elif self.is_muted:
            label = f"[dim]○ Mic {self.mic_number}[/] [{self.device_index}] {self.device_name} [dim]muted[/]"
        elif self.below_gate:
            label = f"[dim]Mic {self.mic_number}[/] [{self.device_index}] {self.device_name} [dim]gate[/]"
        else:
            label = f"[bold]Mic {self.mic_number}[/] [{self.device_index}] {self.device_name}"

        return f"{label}\n  {bar} {db_str}"


# ── Playback progress widget ────────────────────────────────────────

class PlaybackBar(Widget):
    DEFAULT_CSS = """
    PlaybackBar {
        height: 3;
        padding: 0 2;
    }
    """

    progress: reactive[float] = reactive(0.0)
    filename: reactive[str] = reactive("")
    playing: reactive[bool] = reactive(False)

    def render(self) -> str:
        if not self.filename:
            return "[dim]No file selected[/]"
        bar_w = 40
        filled = int(self.progress * bar_w)
        bar = "[cyan]█[/]" * filled + "[#333333]░[/]" * (bar_w - filled)
        pct = int(self.progress * 100)
        state = "[bold cyan]▶ Playing[/]" if self.playing else "[dim]■ Stopped[/]"
        return f"{state}  {self.filename}\n  {bar} {pct}%"


# ── Main App ─────────────────────────────────────────────────────────

class AudioMixerApp(App):
    CSS = """
    #device-list {
        height: 1fr;
        border: solid $primary;
        padding: 1 2;
    }
    #btn-bar {
        height: 3;
        align: center middle;
    }
    #btn-bar Button {
        margin: 0 1;
    }
    #meters {
        height: 1fr;
        padding: 1 0;
    }
    #live-status {
        height: 3;
        padding: 0 2;
        dock: bottom;
        border-top: solid $primary;
    }
    #recordings-list {
        height: 1fr;
        border: solid $primary;
        padding: 1 2;
    }
    #rec-btn-bar {
        height: 3;
        align: center middle;
    }
    #rec-btn-bar Button {
        margin: 0 1;
    }
    #playback-area {
        height: auto;
        padding: 1 0;
    }
    #rec-status {
        height: 3;
        padding: 0 2;
        dock: bottom;
        border-top: solid $primary;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        # Live screen
        Binding("r", "toggle_record", "Record"),
        Binding("s", "stop_and_save", "Stop & Save"),
        Binding("d", "toggle_dominant", "Dominant"),
        Binding("l", "show_recordings", "Recordings"),
        Binding("m", "change_mics", "Mics"),
        # Recordings screen
        Binding("p", "play_recording", "Play"),
        Binding("x", "stop_playback", "Stop Play"),
        Binding("b", "back_to_live", "Back"),
    ]

    TITLE = "Audio Mixer"

    def __init__(self) -> None:
        super().__init__()
        self.backend = AudioBackend()
        self.input_devices: list[AudioDevice] = []
        self.selected_devices: list[AudioDevice] = []
        self.streams: list = []
        self.levels: dict[int, float] = {}
        self.peaks: dict[int, float] = {}
        self.meters: dict[int, VolumeMeter] = {}
        self.is_live = False
        self.is_recording = False
        self.record_buffers: dict[int, list] = {}
        self.record_start_time: float = 0
        self._screen = "select"
        self._playback_stop = False
        self._playback_thread = None
        self._recording_files: list[str] = []
        self.dominant_mode = False
        self.dominant_idx: int | None = None
        self._rms_per_block: dict[int, float] = {}
        self._smoothed_rms: dict[int, float] = {}
        self._dominant_hold_until: float = 0
        self._gain: dict[int, float] = {}
        self._zero_block_count: dict[int, int] = {}
        self._is_dead: dict[int, bool] = {}

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        """Only show bindings relevant to current screen."""
        live_actions = {"toggle_record", "stop_and_save", "toggle_dominant", "show_recordings", "change_mics"}
        rec_actions = {"play_recording", "stop_playback", "back_to_live"}
        select_actions = {"show_recordings"}

        if self._screen == "live":
            if action in rec_actions:
                return False
        elif self._screen == "recordings-screen":
            if action in live_actions:
                return False
        elif self._screen == "select":
            if action in live_actions or action in rec_actions:
                if action not in select_actions:
                    return False
        return True

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        # ── Selection screen ──
        with Vertical(id="select-screen"):
            yield Label("[bold]Select microphones to monitor:[/]\n")
            with VerticalScroll(id="device-list"):
                for dev in self._get_input_devices():
                    is_auto = AUTO_SELECT_KEYWORD in dev.name.lower()
                    yield Checkbox(f"[{dev.index}] {dev.name}", id=f"dev-{dev.index}", value=is_auto)
            with Horizontal(id="btn-bar"):
                yield Button("Go Live", id="btn-live", variant="primary")
                yield Button("Recordings", id="btn-recordings", variant="default")

        # ── Live screen ──
        with Vertical(id="live-screen"):
            with VerticalScroll(id="meters"):
                pass
            yield Static("", id="live-status")

        # ── Recordings screen ──
        with Vertical(id="recordings-screen"):
            yield Label("[bold]Recordings:[/]\n")
            yield ListView(id="recordings-list")
            with Vertical(id="playback-area"):
                yield PlaybackBar(id="playback-bar")
            with Horizontal(id="rec-btn-bar"):
                yield Button("Play", id="btn-play", variant="success")
                yield Button("Stop", id="btn-stop-playback", variant="error")
                yield Button("Delete", id="btn-delete", variant="warning")
                yield Button("Refresh", id="btn-refresh", variant="default")
                yield Button("Back", id="btn-back", variant="primary")
            yield Static("", id="rec-status")

    def on_mount(self) -> None:
        self.query_one("#live-screen").display = False
        self.query_one("#recordings-screen").display = False
        # Start web server for remote playback
        rec_dir = os.environ.get("RECORDINGS_DIR", os.getcwd())
        os.makedirs(rec_dir, exist_ok=True)
        self._web_server = start_web_server(port=8080, directory=rec_dir)
        self.notify("Web UI: http://0.0.0.0:8080", severity="information")
        # Auto-start if fifine mics found
        auto_devices = [d for d in self.input_devices if AUTO_SELECT_KEYWORD in d.name.lower()]
        if auto_devices:
            self.set_timer(0.1, self._auto_go_live)

    def _auto_go_live(self) -> None:
        self._go_live()

    def _get_input_devices(self) -> list[AudioDevice]:
        if not self.input_devices:
            self.input_devices = self.backend.discover()
        return self.input_devices

    def _switch_screen(self, screen_name: str) -> None:
        for sid in ("select-screen", "live-screen", "recordings-screen"):
            self.query_one(f"#{sid}").display = False
        self.query_one(f"#{screen_name}").display = True
        self._screen = screen_name
        self.refresh_bindings()

    # ── Go Live ──────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        if btn == "btn-live":
            self._go_live()
        elif btn == "btn-recordings":
            self._show_recordings_screen()
        elif btn == "btn-play":
            self._play_selected()
        elif btn == "btn-stop-playback":
            self._safe_stop_playback()
        elif btn == "btn-delete":
            self._delete_selected()
        elif btn == "btn-refresh":
            self._refresh_recordings()
        elif btn == "btn-back":
            self._back_from_recordings()

    def _go_live(self) -> None:
        # Stop existing streams if re-entering
        if self.is_live:
            self._stop_streams()

        self.selected_devices = []
        for dev in self.input_devices:
            try:
                cb = self.query_one(f"#dev-{dev.index}", Checkbox)
                if cb.value:
                    self.selected_devices.append(dev)
            except Exception:
                pass

        if not self.selected_devices:
            self.notify("Select at least one microphone!", severity="error")
            return

        self._switch_screen("live-screen")
        self._screen = "live"

        # Clear old meters
        meters_container = self.query_one("#meters")
        for meter in list(self.meters.values()):
            meter.remove()
        self.meters.clear()

        # Create fresh meters
        for i, dev in enumerate(self.selected_devices):
            meter = VolumeMeter(dev.index, dev.name, i + 1)
            meter.id = f"meter-{dev.index}"
            meters_container.mount(meter)
            self.meters[dev.index] = meter
            self.levels[dev.index] = 0.0
            self.peaks[dev.index] = 0.0

        self._start_streams()
        self._start_ui_refresh()

    def _stop_streams(self) -> None:
        self.is_live = False
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams.clear()
        self._rms_per_block.clear()
        self._smoothed_rms.clear()
        self._gain.clear()
        self._zero_block_count.clear()
        self._is_dead.clear()
        self.dominant_idx = None

    def _start_streams(self) -> None:
        self.is_live = True
        for dev in self.selected_devices:
            idx = dev.index
            self.record_buffers[idx] = []

            def make_callback(dev_idx):
                def callback(indata, frames, time_info, status):
                    mono = indata[:, 0] if indata.shape[1] > 1 else indata.flatten()
                    rms = float(np.sqrt(np.mean(mono ** 2)))
                    scaled = min(rms * 10, 1.0)
                    self.levels[dev_idx] = scaled
                    self._rms_per_block[dev_idx] = rms

                    # Dead mic detection
                    if rms < 1e-10:
                        self._zero_block_count[dev_idx] = self._zero_block_count.get(dev_idx, 0) + 1
                    else:
                        self._zero_block_count[dev_idx] = 0
                    self._is_dead[dev_idx] = self._zero_block_count.get(dev_idx, 0) >= DEAD_MIC_BLOCKS

                    # Smooth RMS (EMA)
                    alpha = 0.3 if rms > self._smoothed_rms.get(dev_idx, 0) else 0.05
                    self._smoothed_rms[dev_idx] = (
                        alpha * rms + (1 - alpha) * self._smoothed_rms.get(dev_idx, 0)
                    )

                    if scaled > self.peaks[dev_idx]:
                        self.peaks[dev_idx] = scaled

                    # Dominant mic determination
                    # Logic:
                    # 1. If multiple mics hear the same source (all above gate),
                    #    always use the loudest — it's closest to the speaker.
                    # 2. If current dominant is still speaking, hold it.
                    # 3. Only switch when current drops or a clearly louder source appears.
                    if self.dominant_mode:
                        now = time.monotonic()
                        active_rms = {
                            k: v for k, v in self._smoothed_rms.items()
                            if v >= NOISE_GATE_RMS and not self._is_dead.get(k, False)
                        }

                        current_dom = self.dominant_idx

                        if not active_rms:
                            # Everyone is quiet — release after hold
                            if now >= self._dominant_hold_until:
                                self.dominant_idx = None
                        elif current_dom is None or current_dom not in self._smoothed_rms:
                            # No current dominant — pick loudest above gate
                            loudest = max(active_rms, key=active_rms.get)
                            self.dominant_idx = loudest
                            self._dominant_hold_until = now + DOMINANT_MIN_HOLD
                        else:
                            current_rms = self._smoothed_rms.get(current_dom, 0)
                            loudest = max(active_rms, key=active_rms.get)
                            loudest_rms = active_rms[loudest]

                            # Detect "same source, multiple mics" scenario:
                            # If most mics are above gate, it's likely one speaker
                            # heard from different distances → pick loudest (closest).
                            all_alive = {
                                k for k in self._smoothed_rms
                                if not self._is_dead.get(k, False)
                            }
                            above_gate_ratio = len(active_rms) / max(len(all_alive), 1)

                            if above_gate_ratio >= 0.5 and loudest != current_dom:
                                # Many mics hear the same thing — this is one speaker.
                                # Switch to loudest (closest mic) if it's clearly louder.
                                if loudest_rms > current_rms * 1.3:
                                    self.dominant_idx = loudest
                                    self._dominant_hold_until = now + DOMINANT_MIN_HOLD
                                # Otherwise keep current (stable, avoid ping-pong)
                            else:
                                # Few mics active — likely someone speaking close to one mic.
                                # Hold current dominant while they're still speaking.
                                dom_still_speaking = current_rms >= DOMINANT_RELEASE_RMS

                                if dom_still_speaking:
                                    self._dominant_hold_until = now + DOMINANT_MIN_HOLD
                                elif now >= self._dominant_hold_until:
                                    if loudest != current_dom and loudest_rms > current_rms * DOMINANT_SWITCH_RATIO:
                                        self.dominant_idx = loudest
                                        self._dominant_hold_until = now + DOMINANT_MIN_HOLD
                                    elif current_rms < NOISE_GATE_RMS:
                                        self.dominant_idx = loudest if active_rms else None

                    # Recording
                    if self.is_recording:
                        if self.dominant_mode:
                            if self.dominant_idx is None:
                                target = 0.0
                            elif dev_idx == self.dominant_idx:
                                target = 1.0
                            else:
                                target = 0.0
                            current_gain = self._gain.get(dev_idx, 0.0)
                            new_gain = current_gain + 0.15 * (target - current_gain)
                            self._gain[dev_idx] = new_gain
                            self.record_buffers[dev_idx].append(mono * new_gain)
                        else:
                            self.record_buffers[dev_idx].append(mono.copy())
                return callback

            stream = self.backend.open_stream(dev, make_callback(idx))
            stream.start()
            self.streams.append(stream)

    @work(thread=True)
    def _start_ui_refresh(self) -> None:
        while self.is_live:
            for idx, meter in self.meters.items():
                meter.level = self.levels.get(idx, 0.0)
                current_peak = self.peaks.get(idx, 0.0)
                self.peaks[idx] = max(current_peak * 0.95, self.levels.get(idx, 0.0))
                meter.peak = self.peaks[idx]

                is_dead = self._is_dead.get(idx, False)
                smoothed = self._smoothed_rms.get(idx, 0)
                below_gate = smoothed < NOISE_GATE_RMS

                meter.is_dead = is_dead
                meter.below_gate = below_gate and not is_dead

                if self.dominant_mode and not is_dead:
                    meter.is_dominant = (idx == self.dominant_idx) and not below_gate
                    meter.is_muted = (idx != self.dominant_idx) and not below_gate
                else:
                    meter.is_dominant = False
                    meter.is_muted = False
                meter.refresh()

            mode_str = "[bold cyan]DOMINANT[/]" if self.dominant_mode else "[dim]MIX ALL[/]"
            dead_count = sum(1 for v in self._is_dead.values() if v)
            dead_str = f" │ [bold red]{dead_count} dead[/]" if dead_count else ""
            if self.is_recording:
                elapsed = time.time() - self.record_start_time
                status = f"[bold red]● REC[/] {elapsed:.1f}s │ {mode_str}{dead_str}"
            else:
                status = f"{mode_str}{dead_str}"
            self.call_from_thread(self._update_live_status, status)

            time.sleep(0.05)

    def _update_live_status(self, text: str) -> None:
        try:
            self.query_one("#live-status", Static).update(text)
        except Exception:
            pass

    # ── Actions: Live screen ─────────────────────────────────────────

    def action_toggle_dominant(self) -> None:
        if self._screen != "live":
            return
        self.dominant_mode = not self.dominant_mode
        mode = "DOMINANT MIC" if self.dominant_mode else "MIX ALL"
        self.notify(f"Mode: {mode}", severity="information")

    def action_toggle_record(self) -> None:
        if self._screen != "live":
            return
        if not self.is_recording:
            self.is_recording = True
            self.record_start_time = time.time()
            for idx in self.record_buffers:
                self.record_buffers[idx] = []
            self.notify("Recording started", severity="information")
        else:
            self.action_stop_and_save()

    def action_stop_and_save(self) -> None:
        if self._screen != "live" or not self.is_recording:
            return
        self.is_recording = False
        elapsed = time.time() - self.record_start_time
        self._save_recording(elapsed)

    def action_change_mics(self) -> None:
        if self._screen != "live":
            return
        if self.is_recording:
            self.notify("Stop recording first!", severity="error")
            return
        self._stop_streams()
        # Clear meters
        for meter in list(self.meters.values()):
            meter.remove()
        self.meters.clear()
        self._switch_screen("select-screen")
        self._screen = "select"

    def _save_recording(self, elapsed: float) -> None:
        all_arrays = {}
        max_len = 0
        for idx, chunks in self.record_buffers.items():
            if chunks:
                arr = np.concatenate(chunks)
                all_arrays[idx] = arr
                max_len = max(max_len, len(arr))

        if not all_arrays or max_len == 0:
            self.notify("Nothing recorded!", severity="warning")
            return

        mixed = np.zeros(max_len, dtype=np.float32)
        for arr in all_arrays.values():
            padded = np.zeros(max_len, dtype=np.float32)
            padded[: len(arr)] = arr
            mixed += padded
        mixed /= len(all_arrays)

        peak = np.max(np.abs(mixed))
        if peak > 0:
            mixed = mixed / peak * 0.95

        rec_dir = os.environ.get("RECORDINGS_DIR", ".")
        os.makedirs(rec_dir, exist_ok=True)
        output_path = os.path.join(rec_dir, f"mixed_{int(time.time())}.wav")
        sf.write(output_path, mixed, SAMPLE_RATE)
        self.notify(f"Saved {output_path} ({elapsed:.1f}s)", severity="information")

    # ── Actions: Recordings screen ───────────────────────────────────

    def action_show_recordings(self) -> None:
        if self._screen in ("live", "select"):
            self._show_recordings_screen()

    def action_play_recording(self) -> None:
        if self._screen == "recordings-screen":
            self._play_selected()

    def action_stop_playback(self) -> None:
        if self._screen == "recordings-screen":
            self._safe_stop_playback()

    def action_back_to_live(self) -> None:
        if self._screen == "recordings-screen":
            self._back_from_recordings()

    def _show_recordings_screen(self) -> None:
        self._safe_stop_playback()
        self._switch_screen("recordings-screen")
        self._screen = "recordings-screen"
        self._refresh_recordings()

    def _back_from_recordings(self) -> None:
        self._safe_stop_playback()
        if self.is_live:
            self._switch_screen("live-screen")
            self._screen = "live"
        else:
            self._switch_screen("select-screen")
            self._screen = "select"

    def _refresh_recordings(self) -> None:
        lv = self.query_one("#recordings-list", ListView)
        lv.clear()
        rec_dir = os.environ.get("RECORDINGS_DIR", ".")
        self._recording_files = sorted(glob.glob(os.path.join(rec_dir, "*.wav")), reverse=True)
        if not self._recording_files:
            lv.append(ListItem(Label("[dim]No recordings found[/]")))
        else:
            for f in self._recording_files:
                info = sf.info(f)
                duration = info.frames / info.samplerate
                size_kb = os.path.getsize(f) / 1024
                lv.append(ListItem(Label(f"[bold]{f}[/]  ({duration:.1f}s, {size_kb:.0f} KB)")))

    def _get_selected_file(self) -> str | None:
        lv = self.query_one("#recordings-list", ListView)
        idx = lv.index
        if idx is None or not self._recording_files:
            return None
        if idx < len(self._recording_files):
            return self._recording_files[idx]
        return None

    def _play_selected(self) -> None:
        filepath = self._get_selected_file()
        if not filepath:
            self.notify("Select a recording first", severity="warning")
            return
        self._safe_stop_playback()
        self._play_file(filepath)

    @work(thread=True)
    def _play_file(self, filepath: str) -> None:
        import subprocess as sp

        self._playback_stop = False
        data, sr = sf.read(filepath, dtype="float32")
        total_frames = len(data)

        bar = self.query_one("#playback-bar", PlaybackBar)
        bar.filename = filepath
        bar.playing = True
        bar.progress = 0.0

        # Convert to s16le for paplay/aplay
        pcm = (data * 32767).astype(np.int16).tobytes()

        if self.backend.use_pulse:
            cmd = ["paplay", "--raw", "--channels=1", f"--rate={sr}", "--format=s16le"]
        else:
            try:
                import sounddevice as sd
                sd.play(data, sr)
                while sd.get_stream().active and not self._playback_stop:
                    pos = int(sd.get_stream().time * sr)
                    bar.progress = min(pos / total_frames, 1.0)
                    bar.refresh()
                    time.sleep(0.05)
                sd.stop()
                bar.playing = False
                bar.progress = 1.0 if not self._playback_stop else bar.progress
                bar.refresh()
                return
            except Exception:
                cmd = ["aplay", "-f", "S16_LE", "-r", str(sr), "-c", "1"]

        self._playback_proc = sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE)
        chunk_size = sr * 2  # 1 second of s16le
        pos = 0

        while pos < len(pcm) and not self._playback_stop:
            end = min(pos + chunk_size, len(pcm))
            try:
                self._playback_proc.stdin.write(pcm[pos:end])
            except BrokenPipeError:
                break
            pos = end
            bar.progress = min(pos / len(pcm), 1.0)
            bar.refresh()

        if self._playback_proc.stdin:
            try:
                self._playback_proc.stdin.close()
            except Exception:
                pass
        self._playback_proc.wait()
        self._playback_proc = None
        bar.playing = False
        bar.progress = 1.0 if not self._playback_stop else bar.progress
        bar.refresh()

    def _safe_stop_playback(self) -> None:
        self._playback_stop = True
        proc = getattr(self, "_playback_proc", None)
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._playback_proc = None
        try:
            bar = self.query_one("#playback-bar", PlaybackBar)
            bar.playing = False
            bar.refresh()
        except Exception:
            pass

    def _delete_selected(self) -> None:
        filepath = self._get_selected_file()
        if not filepath:
            self.notify("Select a recording first", severity="warning")
            return
        self._safe_stop_playback()
        try:
            os.remove(filepath)
            self.notify(f"Deleted {filepath}", severity="information")
        except OSError as e:
            self.notify(f"Error: {e}", severity="error")
        self._refresh_recordings()

    # ── Cleanup ──────────────────────────────────────────────────────

    def on_unmount(self) -> None:
        self._cleanup()

    def action_quit(self) -> None:
        self._cleanup()
        self.exit()

    def _cleanup(self) -> None:
        self.is_live = False
        self.is_recording = False
        self._safe_stop_playback()
        for s in self.streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self.streams.clear()


if __name__ == "__main__":
    app = AudioMixerApp()
    app.run()
