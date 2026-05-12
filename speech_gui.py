"""
Speech Enhancement GUI
======================
A simple Tkinter interface for the speech enhancement pipeline:
  Noisy Audio → CNN Denoiser → wav2vec 2.0 → Transcript

Run this file alongside your notebook environment where the pipeline
functions (enhance_waveform, transcribe_waveform, add_noise, etc.) are
already defined, OR run it standalone — it imports from the notebook
logic via a helper module (see USAGE below).

USAGE (standalone):
    python speech_gui.py

WORKFLOW:
    1. Train the model in the notebook
    2. Run the new "Save Model" cell — it writes best_enhancer.pt
    3. Launch this GUI — it auto-detects best_enhancer.pt if in the same folder
    4. Click "Load Models", then browse an audio file and run the pipeline

The GUI exposes:
  1. Load Audio — pick any WAV/FLAC file from disk
  2. Noise Level — Low / Medium / High
  3. Run Pipeline — enhance + transcribe
  4. Results panel — shows transcripts + metrics
  5. Waveform viewer — plots clean / noisy / enhanced
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # Fix for protobuf descriptor error
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ── Optional: matplotlib for waveform plots ──────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Optional: numpy / torch for the actual pipeline ──────────────────────────
torch_import_error = ""
try:
    import numpy as np
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError as e:
    HAS_TORCH = False
    torch_import_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (dark, technical, minimal)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#0f1117"
PANEL    = "#1a1d27"
BORDER   = "#2a2d3a"
ACCENT   = "#4f9cf9"
ACCENT2  = "#7ee787"
WARN     = "#f97316"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"
ENTRY_BG = "#252836"
FONT_MONO = ("Courier New", 10)
FONT_BODY = ("Segoe UI", 10) if sys.platform == "win32" else ("Helvetica Neue", 10)
FONT_HEAD = ("Segoe UI Semibold", 11) if sys.platform == "win32" else ("Helvetica Neue", 11)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (copied from notebook so GUI is fully self-contained)
# ──────────────────────────────────────────────────────────────────────────────
if HAS_TORCH:
    import torch.nn as nn

    N_FFT      = 512
    HOP_LENGTH = 128
    WIN_LENGTH = 512
    SR         = 16000

    class TinySpectrogramDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16,  1, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.net(x)

# Global model handles (populated by _load_models_thread)
_enhancer_model = None
_asr_processor  = None
_asr_model_obj  = None
_device         = None
_models_loaded  = False

def _init_models(weights_path=None, use_finetuned_asr=True, log_fn=print):
    """Load TinySpectrogramDenoiser + wav2vec2 into globals.
    weights_path: .pt file saved with torch.save(model.state_dict(), ...)
                  If None the enhancer acts as identity (no denoising weights).
    """
    global _enhancer_model, _asr_processor, _asr_model_obj, _device, _models_loaded
    if not HAS_TORCH:
        raise RuntimeError(f"PyTorch not installed. Error: {torch_import_error}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fn(f"  Device: {_device}")

    # Enhancement model
    _enhancer_model = TinySpectrogramDenoiser().to(_device)
    if weights_path and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=_device)
        _enhancer_model.load_state_dict(state)
        log_fn(f"  Enhancer weights loaded: {os.path.basename(weights_path)}")
    else:
        log_fn("  No weights file found — enhancer will pass audio through unchanged.")
    _enhancer_model.eval()

    # ASR model
    log_fn("  Loading facebook/wav2vec2-base-960h  (downloads ~360 MB on first run)...")
    from transformers import AutoProcessor, AutoModelForCTC
    _asr_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    _asr_model_obj = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(_device)
    
    # Try to load fine-tuned ASR layers if available
    if weights_path and os.path.isfile(weights_path) and use_finetuned_asr:
        model_dir = os.path.dirname(weights_path)
        asr_weights = os.path.join(model_dir, "wav2vec2_finetuned_layers.pth")
        if os.path.isfile(asr_weights):
            log_fn(f"  Loading fine-tuned ASR weights: {os.path.basename(asr_weights)}")
            asr_state = torch.load(asr_weights, map_location=_device)
            _asr_model_obj.load_state_dict(asr_state, strict=False)
            log_fn("  ASR fine-tuned layers loaded.")

    _asr_model_obj.eval()
    log_fn("  ASR model ready.")
    _models_loaded = True

# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _load_waveform(path, sr=16000):
    """Load audio file, return mono float32 tensor at sr Hz.
    Tries soundfile -> sox_io -> torchaudio default -> soundfile direct -> librosa.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch / torchaudio not installed.")

    wav, orig_sr = None, None

    for backend in ("soundfile", "sox_io"):
        try:
            wav, orig_sr = torchaudio.load(path, backend=backend)
            break
        except Exception:
            pass

    if wav is None:
        try:
            wav, orig_sr = torchaudio.load(path)
        except Exception:
            pass

    if wav is None:
        try:
            import soundfile as sf
            data, orig_sr = sf.read(path, always_2d=True)
            wav = torch.tensor(data.T, dtype=torch.float32)
        except Exception:
            pass

    if wav is None:
        try:
            import librosa
            data, orig_sr = librosa.load(path, sr=None, mono=False)
            if data.ndim == 1:
                data = data[np.newaxis, :]
            wav = torch.tensor(data, dtype=torch.float32)
        except Exception:
            pass

    if wav is None:
        raise RuntimeError(
            "Could not open audio file with any available backend.\n"
            "Install soundfile:  pip install soundfile"
        )

    wav = wav.mean(dim=0)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    wav = wav / (wav.abs().max() + 1e-8)
    return wav

def _add_noise_internal(clean, level="medium"):
    snr_map = {"low": 17.5, "medium": 7.5, "high": -5.5}
    snr_db  = snr_map.get(level, 7.5)
    noise   = torch.randn_like(clean)
    noise   = noise / (noise.abs().max() + 1e-8)
    clean_p = torch.mean(clean ** 2) + 1e-8
    noise_p = torch.mean(noise ** 2) + 1e-8
    scale   = torch.sqrt(clean_p / noise_p / (10 ** (snr_db / 10.0)))
    noisy   = clean + scale * noise
    return noisy / (noisy.abs().max() + 1e-8)

def _enhance_waveform(clean, level="medium"):
    """Add noise then run the CNN enhancer. Returns (noisy, enhanced)."""
    noisy = _add_noise_internal(clean, level)

    if not _models_loaded or _enhancer_model is None:
        return noisy, noisy   # identity fallback

    window = torch.hann_window(WIN_LENGTH).to(_device)

    def _stft(w):
        return torch.stft(w.to(_device), n_fft=N_FFT, hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH, window=window, return_complex=True)

    with torch.no_grad():
        noisy_dev  = noisy.to(_device)
        noisy_spec = _stft(noisy_dev)
        noisy_mag  = noisy_spec.abs().unsqueeze(0).unsqueeze(0)
        pred_mag   = _enhancer_model(noisy_mag).squeeze(0).squeeze(0)
        phase      = torch.angle(noisy_spec)
        enh_spec   = torch.polar(pred_mag, phase)
        enhanced   = torch.istft(enh_spec, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                 win_length=WIN_LENGTH, window=window,
                                 length=noisy_dev.shape[-1])
        enhanced   = enhanced / (enhanced.abs().max() + 1e-8)

    return noisy, enhanced.detach().cpu()

def _transcribe_wave(wave):
    if not _models_loaded or _asr_processor is None:
        return "[Models not loaded]"
    arr    = wave.detach().cpu().numpy()
    inputs = _asr_processor(arr, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = _asr_model_obj(inputs.input_values.to(_device)).logits
    ids  = torch.argmax(logits, dim=-1)
    return _asr_processor.batch_decode(ids)[0].lower().strip()

def _compute_snr(clean, estimate):
    c = clean.numpy(); e = estimate.numpy()
    noise = c - e
    return 10.0 * float(np.log10((np.sum(c**2) + 1e-8) / (np.sum(noise**2) + 1e-8)))

def _compute_wer(reference, hypothesis):
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    d = [[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1]==hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(ref)][len(hyp)] / max(1, len(ref))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GUI CLASS
# ─────────────────────────────────────────────────────────────────────────────
class SpeechGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Speech Enhancement Pipeline")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("940x720")
        self.minsize(760, 580)

        # State
        self._audio_path = tk.StringVar(value="No file selected")
        self._noise_level = tk.StringVar(value="medium")
        self._reference   = tk.StringVar(value="")
        self._status      = tk.StringVar(value="Ready")
        self._waves       = {}   # {"clean": tensor, "noisy": tensor, "enhanced": tensor}

        self._build_styles()
        self._build_ui()
        self._auto_detect_weights()

    # ── Styles ────────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",       background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel",       background=BG,    foreground=TEXT, font=FONT_BODY)
        s.configure("Head.TLabel",  background=PANEL, foreground=ACCENT, font=FONT_HEAD)
        s.configure("Sub.TLabel",   background=PANEL, foreground=MUTED, font=FONT_BODY)
        s.configure("Val.TLabel",   background=PANEL, foreground=ACCENT2, font=FONT_MONO)
        s.configure("TRadiobutton", background=PANEL, foreground=TEXT,
                    selectcolor=PANEL, font=FONT_BODY)
        s.configure("Accent.TButton",
                    background=ACCENT, foreground="#ffffff",
                    font=("Segoe UI Semibold", 10) if sys.platform=="win32" else ("Helvetica Neue", 10),
                    padding=(14, 7), relief="flat")
        s.map("Accent.TButton",
              background=[("active", "#3b82f6"), ("disabled", BORDER)],
              foreground=[("disabled", MUTED)])
        s.configure("TProgressbar", troughcolor=BORDER, background=ACCENT,
                    thickness=4)
        s.configure("TSeparator", background=BORDER)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Title bar ────────────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=PANEL, height=52)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        tk.Label(title_bar, text="◈  Speech Enhancement Pipeline",
                 bg=PANEL, fg=ACCENT,
                 font=("Segoe UI Semibold", 13) if sys.platform=="win32" else ("Helvetica Neue", 13)
                 ).pack(side="left", padx=18, pady=14)
        tk.Label(title_bar, text="wav2vec 2.0 + CNN Denoiser",
                 bg=PANEL, fg=MUTED, font=FONT_BODY).pack(side="right", padx=18)

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # ── Main content ─────────────────────────────────────────────────────
        content = ttk.Frame(self, style="TFrame", padding=16)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=0, minsize=260)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        self._build_left_panel(content)
        self._build_right_panel(content)

        # ── Status bar ───────────────────────────────────────────────────────
        status_bar = tk.Frame(self, bg=BORDER, height=28)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self._status,
                 bg=BORDER, fg=MUTED, font=FONT_BODY, anchor="w"
                 ).pack(side="left", padx=10, pady=4)
        self._progress = ttk.Progressbar(status_bar, mode="indeterminate",
                                         style="TProgressbar", length=120)
        self._progress.pack(side="right", padx=10, pady=8)

    # ── Left control panel ────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        lf = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        def section(label):
            tk.Label(lf, text=label, bg=PANEL, fg=MUTED,
                     font=("Segoe UI Semibold", 8) if sys.platform=="win32" else ("Helvetica Neue", 8)
                     ).pack(anchor="w", pady=(14, 3))
            ttk.Separator(lf, orient="horizontal").pack(fill="x", pady=(0, 8))

        # ── Audio file ───────────────────────────────────────────────────────
        section("AUDIO INPUT")
        ttk.Button(lf, text="Browse File…", style="Accent.TButton",
                   command=self._browse_file).pack(fill="x")
        tk.Label(lf, textvariable=self._audio_path, bg=PANEL, fg=MUTED,
                 font=FONT_MONO, wraplength=220, justify="left"
                 ).pack(anchor="w", pady=(6, 0))

        # ── Noise level ──────────────────────────────────────────────────────
        section("NOISE LEVEL")
        for val, label, snr in [("low",    "Low     (15–20 dB SNR)", "#7ee787"),
                                 ("medium", "Medium   (5–10 dB SNR)", ACCENT),
                                 ("high",   "High    (−8 to −3 dB)",  WARN)]:
            rb = tk.Radiobutton(lf, text=label, variable=self._noise_level,
                                value=val, bg=PANEL, fg=snr,
                                selectcolor=PANEL, activebackground=PANEL,
                                font=FONT_BODY, cursor="hand2")
            rb.pack(anchor="w")

        # ── Reference transcript ─────────────────────────────────────────────
        section("REFERENCE TEXT  (optional)")
        self._ref_entry = tk.Text(lf, height=3, width=28,
                                  bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                                  font=FONT_MONO, relief="flat",
                                  padx=6, pady=6, wrap="word")
        self._ref_entry.pack(fill="x")

        # ── Models ──────────────────────────────────────────────────────────
        section("MODELS")
        ttk.Button(lf, text="Browse Weights (.pt)…",
                   command=self._browse_weights).pack(fill="x", pady=(0, 4))
        self._weights_label = tk.Label(lf, text="Looking for exported models...",
                                       bg=PANEL, fg=MUTED, font=FONT_MONO,
                                       wraplength=220, justify="left")
        self._weights_label.pack(anchor="w", pady=(0, 4))
        
        self._use_finetuned_asr = tk.BooleanVar(value=True)
        tk.Checkbutton(lf, text="Load fine-tuned ASR (if found)", 
                       variable=self._use_finetuned_asr,
                       bg=PANEL, fg=TEXT, selectcolor=ENTRY_BG, activebackground=PANEL).pack(anchor="w", pady=(0, 6))

        self._load_btn = ttk.Button(lf, text="⬇  Load Models",
                                    style="Accent.TButton",
                                    command=self._load_models_action)
        self._load_btn.pack(fill="x", pady=(0, 2))
        self._model_status = tk.Label(lf, text="● Not loaded", bg=PANEL, fg=WARN,
                                      font=FONT_BODY)
        self._model_status.pack(anchor="w")

        # ── Run button ───────────────────────────────────────────────────────
        section("ACTION")
        self._run_btn = ttk.Button(lf, text="▶  Run Pipeline",
                                   style="Accent.TButton",
                                   command=self._run_pipeline)
        self._run_btn.pack(fill="x", pady=(0, 4))
        ttk.Button(lf, text="Clear Results",
                   command=self._clear).pack(fill="x")

        # ── Metrics panel ────────────────────────────────────────────────────
        section("METRICS")
        self._metric_labels = {}
        for key in ("SNR (noisy)", "SNR (enhanced)", "WER (noisy)", "WER (enhanced)"):
            row = tk.Frame(lf, bg=PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=key, bg=PANEL, fg=MUTED,
                     font=FONT_BODY, width=16, anchor="w").pack(side="left")
            val_lbl = tk.Label(row, text="—", bg=PANEL, fg=ACCENT2,
                               font=FONT_MONO, anchor="e")
            val_lbl.pack(side="right")
            self._metric_labels[key] = val_lbl

    # ── Right results panel ───────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        rf = ttk.Frame(parent, style="TFrame")
        rf.grid(row=0, column=1, sticky="nsew")
        rf.rowconfigure(0, weight=0)
        rf.rowconfigure(1, weight=1)
        rf.rowconfigure(2, weight=2)
        rf.columnconfigure(0, weight=1)

        # ── Transcript cards ─────────────────────────────────────────────────
        cards = tk.Frame(rf, bg=BG)
        cards.grid(row=0, column=0, sticky="ew")
        cards.columnconfigure((0, 1, 2), weight=1)

        self._transcript_boxes = {}
        configs = [
            ("clean",    "CLEAN TRANSCRIPT",    "#7ee787"),
            ("noisy",    "NOISY TRANSCRIPT",    WARN),
            ("enhanced", "ENHANCED TRANSCRIPT", ACCENT),
        ]
        for col, (key, title, colour) in enumerate(configs):
            card = tk.Frame(cards, bg=PANEL, padx=10, pady=10)
            card.grid(row=0, column=col, sticky="nsew",
                      padx=(0 if col==0 else 6, 0), pady=(0, 10))
            tk.Label(card, text=title, bg=PANEL, fg=colour,
                     font=("Segoe UI Semibold", 8) if sys.platform=="win32" else ("Helvetica Neue", 8)
                     ).pack(anchor="w")
            ttk.Separator(card, orient="horizontal").pack(fill="x", pady=(4, 6))
            txt = scrolledtext.ScrolledText(
                card, height=5, width=24, bg=ENTRY_BG, fg=TEXT,
                insertbackground=TEXT, font=FONT_MONO, relief="flat",
                padx=6, pady=6, wrap="word", state="disabled")
            txt.pack(fill="both", expand=True)
            self._transcript_boxes[key] = txt

        # ── Log console ──────────────────────────────────────────────────────
        log_frame = tk.Frame(rf, bg=PANEL, padx=10, pady=10)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        tk.Label(log_frame, text="PIPELINE LOG", bg=PANEL, fg=MUTED,
                 font=("Segoe UI Semibold", 8) if sys.platform=="win32" else ("Helvetica Neue", 8)
                 ).pack(anchor="w")
        ttk.Separator(log_frame, orient="horizontal").pack(fill="x", pady=(4, 6))
        self._log = scrolledtext.ScrolledText(
            log_frame, height=6, bg=ENTRY_BG, fg=ACCENT2,
            insertbackground=TEXT, font=FONT_MONO, relief="flat",
            padx=6, pady=6, state="disabled")
        self._log.pack(fill="both", expand=True)

        # ── Waveform plot ────────────────────────────────────────────────────
        if HAS_MPL:
            plot_frame = tk.Frame(rf, bg=PANEL, padx=10, pady=10)
            plot_frame.grid(row=2, column=0, sticky="nsew")
            tk.Label(plot_frame, text="WAVEFORMS", bg=PANEL, fg=MUTED,
                     font=("Segoe UI Semibold", 8) if sys.platform=="win32" else ("Helvetica Neue", 8)
                     ).pack(anchor="w")
            ttk.Separator(plot_frame, orient="horizontal").pack(fill="x", pady=(4, 6))
            self._fig = Figure(figsize=(7, 2.8), dpi=96, facecolor=PANEL)
            self._fig.subplots_adjust(hspace=0.55, left=0.06, right=0.98,
                                      top=0.92, bottom=0.12)
            self._axes = self._fig.subplots(1, 3)
            for ax in self._axes: ax.set_facecolor(ENTRY_BG)
            self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
            self._canvas.get_tk_widget().pack(fill="both", expand=True)
            self._draw_placeholder_waveforms()
        else:
            tk.Label(rf, text="Install matplotlib for waveform plots",
                     bg=PANEL, fg=MUTED, font=FONT_BODY
                     ).grid(row=2, column=0, sticky="nsew")

    # ─────────────────────────────────────────────────────────────────────────
    # ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    def _auto_detect_weights(self):
        """Look for exported models next to this script and pre-fill the path."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "speech_enhancement_model", "enhancement_model_joint.pth"),
            os.path.join(script_dir, "speech_enhancement_model", "enhancement_model.pth"),
            os.path.join(script_dir, "best_enhancer.pt")
        ]
        
        for candidate in candidates:
            if os.path.isfile(candidate):
                self._weights_path = candidate
                self._weights_label.config(
                    text=f"Auto-detected: {os.path.basename(candidate)}", fg=ACCENT2)
                self._log_msg(f"Auto-detected weights: {candidate}")
                self._log_msg("Click 'Load Models' to initialise the pipeline.")
                return

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.flac *.mp3 *.ogg"), ("All", "*.*")]
        )
        if path:
            self._audio_path.set(os.path.basename(path))
            self._full_path = path
            self._log_msg(f"Loaded: {path}")
        else:
            self._full_path = None

    def _browse_weights(self):
        path = filedialog.askopenfilename(
            title="Select model weights",
            filetypes=[("PyTorch weights", "*.pt *.pth"), ("All", "*.*")]
        )
        if path:
            self._weights_path = path
            self._weights_label.config(text=os.path.basename(path), fg=ACCENT2)
        # else keep previous selection

    def _load_models_action(self):
        self._load_btn.state(["disabled"])
        self._model_status.config(text="● Loading...", fg=ACCENT)
        self._progress.start(12)
        self._status.set("Loading models...")
        weights = getattr(self, "_weights_path", None)
        use_asr = getattr(self, "_use_finetuned_asr", tk.BooleanVar(value=True)).get()
        threading.Thread(target=self._load_models_thread,
                         args=(weights, use_asr), daemon=True).start()

    def _load_models_thread(self, weights_path, use_asr):
        try:
            _init_models(weights_path=weights_path, use_finetuned_asr=use_asr, log_fn=self._log_msg)
            self.after(0, lambda: self._model_status.config(
                text="● Models ready", fg=ACCENT2))
            self.after(0, lambda: self._status.set("Models loaded."))
        except Exception as exc:
            self._log_msg(f"[ERROR loading models] {exc}")
            self.after(0, lambda: self._model_status.config(
                text="● Load failed", fg="#ef4444"))
            self.after(0, lambda: self._status.set("Model load failed."))
        finally:
            self.after(0, self._progress.stop)
            self.after(0, lambda: self._load_btn.state(["!disabled"]))

    def _run_pipeline(self):
        path = getattr(self, "_full_path", None)
        if not path or not os.path.exists(path):
            messagebox.showwarning("No file", "Please select a valid audio file first.")
            return
        if not HAS_TORCH:
            messagebox.showerror("Missing deps",
                                 "PyTorch / torchaudio are required to run the pipeline.")
            return
        if not _models_loaded:
            if not messagebox.askyesno("Models not loaded",
                    "Models have not been loaded yet.\n\n"
                    "Transcription will show placeholder text.\n"
                    "Click \'Load Models\' first for real results.\n\n"
                    "Run anyway?"):
                return
        self._run_btn.state(["disabled"])
        self._progress.start(12)
        self._status.set("Running pipeline…")
        threading.Thread(target=self._pipeline_thread,
                         args=(path, self._noise_level.get()), daemon=True).start()

    def _pipeline_thread(self, path, level):
        try:
            self._log_msg(f"\n── Starting pipeline ({'noise=' + level}) ──")

            # 1. Load audio
            self._log_msg("Step 1/3  Loading audio…")
            clean = _load_waveform(path)
            self._log_msg(f"  → {len(clean)/16000:.2f} s  |  {len(clean):,} samples  |  16 kHz")

            # 2. Enhance
            self._log_msg("Step 2/3  Enhancing…")
            noisy, enhanced = _enhance_waveform(clean, level)
            snr_n = _compute_snr(clean, noisy)
            snr_e = _compute_snr(clean, enhanced)
            self._log_msg(f"  → SNR noisy: {snr_n:.1f} dB  |  SNR enhanced: {snr_e:.1f} dB")

            # 3. Transcribe
            self._log_msg("Step 3/3  Transcribing…")
            hyp_clean = _transcribe_wave(clean)
            hyp_noisy = _transcribe_wave(noisy)
            hyp_enh   = _transcribe_wave(enhanced)
            self._log_msg(f"  → Clean   : {hyp_clean}")
            self._log_msg(f"  → Noisy   : {hyp_noisy}")
            self._log_msg(f"  → Enhanced: {hyp_enh}")

            # Optional WER
            ref = self._ref_entry.get("1.0", "end").strip()
            wer_n = _compute_wer(ref, hyp_noisy)   if ref else None
            wer_e = _compute_wer(ref, hyp_enh)     if ref else None

            # Update UI on main thread
            self.after(0, self._update_results,
                       clean, noisy, enhanced,
                       hyp_clean, hyp_noisy, hyp_enh,
                       snr_n, snr_e, wer_n, wer_e)
        except Exception as exc:
            self._log_msg(f"\n[ERROR] {exc}")
            self.after(0, self._pipeline_done)

    def _update_results(self, clean, noisy, enhanced,
                        hyp_clean, hyp_noisy, hyp_enh,
                        snr_n, snr_e, wer_n, wer_e):
        # Transcripts
        for key, text in [("clean", hyp_clean),
                           ("noisy", hyp_noisy),
                           ("enhanced", hyp_enh)]:
            box = self._transcript_boxes[key]
            box.config(state="normal")
            box.delete("1.0", "end")
            box.insert("end", text)
            box.config(state="disabled")

        # Metrics
        self._metric_labels["SNR (noisy)"].config(text=f"{snr_n:.1f} dB")
        self._metric_labels["SNR (enhanced)"].config(text=f"{snr_e:.1f} dB")
        self._metric_labels["WER (noisy)"].config(text=f"{wer_n:.3f}" if wer_n is not None else "—")
        self._metric_labels["WER (enhanced)"].config(text=f"{wer_e:.3f}" if wer_e is not None else "—")

        # Waveform plot
        self._waves = {"clean": clean, "noisy": noisy, "enhanced": enhanced}
        if HAS_MPL:
            self._draw_waveforms(clean, noisy, enhanced)

        self._pipeline_done()

    def _pipeline_done(self):
        self._progress.stop()
        self._run_btn.state(["!disabled"])
        self._status.set("Done.")

    def _clear(self):
        for box in self._transcript_boxes.values():
            box.config(state="normal"); box.delete("1.0", "end"); box.config(state="disabled")
        for lbl in self._metric_labels.values():
            lbl.config(text="—")
        if HAS_MPL:
            self._draw_placeholder_waveforms()
        self._log_msg("\n── Results cleared ──")
        self._status.set("Ready")

    # ─────────────────────────────────────────────────────────────────────────
    # WAVEFORM PLOTS
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_waveforms(self, clean, noisy, enhanced):
        sr = 16000
        configs = [
            (clean,    "Clean",    "#7ee787"),
            (noisy,    "Noisy",    "#f97316"),
            (enhanced, "Enhanced", "#4f9cf9"),
        ]
        for ax, (wav, title, colour) in zip(self._axes, configs):
            ax.clear()
            ax.set_facecolor(ENTRY_BG)
            t = np.arange(len(wav)) / sr
            ax.plot(t, wav.numpy(), color=colour, linewidth=0.6, alpha=0.9)
            ax.set_title(title, color=colour, fontsize=8, pad=4)
            ax.set_xlim(0, t[-1])
            ax.tick_params(colors=MUTED, labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
        self._fig.patch.set_facecolor(PANEL)
        self._canvas.draw()

    def _draw_placeholder_waveforms(self):
        for ax, label, colour in zip(self._axes,
                                     ["Clean", "Noisy", "Enhanced"],
                                     ["#7ee787", "#f97316", "#4f9cf9"]):
            ax.clear()
            ax.set_facecolor(ENTRY_BG)
            ax.set_title(label, color=colour, fontsize=8, pad=4)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED, fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
        self._fig.patch.set_facecolor(PANEL)
        self._canvas.draw()

    # ─────────────────────────────────────────────────────────────────────────
    # LOG
    # ─────────────────────────────────────────────────────────────────────────
    def _log_msg(self, msg):
        def _do():
            self._log.config(state="normal")
            self._log.insert("end", msg + "\n")
            self._log.see("end")
            self._log.config(state="disabled")
        self.after(0, _do)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SpeechGUI()
    app.mainloop()
