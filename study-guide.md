# Complete Study Guide — Speech Enhancement + wav2vec 2.0 Project

> **Purpose**: This is your one-stop document to prepare for the TA discussion.
> Everything is here — the paper, the code, the results, the reasoning, and the answers to every question you might get asked.

---

# PART 0: DEEP LEARNING & SPEECH BASICS

> If you already know deep learning, skip to Part 1. If not, read this first — every concept used in the project is explained here.

---

## Neural Networks (The Foundation)

A neural network is a function that takes numbers in and produces numbers out. Between input and output, there are **layers** — each layer is a bunch of math operations (multiply by weights, add a bias, apply an activation). The network **learns** by adjusting those weights so the output gets closer to the correct answer.

**Training** = showing the network many examples, measuring how wrong it is (using a **loss function**), and adjusting weights to reduce the error (using **backpropagation** — which computes how each weight contributed to the error, then nudges it in the right direction).

**Inference** = using a trained network to make predictions on new data. No learning happens — weights are frozen.

---

## Key Building Blocks

### Convolutional Neural Network (CNN)

A CNN slides small **filters** (also called kernels) across the input to detect patterns. Think of it like a magnifying glass scanning an image:

- A 3×3 filter slides across a 2D input (like an image or a spectrogram)
- At each position, it multiplies the filter values by the input values and sums them up → one output number
- Different filters detect different patterns (edges, textures, shapes)
- Stacking multiple CNN layers lets the network detect increasingly complex patterns

**In our project**: The enhancement model is a CNN that scans noisy spectrograms to detect and remove noise patterns. wav2vec 2.0's feature encoder is also a CNN that processes raw audio waveforms.

**Conv2d(1→16, kernel=3×3)** means: takes 1 input channel, outputs 16 channels (16 different filters), each filter is 3×3 pixels. **padding=1** means we add a border of zeros so the output is the same size as the input.

### Transformer

A Transformer is a network architecture built around **self-attention** — the idea that each element in a sequence can "look at" every other element to understand context.

Example: In the sentence "The bank of the river", the word "bank" needs to look at "river" to know it means riverbank, not a financial bank. Self-attention lets each position gather information from all other positions.

**Multi-head attention** = running multiple attention operations in parallel, each focusing on different types of relationships.

**In our project**: wav2vec 2.0 uses a Transformer (12 layers in BASE) to build context-aware representations of speech. Each audio frame can attend to every other frame in the utterance.

### Activation Functions

After each layer's math (multiply + add), we apply a non-linear function. Without this, stacking layers would be pointless (multiple linear operations collapse into one).

- **ReLU**: `output = max(0, input)` — if the value is negative, make it zero. Simple, fast, most common.
- **GELU**: A smoother version of ReLU used in wav2vec 2.0. Instead of a hard cutoff at zero, it has a smooth curve.

**In our project**: Our enhancement CNN uses ReLU after each convolutional layer.

### Loss Functions

A loss function measures "how wrong is the network's prediction?" — training tries to minimize this number.

- **L1 Loss (Mean Absolute Error)**: Average of |predicted - actual| across all values. Treats all errors equally. Produces sharper results.
- **L2 Loss (Mean Squared Error)**: Average of (predicted - actual)². Penalizes big errors more heavily. Can produce blurry results.
- **CTC Loss**: Special loss for sequence problems where input and output have different lengths (explained below).
- **Contrastive Loss**: "Pick the right answer from multiple choices" (explained below).

**In our project**: We train the enhancement model with L1 loss (comparing predicted clean spectrogram to actual clean spectrogram). wav2vec 2.0 uses contrastive loss for pre-training and CTC loss for fine-tuning.

---

## Key Concepts

### Self-Supervised Learning

Training without human labels. The model creates its own learning task from the data:
- Mask part of the input → predict what's behind the mask
- wav2vec 2.0 masks random spans of audio features and must identify what was there
- This is like a fill-in-the-blank exercise, but for audio

**Why it matters**: You can use millions of hours of unlabeled audio (just raw recordings, no transcripts needed) to learn speech patterns. Then you only need a tiny bit of labeled data to fine-tune for ASR.

### Pre-training vs Fine-tuning

- **Pre-training**: Train on a massive unlabeled dataset to learn general patterns (wav2vec 2.0 pre-trains on 960h or 53,000h of raw audio)
- **Fine-tuning**: Take the pre-trained model and train it further on a smaller labeled dataset for a specific task (ASR in this case)

The pre-trained model already "understands" speech. Fine-tuning just teaches it to map that understanding to text.

### CTC (Connectionist Temporal Classification)

The problem: Audio has ~50 frames per second, but text has maybe 5 characters per second. How do you align them?

CTC solves this by:
1. At each audio frame, the model predicts a character OR a special "blank" token
2. The output might look like: `—h—ee—ll—ll—oo—` (where — is blank)
3. CTC collapses this by removing blanks and merging repeated characters → `hello`
4. The loss considers ALL possible alignments that produce the correct text and sums their probabilities

**In our project**: wav2vec 2.0's ASR head uses CTC. We don't train CTC ourselves — it's already baked into the pre-trained model.

### Contrastive Loss (Multiple Choice Test)

Used during wav2vec 2.0's pre-training:
1. Mask a span of audio features
2. The Transformer outputs a prediction for what should be there
3. The model is given K+1 options: 1 correct quantized code + K distractors (from elsewhere in the same utterance)
4. It must pick the correct one
5. Loss = the correct option should have the highest similarity score

This forces the model to learn meaningful representations — it must understand what speech "should" sound like at each position.

### Quantization (Codebook)

Turning continuous values into discrete codes. Like rounding 3.7 to 4, but in a learned high-dimensional space.

wav2vec 2.0 has a **codebook** — a table of, say, 320 learned "speech sound prototypes." Each audio frame gets matched to the closest prototype. This creates discrete speech units, used as targets in the contrastive task.

**Gumbel-Softmax** makes this differentiable (gradients can flow through the discrete selection) by adding random noise and using a temperature parameter that controls how "hard" the selection is.

### Spectrogram

A visual representation of audio showing **frequency vs time**:
- X-axis = time
- Y-axis = frequency (low sounds at bottom, high sounds at top)
- Color/brightness = how much energy (loud = bright)

Created using **STFT (Short-Time Fourier Transform)**: slide a window across the audio, compute the frequency content at each window position. The result is a complex number at each time-frequency point, which has:
- **Magnitude**: how loud that frequency is (what we see and denoise)
- **Phase**: the timing/alignment of the frequency wave (we borrow from noisy input)

**In our project**: We convert audio to spectrogram, denoise the magnitude, keep the noisy phase, and convert back.

### Encoder-Decoder Architecture

- **Encoder**: Compresses input into a compact representation (fewer but richer features)
- **Decoder**: Expands the representation back to the desired output

wav2vec 2.0's CNN is an encoder (waveform → compact features). Our enhancement CNN doesn't have a true encoder-decoder structure — it keeps the same spatial dimensions throughout (all layers use padding=1). A U-Net would be a true encoder-decoder with skip connections.

---

## Training Concepts Used in Our Project

### Optimizer (Adam)

An algorithm that decides how to update weights after computing the loss. Adam is the most popular — it adapts the learning rate for each weight individually based on how its gradients have been behaving. Faster and more stable than basic gradient descent.

### Learning Rate

How big of a step to take when updating weights. Too high → model overshoots and never converges. Too low → model learns painfully slowly.

Our starting LR: **1e-3** (0.001) — a standard choice for Adam.

### Learning Rate Scheduler (ReduceLROnPlateau)

Automatically reduces the learning rate when the model stops improving. Ours halves the LR if validation loss doesn't improve for 2 consecutive epochs. This lets the model make big steps early (exploring) and small steps later (fine-tuning the solution).

In our training: LR stayed at 1e-3 for epochs 1-8, then dropped to 5e-4 at epoch 9.

### Early Stopping

Stop training when the model stops improving on validation data, even if you haven't reached the maximum number of epochs. Prevents **overfitting** (model memorizes training data instead of learning general patterns).

Our patience: 3 epochs. If val loss doesn't improve for 3 straight epochs → stop.

### Checkpointing

Save the model weights at the best epoch (lowest validation loss). After training finishes (or early-stops), restore those best weights. This way you always use the best version of the model, not the last one.

In our training: Best checkpoint was epoch 6 (val L1 = 0.2515).

### Overfitting

When the model performs great on training data but poorly on new data. Signs: training loss keeps going down but validation loss goes up. Our early stopping + small model size (18K params) naturally prevent this.

### Batch Size

How many examples the model sees before updating weights once. Our batch size: 8 spectrograms at a time. Larger = more stable gradients but needs more GPU memory.

### Train / Validation / Test Split

- **Train** (3,200): Model learns from this
- **Validation** (400): Used during training to monitor performance and make decisions (early stopping, LR scheduling). Model never trains on this.
- **Test** (400): Used only at the end to measure final performance. Model never sees this during training.

---

## Speech-Specific Concepts

### ASR (Automatic Speech Recognition)

Converting audio to text. Also called speech-to-text or STT.

### Waveform

The raw audio signal — a 1D sequence of numbers representing air pressure over time. At 16,000 Hz (our sample rate), there are 16,000 numbers per second of audio.

### Sample Rate (SR)

How many audio samples per second. 16,000 Hz (16 kHz) is the standard for speech processing. This means each second of audio = 16,000 numbers.

### STFT (Short-Time Fourier Transform)

Converts a waveform into a spectrogram. Parameters in our project:
- **N_FFT = 512**: Window size for the Fourier transform (512 samples ≈ 32ms)
- **Hop length = 128**: How far the window slides between frames (128 samples ≈ 8ms)
- **Window = Hann**: A smooth bell-shaped window applied before each FFT to reduce artifacts at the edges

### Inverse STFT (iSTFT)

Converts a spectrogram back to a waveform. We use this after enhancement to get the denoised audio back into waveform form.

### Formants

Resonant frequencies of the vocal tract — the peaks in the spectrum that give vowels their distinct sounds. "ah" vs "ee" vs "oo" all have different formant patterns. When the enhancer over-smooths, it can blur these formants and make speech less clear.

---

# PART 1: THE PAPER (wav2vec 2.0)

## What is the paper about?

The paper solves one problem: **How to build great ASR (speech-to-text) without needing massive amounts of labeled data.**

Normally, to train a speech recognition system, you need thousands of hours of audio where someone manually typed out what was said. That's expensive and only exists for a few languages.

wav2vec 2.0 says: let the model first **learn from raw audio alone** (no labels), then fine-tune on a tiny bit of labeled data.

## How does wav2vec 2.0 work?

The model has three parts:

### Part 1: CNN Feature Encoder
- Takes raw audio waveform (16,000 samples per second)
- Seven convolutional layers process it
- Outputs one feature vector every ~20 milliseconds
- Think of it as: **raw sound → compressed representation**

### Part 2: Transformer Encoder
- Takes the CNN features as input
- 12 layers of self-attention (BASE model) or 24 layers (LARGE model)
- Each position can "look at" every other position in the utterance
- Builds **context-aware** representations (each vector knows about the full utterance)
- Think of it as: **local features → global understanding**

### Part 3: Quantization Module
- Takes CNN output and maps it to a **codebook** (like a dictionary of speech sounds)
- Uses Gumbel-Softmax to pick entries in a differentiable way
- These discrete codes become the **training targets**
- Think of it as: **continuous features → discrete speech units**

## How is it trained?

### Pre-training (self-supervised, no labels needed)
1. Take a raw audio clip
2. Pass it through the CNN encoder
3. **Mask** random spans of the CNN output (hide ~49% of time steps)
4. The Transformer tries to **predict** what's behind the mask
5. Specifically: given the context, identify the correct quantized code among distractors
6. This is a **contrastive loss** — like multiple choice: pick the right answer from K+1 options

The loss has two parts:
- **Contrastive loss** (Lm): identify the correct masked latent
- **Diversity loss** (Ld): encourage the model to use all codebook entries, not just a few

### Fine-tuning (needs labels)
1. Add a linear layer on top of the Transformer
2. Train with **CTC loss** (Connectionist Temporal Classification)
3. CTC handles alignment — the model doesn't need to know exactly when each character occurs
4. Fine-tune on as little as 10 minutes of labeled data

## What results does the paper achieve?

On LibriSpeech:

| Setup | test-clean WER | test-other WER |
|---|---|---|
| BASE, no LM | 3.4% | 8.5% |
| BASE + Transformer LM | 2.1% | 4.8% |
| LARGE + Transformer LM | 2.0% | 4.1% |
| LARGE + LV-60k + Transformer LM | **1.8%** | **3.3%** |

With only **10 minutes** of labeled data: 4.8% / 8.2% WER. That's insanely good.

## What does the paper NOT do?

**It never tests on noisy speech.** Every single result is on clean or naturally-varied LibriSpeech. The paper never asks: "What happens if there's background noise?" That's the gap we address.

---

# PART 2: OUR PROJECT

## What is our idea?

wav2vec 2.0 is great on clean speech but its CNN encoder was pre-trained on clean audio. When you feed it noisy audio, the convolutional filters produce garbage features, which propagate through the Transformer and ruin the transcription.

Instead of retraining wav2vec 2.0 on noisy data (would need thousands of GPU hours), we put a **cheap denoiser in front of it**:

```
Paper's pipeline:     Clean Speech  →  wav2vec 2.0  →  Text     ✓ works great
Real world:           Noisy Speech  →  wav2vec 2.0  →  Garbage  ✗ breaks badly
Our pipeline:         Noisy Speech  →  Denoiser     →  wav2vec 2.0  →  Better Text
```

## Why does this make sense?

- wav2vec 2.0's representations are powerful but **input-dependent**
- If the input is noisy, the representations are noisy
- By cleaning the input, we bring it closer to what the model was trained on
- This is like a **domain adaptation** without touching the model
- It's modular: works with any ASR system, not just wav2vec 2.0

---

# PART 3: WHAT WE BUILT (Technical Details)

## Dataset

- **LibriSpeech** — same benchmark as the paper
- 137,876 total utterances available
- We used **4,000** (3,200 train / 400 val / 400 test)
- Speaker-balanced sampling (max 14 clips per speaker) for diversity
- Official splits used → **zero speaker overlap** between train/val/test
- Why only 4,000? Kaggle T4 GPU has 16 GB VRAM and limited runtime

## Noise Generation

We add noise to clean speech to simulate real-world conditions. Two noise sources:

1. **Gaussian noise** — random white noise
2. **MUSAN noise** — 600 real background noise and music clips (from the MUSAN dataset)

The mix depends on noise level — higher noise uses more MUSAN (realistic) and less Gaussian:

| Level | SNR Range | Gaussian Weight | Background Weight | What it sounds like |
|---|---|---|---|---|
| Low | 15–20 dB | 0.65 | 0.35 | Quiet room, light hum |
| Medium | 5–10 dB | 0.45 | 0.55 | Busy cafe |
| High | -8 to -3 dB | 0.30 | 0.70 | Loud street |

**SNR (Signal-to-Noise Ratio)** = how much louder the speech is than the noise.
- 20 dB = speech is 100x more powerful than noise (very clean)
- 0 dB = speech and noise are equal power (very noisy)
- -5 dB = noise is stronger than speech (extremely noisy)

### How noise is actually added (the math)

1. Generate Gaussian noise + load a random MUSAN clip
2. Mix them with weights above
3. Calculate how much to scale the noise to hit the target SNR:
   - `target_noise_power = clean_power / 10^(SNR_dB / 10)`
   - Scale noise to match that power
4. Add: `noisy = clean + scaled_noise`
5. Normalize to [-1, 1]

## Enhancement Model: TinySpectrogramDenoiser

### Why spectrograms?

Raw waveform denoising is hard. Spectrograms show frequency vs time, making noise patterns visually clear and easier for a CNN to learn. We use STFT (Short-Time Fourier Transform) with:
- N_FFT = 512
- Hop length = 128
- Window = Hann, 512 samples

### Architecture

Five convolutional layers, all 3×3 with padding=1 (preserves dimensions):

```
Input (1 channel: noisy magnitude spectrogram)
  → Conv2d(1→16) + ReLU
  → Conv2d(16→32) + ReLU
  → Conv2d(32→32) + ReLU
  → Conv2d(32→16) + ReLU
  → Conv2d(16→1) + ReLU
Output (1 channel: predicted clean magnitude spectrogram)
```

**Total: 18,817 parameters** (wav2vec 2.0 BASE has 95,000,000 — ours is 5,000x smaller)

### Training

- **Loss**: L1 (mean absolute error between predicted and actual clean magnitude)
- **Optimizer**: Adam, learning rate 1e-3
- **Scheduler**: ReduceLROnPlateau — halves LR when validation loss stalls for 2 epochs
- **Early stopping**: Stops if no improvement for 3 epochs
- **Best checkpoint**: Saves the model from the best validation epoch and restores it

**What actually happened during training:**

| Epoch | Train L1 | Val L1 | LR | Note |
|---|---|---|---|---|
| 1 | 0.3289 | 0.3020 | 1e-3 | ★ best |
| 2 | 0.2951 | 0.2712 | 1e-3 | ★ best |
| 3 | 0.2828 | 0.2918 | 1e-3 | |
| 4 | 0.2722 | 0.2598 | 1e-3 | ★ best |
| 5 | 0.2733 | 0.2798 | 1e-3 | |
| 6 | 0.2624 | 0.2515 | 1e-3 | ★ best (kept this one) |
| 7 | 0.2669 | 0.2594 | 1e-3 | |
| 8 | 0.2613 | 0.2648 | 1e-3 | |
| 9 | 0.2595 | 0.2625 | 5e-4 | Early stopped here |

The scheduler kicked in at epoch 9 (reduced LR to 5e-4), and early stopping triggered because epochs 7, 8, 9 all failed to beat epoch 6's validation loss. Model weights from epoch 6 were restored.

### How enhancement actually works at inference

1. Take noisy waveform
2. Compute STFT → get complex spectrogram
3. Split into **magnitude** (what we denoise) and **phase** (kept as-is)
4. Pass noisy magnitude through the CNN → get predicted clean magnitude
5. Combine predicted magnitude + noisy phase → complex spectrogram
6. Inverse STFT → enhanced waveform

**Key limitation**: We borrow the **noisy phase**. Phase carries fine temporal detail. This puts a ceiling on quality. Fixing this would need a much more complex model.

## ASR Model

- **facebook/wav2vec2-base-960h** from Hugging Face
- wav2vec 2.0 BASE (95M params), pre-trained on 960h LibriSpeech, fine-tuned with CTC
- **Completely frozen** — we don't train or modify it at all
- Greedy CTC decoding (no language model)
- This matches the paper's "BASE, LS-960, no LM" setup → paper gets 3.4% WER, we get 3.67%

---

# PART 4: ALL RESULTS

## Main Result (Medium Noise)

| Metric | Clean | Noisy | Enhanced | Change |
|---|---|---|---|---|
| **WER** | 3.67% | 38.76% | **25.17%** | -13.59 pp (35% relative reduction) |
| **SNR** | — | 7.67 dB | **10.90 dB** | +3.23 dB |
| **PESQ** | — | 1.123 | **1.391** | +0.268 (higher = better, max ~4.5) |
| **STOI** | — | 0.848 | **0.812** | -0.036 (higher = better, max 1.0) |

### What each metric means — Deep Dive

---

#### WER — Word Error Rate

**What it measures**: The percentage of words the ASR system got wrong compared to the true transcript.

**How it works**: WER compares the reference transcript to the hypothesis (what the model produced) and counts three types of errors:
- **Substitutions (S)**: A word was replaced with a wrong word ("king" → "thing")
- **Deletions (D)**: A word was missed entirely ("the king said" → "king said")
- **Insertions (I)**: An extra word was added ("king said" → "the king said hello")

The formula: `WER = (S + D + I) / Total words in reference`

**Range**: 0% (perfect) to 100%+ (yes, it can exceed 100% if the model inserts many extra words)

**In our project**:
- Clean: 3.67% → nearly perfect, wav2vec 2.0 barely makes mistakes
- Noisy: 38.76% → roughly 4 out of 10 words are wrong
- Enhanced: 25.17% → roughly 1 in 4 words wrong — still not great, but much better than noisy

**Why we use it**: It's the **primary metric for ASR quality**. The wav2vec 2.0 paper reports all results in WER. It directly measures "can you read the transcript and understand it?"

**Library**: `jiwer` (Python package)

---

#### SNR — Signal-to-Noise Ratio

**What it measures**: How much louder the desired signal (speech) is compared to the noise, measured in decibels (dB).

**How it works**: We compare the energy (power) of the clean speech to the energy of the error (difference between clean and the signal being measured):

`SNR = 10 × log10(power of clean / power of (clean - estimate))`

A higher number means the signal is much louder than the noise (cleaner). A lower number means noise dominates.

**Scale (intuition)**:
- 30+ dB = studio quality, noise is nearly inaudible
- 20 dB = clean room, very faint background
- 10 dB = noise clearly present but speech still dominant
- 0 dB = speech and noise equally loud
- Negative dB = noise louder than speech

**In our project**:
- Noisy SNR: 7.67 dB → noise clearly audible, moderately corrupted
- Enhanced SNR: 10.90 dB → noise reduced, speech more dominant
- Improvement: +3.23 dB → meaningful; every 3 dB roughly doubles the signal-to-noise power ratio

**Why we use it**: It's the simplest signal-level metric. It tells us whether the enhancer actually removed noise energy, independent of how ASR interprets it.

**Computed by**: Our own function (no library needed — just numpy math)

---

#### PESQ — Perceptual Evaluation of Speech Quality

**What it measures**: How a human listener would rate the quality of the speech signal. It models human auditory perception.

**How it works**: PESQ (ITU-T P.862) is an **intrusive metric** — it needs both the clean reference and the degraded/enhanced signal. It:
1. Aligns the two signals in time (handles small delays)
2. Transforms both into an internal representation that models the human auditory system (bark scale, loudness mapping)
3. Computes the **disturbance** — the perceptual difference between what you hear in the degraded signal vs. the clean reference
4. Combines disturbances across time and frequency into a single score

Think of it as: "If a human listened to the clean version and then the noisy version, how annoyed would they be?"

**Range**: Approximately **1.0 to 4.5** (wideband mode, which we use)
- ~1.0 = very bad, heavily distorted
- ~2.0 = poor, clearly degraded
- ~3.0 = fair, some noticeable degradation
- ~4.0 = good, minor imperfections
- ~4.5 = excellent, nearly indistinguishable from original

**In our project**:
- Noisy PESQ: 1.123 → very poor quality (human would say "this sounds terrible")
- Enhanced PESQ: 1.391 → still poor but noticeably better
- The numbers are low overall because medium noise (5-10 dB) is genuinely harsh. State-of-the-art enhancers with millions of parameters typically reach ~2.5-3.0 at similar noise

**Why we use it**: WER tells us about the transcript quality, but PESQ tells us about the **audio quality itself**. You could have decent WER but terrible-sounding audio (or vice versa). PESQ confirms the enhancement actually makes the audio sound better to a human ear, not just to the ASR model.

**Library**: `pesq` (Python package, implements ITU-T P.862)

---

#### STOI — Short-Time Objective Intelligibility

**What it measures**: How intelligible (understandable) the speech is — specifically, can a listener make out the words?

**How it works**: STOI focuses on the **temporal envelope** of speech in short time windows:
1. Splits both clean and degraded signals into short overlapping frames (~386 ms)
2. For each frame, computes the energy in different frequency bands (1/3-octave bands)
3. Computes the **correlation** between the clean and degraded temporal envelopes in each band
4. Averages across all bands and frames

High correlation = the temporal pattern of speech energy is preserved = you can still tell what words are being said. Low correlation = the noise has disrupted the energy patterns = words become muddy and unclear.

Think of it as: "Are the speech patterns (when sound goes up, down, pauses) still recognizable, or has noise scrambled them?"

**Range**: **0 to 1**
- 0.0 = completely unintelligible (random noise)
- 0.5 = very poor intelligibility
- 0.75 = moderate intelligibility
- 0.85+ = good intelligibility
- 1.0 = perfect (identical to clean)

**In our project**:
- Noisy STOI: 0.848 → good intelligibility (humans could still mostly understand)
- Enhanced STOI: 0.812 → slightly worse!

**Why STOI went DOWN** (this is important to understand):

STOI is very sensitive to the **phase** of the signal. Our enhancer fixes the magnitude spectrogram but reuses the noisy phase. This creates a mismatch:
- The magnitude says "clean speech"
- The phase says "noisy signal"

STOI's temporal envelope correlation picks up on this inconsistency. The smoothing that our CNN applies to the magnitude also slightly blurs the temporal envelope, reducing the correlation further.

**The key insight**: STOI measures intelligibility to humans, but wav2vec 2.0 processes audio differently than humans. wav2vec 2.0's CNN encoder is more bothered by additive noise energy than by phase inconsistency, so WER improves even though STOI slightly decreases. This tells us something interesting about what wav2vec 2.0 is actually sensitive to.

**Library**: `pystoi` (Python package)

---

#### How the Four Metrics Work Together

| Metric | Measures | Level | Improved? | What it tells us |
|---|---|---|---|---|
| **WER** | Transcript accuracy | ASR output | Yes (+13.59 pp) | The enhancer helps the ASR model produce better text |
| **SNR** | Noise energy removal | Signal | Yes (+3.23 dB) | The enhancer physically removes noise from the waveform |
| **PESQ** | Perceptual quality | Signal (human ear model) | Yes (+0.268) | The enhanced audio sounds better to a human |
| **STOI** | Intelligibility | Signal (temporal patterns) | No (-0.036) | Phase mismatch hurts temporal coherence slightly |

Three out of four metrics improve. The one that doesn't (STOI) has a clear, explainable reason (phase borrowing). Using all four together gives a much more complete picture than any single metric alone. This multi-metric approach is what separates a thorough evaluation from a superficial one.

## Per-Noise-Level Results

| Level | WER Clean | WER Noisy | WER Enhanced | Improvement | Helped? |
|---|---|---|---|---|---|
| Low | 3.67% | 5.85% | 7.04% | **-1.18 pp** | No — hurts |
| Medium | 3.67% | 36.73% | 24.26% | **+12.47 pp** | Yes — strong |
| High | 3.67% | 97.60% | 92.17% | **+5.43 pp** | Yes — modest |

| Level | PESQ Noisy | PESQ Enhanced | STOI Noisy | STOI Enhanced |
|---|---|---|---|---|
| Low | 1.498 | 1.913 | 0.946 | 0.905 |
| Medium | 1.118 | 1.382 | 0.847 | 0.810 |
| High | 1.042 | 1.080 | 0.615 | 0.561 |

## Per-Sample Analysis

Out of 150 test samples:
- **86 (57%)** — enhancement helped (lower WER)
- **36 (24%)** — enhancement hurt (higher WER)
- **28 (19%)** — no change
- Mean WER improvement: **+0.083** per sample
- Std: 0.193

---

# PART 5: HOW TO EXPLAIN EVERY RESULT

## "Why does noise break wav2vec 2.0 so badly?"

The CNN feature encoder learned its convolutional filters from clean audio. Those filters expect clean spectral patterns. When noise is added, the filters activate on noise energy too, producing corrupted features. The Transformer then builds context over those corrupted features, compounding errors. By medium noise (5-10 dB), the model goes from 3.67% to 38.76% WER — a 10x degradation.

## "Why does your enhancement help?"

It brings the noisy input closer to the clean distribution the model was pre-trained on. The CNN removes noise energy from the spectrogram, so wav2vec 2.0's feature encoder sees something closer to clean speech. It's like wiping fog off a camera lens — the camera (wav2vec) hasn't changed, but the image it receives is clearer.

## "Why does enhancement HURT at low noise?"

At low noise (15-20 dB SNR), the speech is already nearly clean. wav2vec 2.0 handles it fine (5.85% WER). But the enhancer doesn't know the noise is mild — it still tries to denoise, and in doing so it **over-smooths** the spectrogram, removing subtle speech details. The cure is worse than the disease. WER goes up from 5.85% to 7.04%.

**If asked**: "In a real system, you'd add a noise detector that only activates the enhancer when SNR is below a threshold."

## "Why does STOI decrease even though everything else improves?"

STOI measures short-time intelligibility and is very sensitive to **phase**. Our model only fixes the magnitude spectrogram and reuses the noisy phase. This creates **magnitude-phase mismatches** — the magnitude says "clean" but the phase still says "noisy." STOI penalizes this inconsistency.

Meanwhile, wav2vec 2.0's CNN encoder is more bothered by additive noise energy than by phase inconsistency, so WER still improves. This is actually an interesting finding about what wav2vec 2.0 is sensitive to.

## "Why is 25% WER still high?"

Because our model is tiny (18K parameters) and operates under severe constraints:
- Magnitude-only processing (no phase correction)
- 5 conv layers, no skip connections
- Medium noise is genuinely challenging (speech is only 3-10x louder than noise)

State-of-the-art enhancement systems with millions of parameters achieve ~5-15% enhanced WER at similar noise levels. Our model is 100-1000x smaller. The point isn't achieving perfect denoising — it's showing that even minimal preprocessing meaningfully helps.

## "Why not just fine-tune wav2vec 2.0 on noisy data?"

Three reasons:
1. **Cost**: Pre-training wav2vec 2.0 takes days on 64-128 V100 GPUs. Fine-tuning still needs multiple GPUs.
2. **Data**: You'd need noisy-clean paired data or noisy transcribed data at scale.
3. **Modularity**: Our approach works with ANY ASR model. Fine-tuning is model-specific.

Our approach trains in minutes on a free T4 GPU. That's the practical tradeoff.

## "What's CTC and why do you use it?"

CTC (Connectionist Temporal Classification) is a loss function for sequence-to-sequence problems where you don't know the exact alignment between input and output. The audio might have 500 time frames but the text has 20 characters — CTC handles this by allowing the model to output "blank" tokens and collapsing repeated characters. wav2vec 2.0 uses CTC for its fine-tuned ASR head. We didn't choose it — the pre-trained model already uses it.

## "What's a contrastive loss?"

Imagine a multiple-choice test. The model sees the context around a masked position and must pick the correct quantized code from K+1 options (1 correct + K distractors from the same utterance). The loss encourages the correct answer's score to be higher than all distractors. This is what makes wav2vec 2.0's pre-training work without labels.

## "Why L1 loss for the enhancement model?"

L1 (mean absolute error) penalizes the average magnitude of errors between predicted and target spectrograms. Compared to L2 (mean squared error), L1 produces sharper spectrograms with less blurring. For speech enhancement, this matters because blurring smears formant transitions and reduces intelligibility.

## "What's the Gumbel-Softmax in the paper?"

It's a trick to make discrete choices differentiable. The quantization module needs to pick one codebook entry (discrete choice), but backpropagation needs gradients (continuous). Gumbel-Softmax adds noise to the logits and uses a temperature parameter to approximate a hard choice with a soft, differentiable one. During forward pass it picks the argmax; during backward pass it uses the soft gradient.

---

# PART 6: POTENTIAL TA QUESTIONS & ANSWERS

### Q: "Did you outperform the paper?"
**A**: No, and that's not the goal. The paper achieves 1.8-3.4% WER on clean speech. We don't compete with that. Our contribution is different: we showed that wav2vec 2.0 is vulnerable to noise (the paper never tested this) and proposed a cheap fix. We complement the paper, not compete with it.

### Q: "Why didn't you use a language model?"
**A**: To keep the comparison fair and the setup simple. The paper reports results both with and without language models. Without LM, their BASE model gets 3.4% — we get 3.67%, which is consistent. Adding an LM would improve all three conditions (clean, noisy, enhanced) but wouldn't change the relative improvement from enhancement.

### Q: "Could you use a better enhancement model?"
**A**: Yes. A U-Net with skip connections would preserve more detail. Complex-valued masks would fix the phase problem. Attention-based models could handle non-stationary noise better. We chose the simplest model that demonstrates the concept within Kaggle T4 constraints.

### Q: "Why not test on test-other?"
**A**: We used test-clean because we wanted to isolate the effect of our synthetic noise from natural recording variability. test-other already has harder conditions (accents, microphone quality), which would conflate two sources of degradation.

### Q: "What's novel about your work?"
**A**: Three things: (1) We quantified wav2vec 2.0's noise vulnerability, which the paper never did. (2) We showed a modular preprocessing approach works without modifying the ASR model. (3) We provided multi-metric evaluation (WER + SNR + PESQ + STOI) with per-sample failure analysis — going beyond aggregate numbers.

### Q: "Why do some samples get worse after enhancement?"
**A**: 36 out of 150 samples got worse. This happens when: (a) the original noise was mild and the enhancer over-smoothed the signal, (b) short utterances where one garbled word dominates WER, or (c) the enhancer removed speech energy along with noise. This is a known limitation of magnitude-only enhancement.

### Q: "What would you do differently with more compute?"
**A**: Four things: (1) Larger enhancement model (U-Net, ~500K params). (2) Phase-aware enhancement using complex masks. (3) Train on more data with more diverse noise types (reverb, competing speakers). (4) Compare against noise-augmented fine-tuning of wav2vec 2.0 as a stronger baseline.

### Q: "How does your clean WER compare to the paper?"
**A**: Paper's BASE with no LM: 3.4%. Ours: 3.67%. The small difference (0.27 pp) is because we evaluate on a 150-sample subset rather than the full 2,620-utterance test-clean set. Our setup is consistent with the paper.

### Q: "Why use LibriSpeech?"
**A**: Because it's the exact benchmark used in the wav2vec 2.0 paper. Using the same dataset lets us directly compare our clean-speech baseline to the paper's numbers and ensures our results are in the same domain.

---

# PART 7: KEY NUMBERS TO MEMORIZE

| What | Number |
|---|---|
| Our clean WER | **3.67%** |
| Paper's clean WER (same config) | **3.4%** |
| Medium noise WER | **38.76%** |
| Enhanced WER | **25.17%** |
| WER improvement | **13.59 pp (35% relative)** |
| SNR improvement | **+3.23 dB** |
| Enhancement model size | **18,817 parameters** |
| wav2vec 2.0 BASE size | **95,000,000 parameters** |
| Training epochs (early stopped) | **9 (best at epoch 6)** |
| Best validation L1 loss | **0.2515** |
| Samples helped / hurt / unchanged | **86 / 36 / 28** out of 150 |
| PESQ improvement | **1.123 → 1.391** |
| STOI change | **0.848 → 0.812** (slight decrease — phase issue) |

---

# PART 8: ONE-MINUTE SUMMARY

> "wav2vec 2.0 is a self-supervised model that learns speech representations from raw audio and achieves state-of-the-art ASR on clean speech. But it was only ever tested on clean data. We investigated what happens under noise and found it degrades dramatically — from 3.67% WER to 38.76% under medium noise. We proposed a lightweight CNN denoiser with only 18,000 parameters that preprocesses the audio before it reaches wav2vec 2.0. This reduced WER to 25.17% — a 35% relative improvement — validated by four metrics across three noise levels. The key insight is that even a tiny preprocessing step can meaningfully improve the robustness of powerful pre-trained ASR models, without modifying them at all."

That's your entire project in 30 seconds.
