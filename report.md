# Speech Enhancement as a Preprocessing Step for wav2vec 2.0 ASR Under Noisy Conditions

---

## 1. Paper Brief: wav2vec 2.0

### Problem

Building accurate automatic speech recognition (ASR) systems traditionally requires thousands of hours of transcribed speech. For most of the world's nearly 7,000 languages, that amount of labeled data simply does not exist. The question the paper tackles is: can we learn useful speech representations from raw, unlabeled audio alone, and then fine-tune on a small amount of labeled data to get competitive ASR?

### Core Idea

wav2vec 2.0 (Baevski et al., 2020) introduces a self-supervised framework that learns directly from raw 16 kHz waveforms. The architecture has three main components:

1. **CNN Feature Encoder** — a stack of seven temporal convolution blocks that converts the raw waveform into a sequence of latent speech representations, producing one vector roughly every 20 ms.
2. **Transformer Encoder** — 12 blocks (BASE) or 24 blocks (LARGE) of multi-head self-attention that build contextualized representations over the entire utterance.
3. **Quantization Module** — a Gumbel-Softmax codebook that discretizes the CNN output into a finite set of speech units, used as targets during pre-training.

During pre-training, random spans of the CNN features are masked, and the Transformer must identify the correct quantized latent among distractors via a contrastive loss. After pre-training, a linear CTC head is added and the model is fine-tuned on labeled speech.

### Results

The paper reports strong results on LibriSpeech:

| Model | Unlabeled Data | LM | test-clean | test-other |
|---|---|---|---|---|
| BASE (95M params) | LS-960 | None | 3.4% | 8.5% |
| BASE | LS-960 | Transformer | 2.1% | 4.8% |
| LARGE (317M params) | LV-60k | Transformer | **1.8%** | **3.3%** |

With just 10 minutes of labeled data, the LARGE model still achieves 4.8/8.2 WER — demonstrating that self-supervised pre-training captures the structure of speech remarkably well.

### The Gap We Address

All of the paper's evaluations are conducted on clean or naturally varied speech (LibriSpeech test-clean and test-other). The "other" set contains more challenging recordings but not synthetic additive noise. The paper never quantifies how wav2vec 2.0 behaves when exposed to real-world noise — the kind encountered in phone calls, street conversations, or meeting rooms.

The CNN feature encoder was pre-trained on clean audio. When the input distribution shifts due to additive noise, the learned convolutional filters produce degraded latent representations, which propagate through the Transformer and increase CTC decoding errors. This is the vulnerability we investigate.

---

## 2. Our Approach

### What We Do NOT Do

We do not modify wav2vec 2.0 in any way. We do not re-train it, fine-tune it, or change its architecture. The model remains a frozen, pre-trained checkpoint used purely for inference.

### What We Do

We extend the wav2vec 2.0 pipeline by adding a **speech enhancement preprocessing step** before the ASR model:

```
The Paper:     Clean Speech  →  wav2vec 2.0  →  Text
Real World:    Noisy Speech  →  wav2vec 2.0  →  Degraded Text
Our Pipeline:  Noisy Speech  →  Enhancement  →  wav2vec 2.0  →  Improved Text
```

### Why This Works

wav2vec 2.0's strength comes from its learned representations — but those representations are only as good as the input signal. If the raw waveform is corrupted by noise, the CNN encoder extracts noisy features, the Transformer builds context over noisy features, and the CTC decoder produces noisy transcriptions.

By denoising the signal before it reaches wav2vec 2.0, we bring the input closer to the clean distribution the model was pre-trained on. This is effectively a lightweight **domain adaptation** strategy: rather than adapting the model to noisy data (expensive), we adapt the data to match what the model expects (cheap).

This approach is:

- **Modular** — the enhancement model is independent; it could be placed before any ASR system.
- **Cheap** — our enhancement model has 18,817 parameters and trains in minutes on a single GPU.
- **Practical** — no access to the ASR model's internals is required.

---

## 3. Methodology

### 3.1 Dataset

We use the **LibriSpeech** corpus, the same benchmark used in the wav2vec 2.0 paper.

- **Source**: LibriSpeech train-clean-100, train-clean-360, dev-clean, and test-clean splits ([Kaggle: victorling/librispeech-clean](https://www.kaggle.com/datasets/victorling/librispeech-clean))
- **Total available**: 137,876 utterances
- **Subset used**: 4,000 utterances (3,200 train / 400 val / 400 test), sampled with speaker balancing to ensure diversity
- **Noise corpus**: MUSAN dataset — 600 real-world noise and music clips ([Kaggle: nhattruongdev/musan-noise](https://www.kaggle.com/datasets/nhattruongdev/musan-noise))

The subset was chosen to fit within Kaggle T4 GPU constraints (16 GB VRAM, limited runtime). We used official LibriSpeech splits to ensure zero speaker overlap between train, validation, and test sets.

### 3.2 Noise Generation

We simulate noisy conditions by adding a mixture of Gaussian noise and real background noise (MUSAN) to clean utterances at controlled signal-to-noise ratios (SNR):

| Noise Level | SNR Range | Real-World Equivalent |
|---|---|---|
| Low | 15–20 dB | Quiet office, light background hum |
| Medium | 5–10 dB | Busy cafe, car with windows cracked |
| High | -8 to -3 dB | Loud street, construction nearby |

The noise mixture weights vary by level — higher noise levels use more background noise relative to Gaussian, mimicking realistic environments where environmental noise dominates. Each utterance gets a random SNR drawn uniformly from the level's range.

### 3.3 Speech Enhancement Model

Our enhancement model is a lightweight CNN called **TinySpectrogramDenoiser**. It operates in the STFT magnitude domain:

**Input**: noisy magnitude spectrogram (1-channel image)
**Target**: clean magnitude spectrogram
**Loss**: L1 (mean absolute error)

Architecture:

| Layer | Channels | Kernel | Activation |
|---|---|---|---|
| Conv2d | 1 → 16 | 3×3 | ReLU |
| Conv2d | 16 → 32 | 3×3 | ReLU |
| Conv2d | 32 → 32 | 3×3 | ReLU |
| Conv2d | 32 → 16 | 3×3 | ReLU |
| Conv2d | 16 → 1 | 3×3 | ReLU |

**Total parameters**: 18,817

All layers use `padding=1` to preserve spatial dimensions. The model predicts a clean magnitude spectrogram, which is then combined with the noisy phase to reconstruct the enhanced waveform via inverse STFT. Borrowing the noisy phase is a standard simplification — phase estimation would require a significantly more complex model.

**Training setup:**

- Optimizer: Adam (lr = 1e-3)
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Early stopping: patience = 3 epochs
- Best model checkpoint restored after training
- Trained for 9 epochs, early-stopped; best model from epoch 6 (Val L1: 0.2515)

### 3.4 ASR Model

We use the pre-trained **`facebook/wav2vec2-base-960h`** from Hugging Face — a wav2vec 2.0 BASE model (95M parameters) pre-trained on 960 hours of LibriSpeech and fine-tuned with CTC. The model is loaded frozen and used only for greedy CTC decoding (no language model), matching the paper's "None" LM configuration.

---

## 4. Experiments

### Evaluation Protocol

For each of the 150 test utterances, we generate three transcriptions:

| Condition | Pipeline |
|---|---|
| Clean | Original utterance → wav2vec 2.0 |
| Noisy | Utterance + medium noise → wav2vec 2.0 |
| Enhanced | Utterance + medium noise → Enhancement model → wav2vec 2.0 |

We evaluate using four metrics:

- **WER** (Word Error Rate) — primary ASR quality metric
- **SNR** (Signal-to-Noise Ratio) — signal-level noise measurement in dB
- **PESQ** (Perceptual Evaluation of Speech Quality) — ITU-T P.862 perceptual metric, range ≈ 1.0–4.5
- **STOI** (Short-Time Objective Intelligibility) — intelligibility correlation metric, range 0–1

### Aggregate Results (Medium Noise)

| Metric | Clean | Noisy | Enhanced |
|---|---|---|---|
| **WER** | 0.0367 | 0.3876 | **0.2517** |
| **SNR (dB)** | — | 7.67 | **10.90** |
| **PESQ** | — | 1.123 | **1.391** |
| **STOI** | — | 0.848 | 0.812 |

### Per-Noise-Level Ablation

| Noise Level | WER Clean | WER Noisy | WER Enhanced | WER Improvement |
|---|---|---|---|---|
| Low | 0.0367 | 0.0585 | 0.0704 | -0.0118 |
| Medium | 0.0367 | 0.3673 | 0.2426 | **+0.1247** |
| High | 0.0367 | 0.9760 | 0.9217 | +0.0543 |

### Per-Sample Statistics

| Outcome | Count (of 150) |
|---|---|
| Enhancement helped | 86 (57%) |
| Enhancement hurt | 36 (24%) |
| No change | 28 (19%) |
| Mean WER delta | +0.083 |
| Std WER delta | 0.193 |

---

## 5. Results

### WER Comparison

The core result: enhancement reduces WER from 38.76% to 25.17% under medium noise — a **13.59 percentage point** improvement, corresponding to a **35% relative WER reduction**.

[INSERT FIGURE: WER bar chart — Clean vs Noisy vs Enhanced]

### Signal-Level Quality

SNR improved by 3.23 dB on average, confirming that the enhancer removes noise energy. PESQ improved from 1.123 to 1.391, indicating the enhanced signal is perceptually closer to clean.

[INSERT FIGURE: SNR / PESQ / STOI bar charts side by side]

### SNR Distribution

The SNR histogram shows that the enhancement consistently shifts the distribution toward higher SNR values, with the improvement being relatively uniform across samples.

[INSERT FIGURE: SNR distribution histogram — Noisy vs Enhanced]

### WER by Noise Level

The enhancement is most effective at medium noise levels and provides modest gains at high noise. At low noise, the enhancement slightly hurts — the input was already clean enough.

[INSERT FIGURE: WER by noise level grouped bar chart]

[INSERT FIGURE: WER improvement by noise level bar chart]

### PESQ and STOI by Noise Level

PESQ improves at all noise levels. STOI decreases slightly, which is an expected artifact of magnitude-only enhancement (discussed in Section 6).

[INSERT FIGURE: PESQ by noise level chart]

[INSERT FIGURE: STOI by noise level chart]

### Spectrogram Comparison

Visual inspection of spectrograms confirms that the enhancement model removes noise energy while preserving the primary speech formant structure.

[INSERT FIGURE: Spectrogram triple — Clean / Noisy / Enhanced]

### Residual Analysis

The residual spectrogram (what the enhancer removed vs. what should have been removed) shows that the model primarily targets high-frequency noise components and broadband MUSAN noise.

[INSERT FIGURE: Residual spectrogram — True Noise / Removed / Remaining]

### Waveform Comparison

[INSERT FIGURE: Waveform triple — Clean / Noisy / Enhanced]

### Training Curve

The model converged at epoch 6 with early stopping triggered at epoch 9. The training curve shows healthy learning with no overfitting.

[INSERT FIGURE: Training loss curve with best epoch marker]

### Transcript Examples

| Reference | Noisy Hypothesis | Enhanced Hypothesis |
|---|---|---|
| *the king's ears were now open to montrose's counsel* | *old eweowo emo pose te e epo nowe tepo eamo th* | *polot ingams were now open an montrose's coltn* |
| *there was infinite scepticism around him* | *there was intinit tet e e en lon en o es of ti* | *there was infinite sceptricism round him on th* |
| *bragelonne watched for some time the conduct of* | *ride along watched o sum time e the on a upeto* | *rag alon a watched for some time the conduct u* |

The enhanced transcriptions are not perfect but are dramatically more recognizable than the noisy ones.

### Per-Sample Scatter Plot

[INSERT FIGURE: Scatter plot — per-sample WER (noisy) vs WER (enhanced) with y=x line]

Points below the red diagonal represent samples where enhancement helped. The majority of samples fall below the line, confirming the overall positive effect.

---

## 6. Discussion

### Impact of Noise on wav2vec 2.0

Our results provide clear evidence that wav2vec 2.0, despite its strong self-supervised representations, is highly sensitive to additive noise. The degradation is severe and non-linear:

- At low noise (15–20 dB SNR), WER increases modestly from 3.67% to 5.85% — a 59% relative increase.
- At medium noise (5–10 dB SNR), WER explodes to 38.76% — a 956% relative increase.
- At high noise (-8 to -3 dB SNR), the model essentially fails at 97.60% WER.

This makes sense architecturally: the CNN feature encoder was pre-trained on clean LibriSpeech audio. Its learned filters expect clean spectral patterns. When noise corrupts the input, the extracted latent representations are degraded, and the Transformer builds context over noisy features, compounding the error through the entire network.

### Impact of Enhancement

The lightweight enhancement front-end recovers a significant portion of the noise-induced degradation:

- Under medium noise, WER drops from 38.76% to 25.17% — a **35% relative reduction**.
- SNR improves by +3.23 dB on average.
- PESQ improves from 1.123 to 1.391.
- 86 out of 150 samples (57%) show improved transcriptions.

The 25.17% enhanced WER is not close to the clean-speech performance of 3.67%, but this is expected: an 18,817-parameter CNN operating only on magnitude spectrograms with borrowed noisy phase has inherent limitations. The key finding is that even this minimal intervention meaningfully improves ASR robustness.

### Relation to the Paper

wav2vec 2.0 demonstrates that self-supervised pre-training produces powerful speech representations. Our work reveals a practical constraint the paper does not address: **those representations are only as strong as the input signal quality**.

The paper's best result (1.8% WER) is obtained on clean LibriSpeech with a LARGE model and a Transformer language model. We use the BASE model with no language model and achieve 3.67% on clean speech — consistent with the paper's reported 3.4% for the same configuration. This confirms our experimental setup aligns with the paper.

Our contribution is complementary: the paper focuses on learning better representations; we focus on ensuring those representations receive clean enough input to function well. Together, these address both sides of the ASR robustness problem.

### Noise-Level Behavior

The per-noise-level results reveal an important pattern:

- **Low noise**: Enhancement slightly **hurts** WER (5.85% → 7.04%). When the signal is already clean, the enhancer introduces more distortion than it removes — it over-smooths formants and damages the signal the CNN encoder would have handled fine on its own.
- **Medium noise**: Enhancement **helps the most** (38.76% → 25.17%). This is the sweet spot where noise is severe enough to damage ASR but not so severe that the speech content is irrecoverable.
- **High noise**: Enhancement provides **modest gains** (97.60% → 92.17%). At extreme noise levels, the speech signal is largely destroyed and a lightweight model cannot recover enough content to matter.

This suggests that in a practical deployment, the enhancement step should be gated — applied only when noise is detected above a threshold.

### The STOI Decrease

STOI decreased slightly from 0.848 to 0.812 after enhancement. This does not contradict the other metrics — it is an expected consequence of magnitude-only enhancement. STOI is sensitive to short-time temporal fine structure, which is encoded in the phase. By modifying the magnitude without fixing the phase, we create magnitude-phase inconsistencies that STOI penalizes. Meanwhile, wav2vec 2.0's CNN encoder appears to tolerate spectral smoothing better than additive noise — which explains why WER improves even as STOI slightly decreases.

### Limitations

- **Small evaluation subset**: We evaluate on 150 utterances from LibriSpeech test-clean. Full-corpus evaluation would provide tighter confidence intervals.
- **Lightweight architecture**: A 5-layer CNN with 18K parameters cannot capture complex noise patterns as well as deeper architectures (U-Net, attention-based models).
- **Phase borrowing**: Reusing the noisy phase for reconstruction limits enhancement quality, particularly for non-stationary noise.
- **No fine-tuning comparison**: We do not compare against noise-aware fine-tuning of wav2vec 2.0, which would be the strongest (but most expensive) baseline.
- **Single noise generation method**: We use Gaussian + MUSAN noise. Real-world noise includes reverberation, competing speakers, and far-field effects not covered here.

---

## 7. Conclusion

This project investigates the noise robustness of wav2vec 2.0 and proposes a lightweight, modular solution. Our main findings:

1. **wav2vec 2.0 is highly vulnerable to additive noise.** Medium noise (5–10 dB SNR) increases WER from 3.67% to 38.76% — a tenfold relative degradation. High noise renders the model almost entirely non-functional at 97.60% WER. The paper does not quantify this behavior.

2. **A lightweight speech enhancement front-end significantly improves robustness.** Our 18,817-parameter CNN denoiser reduces medium-noise WER from 38.76% to 25.17% — a 35% relative improvement — validated by WER, SNR, PESQ, and per-sample analysis across three noise levels.

3. **The approach is practical and modular.** No modification to wav2vec 2.0 is needed. The enhancement model trains in under 10 minutes on a single Kaggle T4 GPU and can be placed before any pre-trained ASR system.

This work complements the wav2vec 2.0 paper by addressing input quality rather than model architecture. While wav2vec 2.0 demonstrates that self-supervised learning produces powerful speech representations, our results show that those representations still depend on receiving reasonably clean input — and that a cheap preprocessing step can meaningfully bridge the gap between clean-lab performance and noisy real-world conditions.

---

## References

1. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
2. Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). *Librispeech: An ASR Corpus Based on Public Domain Audio Books.* ICASSP 2015.
3. Snyder, D., Chen, G., & Povey, D. (2015). *MUSAN: A Music, Speech, and Noise Corpus.* arXiv:1510.08484.
4. Rix, A. W., et al. (2001). *Perceptual Evaluation of Speech Quality (PESQ).* ITU-T Recommendation P.862.
5. Taal, C. H., et al. (2011). *An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech.* IEEE TASLP.
6. Graves, A., Fernández, S., & Gomez, F. (2006). *Connectionist Temporal Classification.* ICML 2006.

---

### Datasets Used

- LibriSpeech Clean: [https://www.kaggle.com/datasets/victorling/librispeech-clean](https://www.kaggle.com/datasets/victorling/librispeech-clean)
- MUSAN Noise Corpus: [https://www.kaggle.com/datasets/nhattruongdev/musan-noise](https://www.kaggle.com/datasets/nhattruongdev/musan-noise)
