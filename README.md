# Speech Enhancement as a Preprocessing Step for wav2vec 2.0 ASR

A course project investigating the noise robustness of wav2vec 2.0 and proposing a lightweight speech enhancement front-end to improve ASR performance under noisy conditions.

## Overview

**wav2vec 2.0** (Baevski et al., 2020) achieves state-of-the-art ASR on clean speech through self-supervised pre-training, but it was never evaluated under additive noise. This project:

1. **Quantifies the vulnerability**: wav2vec 2.0 BASE degrades from 3.67% WER on clean speech to 38.76% under medium noise (SNR 5–10 dB).
2. **Proposes a fix**: A lightweight CNN denoiser (18,817 parameters) preprocesses noisy audio before it reaches the frozen wav2vec 2.0 model.
3. **Validates the improvement**: Enhanced WER drops to 25.17% — a **35% relative reduction** — confirmed by WER, SNR, PESQ, and STOI across three noise levels.

### Pipeline

```
Noisy Speech → Speech Enhancement (CNN Denoiser) → wav2vec 2.0 ASR → Text
```

No modification to wav2vec 2.0 is needed. The ASR model stays frozen.

## Key Results

| Condition | WER | SNR | PESQ |
|---|---|---|---|
| Clean | 3.67% | — | — |
| Noisy (medium) | 38.76% | 7.67 dB | 1.123 |
| Enhanced | **25.17%** | **10.90 dB** | **1.391** |

## Repository Contents

| File | Description |
|---|---|
| `speach-project.ipynb` | Full Kaggle notebook — data loading, noise generation, model training, evaluation, and all plots |
| `report.md` | Project report in Markdown with all results and analysis |
| `Report.pdf` | Formatted PDF version of the report |
| `study-guide.md` | Comprehensive study guide covering the paper, the project, all results, and TA discussion prep |
| `2006.11477v3.pdf` | The wav2vec 2.0 paper (Baevski et al., NeurIPS 2020) |

## Datasets

- [LibriSpeech Clean](https://www.kaggle.com/datasets/victorling/librispeech-clean) — 4,000 utterance subset (train/val/test)
- [MUSAN Noise Corpus](https://www.kaggle.com/datasets/nhattruongdev/musan-noise) — 600 real-world noise and music clips

## Technical Details

- **Enhancement model**: 5-layer CNN on STFT magnitude spectrograms, trained with L1 loss, early stopping, and ReduceLROnPlateau
- **ASR model**: `facebook/wav2vec2-base-960h` (frozen, greedy CTC decoding, no language model)
- **Noise levels**: Low (15–20 dB), Medium (5–10 dB), High (-8 to -3 dB) using Gaussian + MUSAN mixtures
- **Compute**: Single Kaggle T4 GPU (16 GB VRAM)

## Reference

Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020. [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
