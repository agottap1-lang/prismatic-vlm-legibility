# VLM Evaluation Pipeline for Robot Motion Legibility (built on Prismatic-VLMs)

A pipeline that uses a pretrained **Prismatic VLM** (`prism-dinosiglip+7b`: DINOv2 + SigLIP vision backbone, LLaMA-2 7B) to evaluate whether a robot's motion reveals its intended goal — i.e., robot-motion **legibility** — from rendered video frames. This work feeds my thesis on VLM-based evaluation of robot motion.

> **Attribution / what is not mine.** This repository is a clone of **TRI-ML/prismatic-vlms** (https://github.com/TRI-ML/prismatic-vlms, Apache-2.0) — the VLM training/inference stack is theirs and the original project README is preserved as [`README_UPSTREAM.md`](README_UPSTREAM.md). **My contribution** is the legibility-evaluation pipeline layered on top of it (everything under `src/legibility_eval/`, `evaluate.py`, `tests/`, the prompt/montage strategies, and the analysis below). I am sharing this to document the evaluation work I built, with full credit to the upstream authors.

---

## Problem
Can an off-the-shelf VLM infer which goal a manipulator is reaching for, from a short sequence of frames? If so, a VLM could act as a cheap, language-grounded evaluator of motion legibility.

## What I built (my contribution)
- **`src/legibility_eval/`** — modular evaluation package: frame sampling (`frames.py`, `raw_frames.py`), montage rendering (horizontal/grid/vertical), trajectory extraction & overlay (`trajectory.py`, `trajectory_encoding.py`), goal/block detection (`block_detection.py`), an inference client (`client.py`), checkpoint- and sequence-based evaluators, and several prompt strategies (task-agnostic, chain-of-thought, sequential reasoning).
- **`evaluate.py`** — single entry point; each run logs JSON with **run id, git commit, dirty flag, model config, sample rate, and frame count** for reproducibility.
- **`tests/`** — regression tests for checkpoint evaluation and color-marker prompting.
- A controlled study across visual-encoding/prompting strategies to isolate *why* the VLM succeeds or fails (plain montages; color-grounded goal regions; explicit trajectory annotation; dense traces).

## Tech stack
Prismatic `prism-dinosiglip+7b` (DINOv2 + SigLIP, LLaMA-2 7B) · PyTorch · bf16 on CUDA (A100) · LIBERO-style robot clips.

## Results (honest, mostly negative — and useful)
| Strategy | Goal-inference accuracy | Finding |
|---|---|---|
| Frame montages (any layout) | ~50% (chance) | Layout doesn't matter |
| Color-grounded goal boxes | 50% | VLM *names* the right color but still picks the wrong goal → not a grounding issue |
| Explicit trajectory + arrows | 37.5% | **Worse** — the VLM misreads arrow direction |

**Key finding:** this VLM shows a systematic left-bias and cannot reliably infer motion direction from static frame montages; adding explicit cues *degraded* performance. This negative result motivated my flagship project ([`multimodal-diffusion-steering`](https://github.com/agottap1-lang/multimodal-diffusion-steering)), where a VLM is instead used for **best-of-N reranking over candidate trajectories with curated frames** — and there it *does* improve legibility.

## Status
This is research code; the analysis above is what I concluded from my runs. Some scripts are exploratory / in development.

## Limitations / next steps
Single VLM family; small clip set; static-frame representation (no video-native model, no fine-tuning). Next: temporally-aware/video VLMs, light fine-tuning on motion-intent data, temporal token encodings.

## License
Upstream code under Apache-2.0 (see [`LICENSE`](LICENSE) and [`README_UPSTREAM.md`](README_UPSTREAM.md)). My added pipeline code is shared under the same terms.
