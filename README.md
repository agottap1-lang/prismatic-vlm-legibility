# VLM Evaluation Pipeline for Robot Motion Legibility (built on Prismatic-VLMs)

A pipeline that uses a pretrained Prismatic VLM (`prism-dinosiglip+7b`: a DINOv2 + SigLIP vision backbone with a LLaMA-2 7B language model) to evaluate whether a robot's motion reveals its intended goal, that is, robot-motion legibility, from rendered video frames. This work feeds my thesis on VLM-based evaluation of robot motion.

**Attribution.** This repository is a clone of TRI-ML/prismatic-vlms (https://github.com/TRI-ML/prismatic-vlms, Apache-2.0). The VLM training and inference stack is theirs, and the original project README is preserved as [`README_UPSTREAM.md`](README_UPSTREAM.md). My contribution is the legibility-evaluation pipeline layered on top of it: everything under `src/legibility_eval/`, `evaluate.py`, `tests/`, the prompt and montage strategies, and the analysis below. I am sharing this to document the evaluation work I built, with credit to the upstream authors.

## Problem

Can an off-the-shelf VLM infer which goal a manipulator is reaching for from a short sequence of frames? If so, a VLM could act as a cheap, language-grounded evaluator of motion legibility.

## What I built

- `src/legibility_eval/`, a modular evaluation package: frame sampling (`frames.py`, `raw_frames.py`), montage rendering (horizontal, grid, vertical), trajectory extraction and overlay (`trajectory.py`, `trajectory_encoding.py`), goal/block detection (`block_detection.py`), an inference client (`client.py`), checkpoint- and sequence-based evaluators, and several prompt strategies (task-agnostic, chain-of-thought, sequential reasoning).
- `evaluate.py`, a single entry point. Each run logs JSON with the run id, git commit, dirty flag, model config, sample rate, and frame count for reproducibility.
- `tests/`, regression tests for checkpoint evaluation and color-marker prompting.
- A controlled study across visual-encoding and prompting strategies to isolate why the VLM succeeds or fails: plain montages, color-grounded goal regions, explicit trajectory annotation, and dense traces.

## Tech stack

Prismatic `prism-dinosiglip+7b` (DINOv2 + SigLIP, LLaMA-2 7B), PyTorch, bf16 on CUDA (A100), and LIBERO-style robot clips.

## Results

The main result is a negative one, and it is useful.

| Strategy | Goal-inference accuracy | Finding |
|---|---|---|
| Frame montages (any layout) | about 50% (chance) | layout does not matter |
| Color-grounded goal boxes | 50% | the VLM names the right color but still picks the wrong goal, so it is not a grounding issue |
| Explicit trajectory + arrows | 37.5% | worse than baseline; the VLM misreads arrow direction |

The main finding is that this VLM shows a systematic left-bias and cannot reliably infer motion direction from static frame montages; adding explicit cues degraded performance. That negative result motivated my flagship project ([`multimodal-diffusion-steering`](https://github.com/agottap1-lang/multimodal-diffusion-steering)), where a VLM is instead used for best-of-N reranking over candidate trajectories with curated frames, and there it does improve legibility.

## Status

This is research code; the analysis above is what I concluded from my runs. Some scripts are exploratory and in development.

## Limitations and next steps

This uses a single VLM family, a small clip set, and a static-frame representation (no video-native model and no fine-tuning). Next steps: temporally aware or video VLMs, light fine-tuning on motion-intent data, and temporal token encodings.

## License

Upstream code is under Apache-2.0 (see [`LICENSE`](LICENSE) and [`README_UPSTREAM.md`](README_UPSTREAM.md)). My added pipeline code is shared under the same terms.
