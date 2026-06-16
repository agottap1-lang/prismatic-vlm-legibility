"""
generate.py

Simple CLI script to interactively test generating from a pretrained VLM; provides a minimal REPL for specifying image
URLs, prompts, and language generation parameters.

Run with: python scripts/generate.py --model_path <PATH TO LOCAL MODEL OR HF HUB>
"""

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional
from io import BytesIO
from datetime import datetime

import draccus
import requests
import torch
from PIL import Image

# Optional dependency for video support
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from prismatic import load
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Default Image URL (Beignets)
DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
)

# --- unified image loader (URL or local path) ---------------------------------------------
def load_image(src: str) -> Image.Image:
    """Load an image from an HTTP(S) URL or a local filesystem path."""
    if not src:
        raise ValueError("Empty image source.")
    src = os.path.expanduser(str(src))
    if src.startswith(("http://", "https://")):
        r = requests.get(src, timeout=30)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    if not os.path.exists(src):
        raise FileNotFoundError(f"Image not found: {src}")
    return Image.open(src).convert("RGB")
# ------------------------------------------------------------------------------------------


# --- simple frame extractor for .mp4 ------------------------------------------------------
def extract_frames(video_path: str, frame_skip: int = 10):
    """
    Yield PIL.Image frames from a video, taking every `frame_skip`-th frame.
    This keeps memory low and works with the existing vlm.generate(image, prompt, ...).
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) not installed. Install with `pip install opencv-python`.")
    video_path = os.path.expanduser(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield Image.fromarray(frame_rgb)
        idx += 1
    cap.release()
# ------------------------------------------------------------------------------------------


def read_multiline_prompt(end_token: str = "END") -> str:
    """
    Read a multi-line prompt from stdin.
    Paste your prompt, then type END on a new line to finish.
    """
    print(f"\n|=>> Paste analysis prompt (finish with a single line containing {end_token}):")
    lines = []
    while True:
        line = input()
        if line.strip() == end_token:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def build_legibility_prompt(goal_A: str, goal_B: str, t_sec: int, video_id: str, mode: str = "single_frame") -> str:
    """
    Build structured prompt for legibility evaluation matching Gemini study.
    
    Args:
        goal_A: Description of goal A (e.g., "pick the left block")
        goal_B: Description of goal B (e.g., "pick the right block")
        t_sec: Current timestamp in seconds
        video_id: Video identifier for context
        mode: "single_frame" or "prefix_frames"
    
    Returns:
        Formatted prompt requesting pA/pB JSON output
    """
    mode_instruction = ""
    if mode == "single_frame":
        mode_instruction = (
            "You are given ONLY ONE image showing the robot at a specific moment.\n"
            "Use ONLY this frame to make your judgment. Do NOT assume you saw earlier or later frames."
        )
    else:  # prefix_frames
        mode_instruction = (
            "You are given MULTIPLE images showing frames from the start (t=0) up to the current time.\n"
            "Frames are ordered from earliest to latest. You have observed the motion up to the current time."
        )
    
    prompt = f"""You are evaluating robot motion legibility for video: {video_id}

Two possible goals:
- Goal A: {goal_A}
- Goal B: {goal_B}

Current time: {t_sec} seconds

{mode_instruction}

Analyze the image(s) and determine which goal the robot is most likely pursuing.

IMPORTANT: Output ONLY valid JSON with this exact format:
{{
  "pA": <probability 0.0-1.0 that robot is pursuing Goal A>,
  "pB": <probability 0.0-1.0 that robot is pursuing Goal B>,
  "cue": "<describe the visual evidence you observed>",
  "legible": "<legible_now if you can confidently determine the goal, otherwise not_legible_yet>"
}}

Rules:
- pA and pB must be numbers between 0.0 and 1.0
- pA + pB should equal 1.0
- Base your judgment ONLY on what you can see in the image(s)
- If uncertain, set pA and pB close to 0.5
"""
    return prompt


def extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object substring from model text."""
    if not text:
        return None
    s = text.strip()

    # Fast path: already a single JSON object
    if s.startswith("{") and s.endswith("}"):
        return s

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1].strip()
    return None


def parse_legibility_response(text: str) -> dict:
    """
    Parse Prismatic response for legibility evaluation.
    Expected JSON format: {"pA": float, "pB": float, "cue": str, "legible": str}
    """
    json_str = extract_first_json_object(text)
    if json_str:
        try:
            data = json.loads(json_str)
            # Validate and normalize probabilities
            pA = float(data.get("pA", 0.5))
            pB = float(data.get("pB", 0.5))
            
            # Clamp to [0, 1]
            pA = max(0.0, min(1.0, pA))
            pB = max(0.0, min(1.0, pB))
            
            # Normalize if sum != 1.0 (with tolerance)
            total = pA + pB
            if abs(total - 1.0) > 0.01:
                if total > 0:
                    pA = pA / total
                    pB = pB / total
                else:
                    pA = pB = 0.5
            
            return {
                "pA": pA,
                "pB": pB,
                "cue": str(data.get("cue", "")),
                "legible": str(data.get("legible", "not_legible_yet"))
            }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            overwatch.warning(f"Failed to parse JSON response: {e}")
    
    # Fallback: return uncertain response
    return {
        "pA": 0.5,
        "pB": 0.5,
        "cue": "Failed to parse response",
        "legible": "not_legible_yet"
    }


def apply_decision_rule(pA: float, pB: float) -> tuple[str, int]:
    """
    Apply Gemini study decision rule: choice and confidence from pA, pB.
    
    Logic:
    - max_p = max(pA, pB)
    - confidence = int(round(max_p * 100))
    - if max_p >= 0.60: choice = 'A' if pA >= pB else 'B'
    - else: choice = 'C' (uncertain)
    
    Returns:
        (choice, confidence) where choice in {'A', 'B', 'C'}
    """
    max_p = max(pA, pB)
    confidence = int(round(max_p * 100))
    
    if max_p >= 0.60:
        choice = 'A' if pA >= pB else 'B'
    else:
        choice = 'C'  # Uncertain
    
    return choice, confidence


def normalize_legibility_json(obj: dict, frame_num: int) -> dict:
    """
    Force exact schema and apply your rule:
      if confidence < 60 => choice="Cannot determine yet" and legible="not_legible_yet"
    """
    # candidates
    candidates = obj.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = [str(candidates)]
    candidates = [str(x) for x in candidates][:4]

    # choice
    choice = str(obj.get("choice", "Cannot determine yet"))

    # confidence
    try:
        confidence = int(obj.get("confidence", 0))
    except Exception:
        confidence = 0
    confidence = max(0, min(100, confidence))

    # cue
    cue = str(obj.get("cue", ""))

    # legible
    legible = str(obj.get("legible", "not_legible_yet"))
    if legible not in ("legible_now", "not_legible_yet"):
        legible = "not_legible_yet"

    # enforce rule
    if confidence < 60:
        choice = "Cannot determine yet"
        legible = "not_legible_yet"

    return {
        "frame": int(frame_num),
        "candidates": candidates,
        "choice": choice,
        "confidence": confidence,
        "cue": cue,
        "legible": legible,
    }


def resolve_hf_token(cfg_hf_token: Union[str, Path]) -> str:
    """
    Resolve HF token from either:
      - a path (relative paths resolved against repo root), or
      - an environment variable name
    """
    repo_root = Path(__file__).resolve().parent.parent  # .../prismatic-vlms
    if isinstance(cfg_hf_token, Path):
        token_path = cfg_hf_token
        if not token_path.is_absolute():
            token_path = (repo_root / token_path).resolve()
        if not token_path.exists():
            raise FileNotFoundError(
                f"HF token file not found: {token_path}\n"
                f"Create it with: echo 'hf_xxx' > {token_path}\n"
                f"Or pass an env var name: --hf_token HF_TOKEN (after export HF_TOKEN=...)"
            )
        return token_path.read_text().strip()

    # env var
    env_name = str(cfg_hf_token)
    if env_name not in os.environ:
        raise KeyError(
            f"Environment variable '{env_name}' not set for HF token.\n"
            f"Set it via: export {env_name}=hf_xxx"
        )
    return os.environ[env_name].strip()


@dataclass
class GenerateConfig:
    # fmt: off
    model_path: Union[str, Path] = "prism-dinosiglip+7b"
    hf_token: Union[str, Path] = Path(".hf_token")   # file path OR env var name

    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1

    # video
    frame_skip: int = 15
    
    # legibility evaluation mode
    legibility_mode: bool = False  # If True, use structured pA/pB prompting
    # fmt: on


@draccus.wrap()
def generate(cfg: GenerateConfig) -> None:
    overwatch.info(f"Initializing Generation Playground with Prismatic Model `{cfg.model_path}`")
    hf_token = resolve_hf_token(cfg.hf_token)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vlm = load(cfg.model_path, hf_token=hf_token)

    # Use bf16 only on CUDA
    if device.type == "cuda":
        vlm.to(device, dtype=torch.bfloat16)
    else:
        vlm.to(device)

    # Initial Setup
    image = load_image(DEFAULT_IMAGE_URL)
    prompt_builder = vlm.get_prompt_builder()
    system_prompt = prompt_builder.system_prompt

    print(
        "[*] Dropping into Prismatic VLM REPL with Default Generation Setup => Initial Conditions:\n"
        f"       => Prompt Template:\n\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n"
        f"       => Default Image URL: `{DEFAULT_IMAGE_URL}`\n"
        f"       => Legibility Mode: {'ENABLED' if cfg.legibility_mode else 'DISABLED'}\n===\n"
    )

    repl_prompt = (
        "|=>> Enter (i)mage (PATH or URL), (v)ideo (.mp4 PATH), (l)egibility test, (p)rompt to update prompt template, "
        "(q)uit to exit, or any other key to enter input questions: "
    )

    repo_root = Path(__file__).resolve().parent.parent
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    while True:
        user_input = input(repl_prompt)

        if user_input.lower().startswith("q"):
            print("\n|=>> Received (q)uit signal => Exiting...")
            return

        elif user_input.lower().startswith("i"):
            src = input("\n|=>> Enter Image PATH or URL: ").strip()
            try:
                image = load_image(src)
                prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
                print("[*] Image loaded successfully.\n")
            except Exception as e:
                print(f"|=>> Failed to load image: {e}\n")
                continue

        elif user_input.lower().startswith("l"):
            # Legibility test mode
            print("\n[*] Entering Legibility Evaluation Mode")
            print("=" * 60)
            
            # Get image
            src = input("|=>> Enter Image PATH or URL: ").strip()
            try:
                test_image = load_image(src)
            except Exception as e:
                print(f"|=>> Failed to load image: {e}\n")
                continue
            
            # Get goals
            goal_A = input("|=>> Enter Goal A description: ").strip()
            goal_B = input("|=>> Enter Goal B description: ").strip()
            t_sec = input("|=>> Enter timestamp (seconds): ").strip()
            video_id = input("|=>> Enter video_id (optional): ").strip() or "test_video"
            
            try:
                t_sec = int(t_sec)
            except ValueError:
                t_sec = 0
            
            # Build legibility prompt
            legibility_prompt = build_legibility_prompt(
                goal_A=goal_A,
                goal_B=goal_B,
                t_sec=t_sec,
                video_id=video_id,
                mode="single_frame"
            )
            
            print(f"\n[*] Generated prompt ({len(legibility_prompt)} chars)")
            print(f"[*] Calling model...")
            
            # Generate response
            generated_text = vlm.generate(
                test_image,
                legibility_prompt,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                max_new_tokens=cfg.max_new_tokens,
                min_length=cfg.min_length,
            )
            
            print(f"\n[*] Raw model response:")
            print("-" * 60)
            print(generated_text)
            print("-" * 60)
            
            # Parse response
            parsed = parse_legibility_response(generated_text)
            choice, confidence = apply_decision_rule(parsed["pA"], parsed["pB"])
            
            print(f"\n[*] Parsed results:")
            print(f"    pA: {parsed['pA']:.3f}")
            print(f"    pB: {parsed['pB']:.3f}")
            print(f"    Choice: {choice}")
            print(f"    Confidence: {confidence}%")
            print(f"    Cue: {parsed['cue']}")
            print(f"    Legible: {parsed['legible']}")
            print("=" * 60 + "\n")

        elif user_input.lower().startswith("v"):
            video_path = input("\n|=>> Enter Video .mp4 PATH: ").strip()

            # Multi-line prompt input to avoid truncation
            analysis_instruction = read_multiline_prompt(end_token="END")
            if not analysis_instruction:
                analysis_instruction = (
                    "Describe what's happening in this frame. Focus on the robot, objects, and the pick action."
                )

            # Create per-video output file (JSONL)
            video_stem = Path(video_path).stem
            safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", video_stem)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = outputs_dir / f"{safe_stem}__{timestamp}.jsonl"

            out_fp = None
            try:
                out_fp = open(out_path, "w", encoding="utf-8")
                print(f"[*] Saving per-frame JSONL to: {out_path}")
                print("[*] Processing video frames (CTRL-C to stop)…\n")

                count = 0
                for frame in extract_frames(video_path, frame_skip=cfg.frame_skip):
                    count += 1

                    frame_prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
                    frame_prompt_builder.add_turn(
                        role="human",
                        message=(
                            f"{analysis_instruction}\n\n"
                            f"FRAME_NUMBER={count}\n"
                            f"IMPORTANT: Output ONLY valid JSON matching the schema.\n"
                            f"IMPORTANT: Use FRAME_NUMBER value exactly for the JSON field 'frame'."
                        ),
                    )
                    prompt_text = frame_prompt_builder.get_prompt()

                    generated_text = vlm.generate(
                        frame,
                        prompt_text,
                        do_sample=cfg.do_sample,
                        temperature=cfg.temperature,
                        max_new_tokens=cfg.max_new_tokens,
                        min_length=cfg.min_length,
                    )

                    # Parse JSON if possible; otherwise fallback
                    jtxt = extract_first_json_object(generated_text)
                    parsed = None
                    if jtxt is not None:
                        try:
                            parsed = json.loads(jtxt)
                        except Exception:
                            parsed = None

                    if not isinstance(parsed, dict):
                        parsed = {
                            "candidates": [],
                            "choice": "Cannot determine yet",
                            "confidence": 0,
                            "cue": "Model did not return valid JSON",
                            "legible": "not_legible_yet",
                        }

                    normalized = normalize_legibility_json(parsed, frame_num=count)

                    # Print pretty JSON like your example
                    print(f"=== Frame {count} ===")
                    print(json.dumps(normalized, indent=2))
                    print()

                    # Save one JSON per line (JSONL)
                    out_fp.write(json.dumps(normalized) + "\n")
                    out_fp.flush()

                if count == 0:
                    print("|=>> No frames extracted (check video or codecs).")

            except KeyboardInterrupt:
                print("\n|=>> Stopped video processing.\n")
            except Exception as e:
                print(f"|=>> Failed to process video: {e}\n")
            finally:
                if out_fp is not None:
                    out_fp.close()

        elif user_input.lower().startswith("p"):
            if system_prompt is None:
                print("\n|=>> Model does not support `system_prompt`!")
                continue

            system_prompt = input("\n|=>> Enter New System Prompt: ")
            prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
            print(
                "\n[*] Set New System Prompt:\n"
                f"    => Prompt Template:\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n"
            )

        else:
            print("\n[*] Entering Chat Session - CTRL-C to start afresh!\n===\n")
            try:
                while True:
                    message = input("|=>> Enter Prompt: ").strip()
                    if message.lower() in {"q", "quit", "exit"}:
                        print("\n|=>> Exiting chat session.\n")
                        break

                    prompt_builder.add_turn(role="human", message=message)
                    prompt_text = prompt_builder.get_prompt()

                    generated_text = vlm.generate(
                        image,
                        prompt_text,
                        do_sample=cfg.do_sample,
                        temperature=cfg.temperature,
                        max_new_tokens=cfg.max_new_tokens,
                        min_length=cfg.min_length,
                    )
                    prompt_builder.add_turn(role="gpt", message=generated_text)
                    print(f"\t|=>> VLM Response >>> {generated_text}\n")

            except KeyboardInterrupt:
                print("\n===\n")
                continue


if __name__ == "__main__":
    generate()
