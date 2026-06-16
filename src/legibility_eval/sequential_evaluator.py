"""
Sequential Reasoning Evaluator for RAW frames approach.

Tests 3 prompting strategies with unmodified frames:
1. Full chain-of-thought (single pass)
2. Differential analysis (multi-turn)
3. Simple spatial (baseline)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image

from .raw_frames import extract_raw_frames, create_frame_grid, create_horizontal_strip, get_video_info
from .sequential_reasoning_prompts import (
    build_sequential_reasoning_prompt,
    build_differential_analysis_prompt,
    build_aggregation_prompt,
    build_simple_spatial_prompt
)


class SequentialReasoningEvaluator:
    """
    Evaluator for raw frames + sequential reasoning approach.
    """
    
    def __init__(self, vlm, strategy: str = "chain_of_thought"):
        """
        Initialize evaluator.
        
        Args:
            vlm: Prismatic VLM instance
            strategy: One of ["chain_of_thought", "differential", "simple"]
        """
        self.vlm = vlm
        self.strategy = strategy
        
        if strategy not in ["chain_of_thought", "differential", "simple"]:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def evaluate_video(
        self,
        video_path: str,
        checkpoint_time: float,
        ground_truth_goal: str,
        goal_descriptions: Dict[str, str],
        n_frames: int = 6,
        layout: str = "grid"
    ) -> Dict[str, Any]:
        """
        Evaluate single video at checkpoint time.
        
        Args:
            video_path: Path to video
            checkpoint_time: Time to evaluate
            ground_truth_goal: Ground truth goal ID ("A" or "B")
            goal_descriptions: Dict mapping goal IDs to descriptions
            n_frames: Number of frames
            layout: "grid" or "horizontal"
        
        Returns:
            Dict with prediction, confidence, correct, reasoning
        """
        # Get video info
        info = get_video_info(video_path)
        fps = info['fps']
        
        # Extract raw frames
        frames = extract_raw_frames(
            video_path=video_path,
            checkpoint_time=checkpoint_time,
            fps=fps,
            n_frames=n_frames,
            target_size=(384, 384)
        )
        
        if len(frames) < n_frames:
            print(f"Warning: Only extracted {len(frames)}/{n_frames} frames")
        
        # Create montage
        if layout == "grid":
            montage = create_frame_grid(frames, grid_cols=3, add_frame_numbers=True)
        else:
            montage = create_horizontal_strip(frames, spacing=10, add_frame_numbers=True)
        
        # Evaluate based on strategy
        if self.strategy == "chain_of_thought":
            result = self._evaluate_chain_of_thought(montage, goal_descriptions, n_frames)
        elif self.strategy == "differential":
            result = self._evaluate_differential(frames, goal_descriptions)
        else:  # simple
            result = self._evaluate_simple(montage, goal_descriptions)
        
        # Add metadata
        result['video_path'] = video_path
        result['checkpoint_time'] = checkpoint_time
        result['ground_truth'] = ground_truth_goal
        result['correct'] = (result.get('predicted_goal') == ground_truth_goal)
        result['strategy'] = self.strategy
        result['n_frames'] = len(frames)
        
        return result
    
    def _evaluate_chain_of_thought(
        self,
        montage: Image.Image,
        goal_descriptions: Dict[str, str],
        n_frames: int
    ) -> Dict[str, Any]:
        """
        Strategy 1: Full chain-of-thought in single pass.
        """
        prompt = build_sequential_reasoning_prompt(
            n_frames=n_frames,
            scene_description="robot manipulation",
            goal_descriptions=goal_descriptions
        )
        
        # Query VLM
        response = self.vlm.generate(
            image=montage,
            prompt_text=prompt,
            temperature=0.2,
            max_new_tokens=1024
        )
        
        # Parse JSON response
        parsed = self._parse_json_response(response)
        
        return {
            'predicted_goal': parsed.get('predicted_goal'),
            'confidence': parsed.get('confidence', 0),
            'reasoning': parsed.get('reasoning', ''),
            'raw_response': response,
            'trajectory_summary': parsed.get('trajectory_summary', '')
        }
    
    def _evaluate_differential(
        self,
        frames: List[Image.Image],
        goal_descriptions: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Strategy 2: Analyze frame pairs separately, then aggregate.
        """
        n_pairs = len(frames) - 1
        motion_summaries = []
        
        # Analyze each consecutive pair
        for i in range(n_pairs):
            # Create pair image (side by side)
            pair_img = self._create_pair_image(frames[i], frames[i+1])
            
            prompt = build_differential_analysis_prompt(i, n_pairs)
            
            response = self.vlm.generate(
                image=pair_img,
                prompt_text=prompt,
                temperature=0.2,
                max_new_tokens=512
            )
            
            parsed = self._parse_json_response(response)
            motion_summaries.append(parsed)
        
        # Aggregate results
        agg_prompt = build_aggregation_prompt(motion_summaries, goal_descriptions)
        
        # For aggregation, we don't need an image (text-only reasoning)
        # But Prismatic requires an image, so use the last pair
        final_pair = self._create_pair_image(frames[-2], frames[-1])
        
        agg_response = self.vlm.generate(
            image=final_pair,
            prompt_text=agg_prompt,
            temperature=0.2,
            max_new_tokens=512
        )
        
        final_parsed = self._parse_json_response(agg_response)
        
        return {
            'predicted_goal': final_parsed.get('predicted_goal'),
            'confidence': final_parsed.get('confidence', 0),
            'reasoning': final_parsed.get('reasoning', ''),
            'motion_summaries': motion_summaries,
            'trajectory_pattern': final_parsed.get('trajectory_pattern', '')
        }
    
    def _evaluate_simple(
        self,
        montage: Image.Image,
        goal_descriptions: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Strategy 3: Simple spatial reasoning (no explicit steps).
        """
        prompt = build_simple_spatial_prompt(goal_descriptions)
        
        response = self.vlm.generate(
            image=montage,
            prompt_text=prompt,
            temperature=0.2,
            max_new_tokens=512
        )
        
        parsed = self._parse_json_response(response)
        
        return {
            'predicted_goal': parsed.get('predicted_goal'),
            'confidence': parsed.get('confidence', 0),
            'reasoning': parsed.get('reasoning', ''),
            'raw_response': response
        }
    
    def _create_pair_image(self, frame1: Image.Image, frame2: Image.Image) -> Image.Image:
        """
        Create side-by-side image of two frames.
        """
        import numpy as np
        
        w, h = frame1.size
        pair = np.ones((h, w * 2 + 20, 3), dtype=np.uint8) * 255
        
        pair[:, :w] = np.array(frame1)
        pair[:, w+20:] = np.array(frame2)
        
        return Image.fromarray(pair)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from VLM response.
        """
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print(f"Warning: Could not find JSON in response")
                return {}
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON parse error: {e}")
            print(f"Attempted to parse: {json_str[:200]}...")
            return {}


def run_single_video_test(
    video_path: str,
    checkpoints: List[float],
    ground_truth_goal: str,
    goal_descriptions: Dict[str, str],
    strategy: str = "chain_of_thought",
    output_dir: Path = Path("results")
) -> Tuple[List[Dict], float]:
    """
    Test single video at multiple checkpoints.
    
    Args:
        video_path: Path to video
        checkpoints: List of times to evaluate
        ground_truth_goal: Ground truth goal ID
        goal_descriptions: Goal descriptions
        strategy: Prompting strategy
        output_dir: Directory to save results
    
    Returns:
        (results_list, accuracy)
    """
    from prismatic import load
    
    print(f"Loading Prismatic VLM...")
    vlm = load("prism-dinosiglip+7b")
    
    evaluator = SequentialReasoningEvaluator(vlm, strategy=strategy)
    
    results = []
    
    for checkpoint in checkpoints:
        print(f"\nEvaluating at t={checkpoint}s...")
        
        result = evaluator.evaluate_video(
            video_path=video_path,
            checkpoint_time=checkpoint,
            ground_truth_goal=ground_truth_goal,
            goal_descriptions=goal_descriptions,
            n_frames=6,
            layout="grid"
        )
        
        results.append(result)
        
        print(f"  Predicted: {result.get('predicted_goal', 'N/A')}")
        print(f"  Ground truth: {ground_truth_goal}")
        print(f"  Correct: {'✓' if result['correct'] else '✗'}")
        print(f"  Confidence: {result.get('confidence', 0)}")
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) if results else 0
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"raw_frames_{strategy}_test.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "strategy": strategy,
            "video": video_path,
            "ground_truth": ground_truth_goal,
            "checkpoints": checkpoints,
            "results": results,
            "summary": {
                "accuracy": accuracy,
                "correct": correct_count,
                "total": len(results)
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nAccuracy: {accuracy:.1%} ({correct_count}/{len(results)})")
    
    return results, accuracy
