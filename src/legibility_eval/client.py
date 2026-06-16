"""
Probability-based Prismatic VLM client.

VLM provides pA and pB directly, our code decides choice and calculates confidence.
"""

from typing import Optional, Dict, List
from pathlib import Path
from PIL import Image
import torch
from prismatic import load as load_prismatic_model

from .frames import create_frame_montage, extract_frame_sequence
from .prompt import (
    build_probability_prompt,
    parse_probability_response,
    calculate_decision_metrics,
)


class PrismaticTemporalClient:
    """
    Probability-based Prismatic VLM client.
    
    Shows 6-frame montage, VLM provides pA/pB, code decides choice and confidence.
    """
    
    def __init__(
        self,
        model_path: str = "prism-dinosiglip+7b",
        hf_token: Optional[str] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize client."""
        self.model_path = model_path
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        
        # Handle HF token
        if hf_token is None:
            hf_token_path = Path(".hf_token")
            if hf_token_path.exists():
                hf_token = hf_token_path.read_text().strip()
        elif isinstance(hf_token, (str, Path)):
            p = Path(hf_token)
            if p.exists():
                hf_token = p.read_text().strip()
        
        # Load model
        self.vlm = load_prismatic_model(model_path, hf_token=hf_token)
        self.vlm.to(self.device, dtype=self.dtype)
    
    def evaluate_temporal(
        self,
        frames: List[Image.Image],
        timestamps: List[float],
        goal_A_desc: str,
        goal_B_desc: str,
        video_id: str,
        mode: str = "prefix_frames",
        layout: str = "horizontal",
        grid_cols: int = 3,
        frame_size: tuple = (320, 320),
        use_color_markers: bool = False,
        use_trajectory: bool = False,
    ) -> dict:
        """
        Evaluate using 6-frame montage - VLM provides pA/pB directly.
        
        Args:
            frames: List of 6 PIL Images from t=0 to t=5
            timestamps: List of timestamps
            goal_A_desc: Description of Goal A
            goal_B_desc: Description of Goal B
            video_id: Video identifier
            mode: Evaluation mode - "single_frame" or "prefix_frames"
            layout: Layout type - "horizontal", "grid", or "vertical"
            grid_cols: Grid columns for "grid" layout
            frame_size: Size to resize each frame to (width, height)
            use_color_markers: If True, use color-based prompts and expect frames with colored markers
            use_trajectory: If True, annotate trajectory on montage
        
        Returns:
            dict with pA (from VLM), pB (from VLM), choice (from code), confidence (from code), cue (from VLM), legible (from VLM)
        """
        # Create frame montage with configurable layout
        montage = create_frame_montage(
            frames=frames,
            timestamps=timestamps,
            grid_cols=grid_cols,
            frame_size=frame_size,
            add_labels=True,
            layout=layout,
        )
        
        # Annotate trajectory if requested
        if use_trajectory:
            from .trajectory import annotate_trajectory_on_montage
            montage = annotate_trajectory_on_montage(
                montage=montage,
                frames=frames,
                timestamps=timestamps,
                scene_type="block_scene",
                layout=layout,
                grid_cols=grid_cols,
                frame_size=frame_size,
            )
        
        # Build probability prompt with goal descriptions
        t_sec = int(timestamps[-1])  # Current time in seconds
        prompt = build_probability_prompt(
            goal_A_desc=goal_A_desc,
            goal_B_desc=goal_B_desc,
            t_sec=t_sec,
            video_id=video_id,
            n_frames=len(frames),
            mode=mode,
            use_color_markers=use_color_markers,
            use_trajectory=use_trajectory,
        )
        
        # Ask VLM for pA, pB, cue, and legibility
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=prompt)
        prompt_text = prompt_builder.get_prompt()
        
        response = self.vlm.generate(
            montage,
            prompt_text,
            do_sample=False,
            max_new_tokens=200,
        )
        
        # Parse VLM response to get pA, pB, cue, legible
        vlm_output = parse_probability_response(response)
        pA = vlm_output["pA"]
        pB = vlm_output["pB"]
        cue = vlm_output["cue"]
        legible = vlm_output["legible"]
        
        # Our code calculates choice and confidence from VLM's pA/pB
        choice, confidence = calculate_decision_metrics(pA, pB)
        
        return {
            "video_id": video_id,
            "pA": round(pA, 3),
            "pB": round(pB, 3),
            "choice": choice,
            "confidence": round(confidence, 3),
            "cue": cue,
            "legible": legible,
            "timestamps": timestamps,
            "vlm_raw_response": vlm_output["raw"],
        }
    
    def evaluate_video_at_time(
        self,
        video_path: str,
        t_current: float,
        goal_A_desc: str,
        goal_B_desc: str,
        video_id: str,
        n_frames: int = 6,
        fps: Optional[float] = None,
        mode: str = "prefix_frames",
        layout: str = "horizontal",
        grid_cols: int = 3,
        frame_size: tuple = (320, 320),
        use_color_markers: bool = False,
        scene_id: Optional[str] = None,
        use_trajectory: bool = False,
    ) -> dict:
        """
        Evaluate video showing n_frames from t=0 to t=current.
        
        Args:
            video_path: Path to video file
            t_current: Current timestamp (e.g., 5 shows frames t=0,1,2,3,4,5)
            goal_A_desc: Description of Goal A
            goal_B_desc: Description of Goal B
            video_id: Video identifier
            n_frames: Number of frames (default 6)
            fps: Video FPS (auto-detected if None)
            mode: Evaluation mode - "single_frame" or "prefix_frames"
            layout: Layout type - "horizontal", "grid", or "vertical"
            grid_cols: Grid columns for "grid" layout
            frame_size: Size to resize each frame to (width, height)
            use_color_markers: If True, add colored bounding boxes to frames
            scene_id: Scene identifier for detecting objects (e.g., "block_scene")
            use_trajectory: If True, annotate trajectory visualization on montage
        
        Returns:
            dict with evaluation results
        """
        import cv2
        
        # Get FPS if not provided
        if fps is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        
        # Extract n_frames evenly from t=0 to t=current
        frames, timestamps = extract_frame_sequence(
            video_path=video_path,
            t_current=t_current,
            fps=fps,
            n_frames=n_frames,
            sample_rate=None,
            add_color_markers=use_color_markers,
            scene_id=scene_id,
        )
        
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Evaluate using temporal montage
        return self.evaluate_temporal(
            frames=frames,
            timestamps=timestamps,
            goal_A_desc=goal_A_desc,
            goal_B_desc=goal_B_desc,
            video_id=video_id,
            mode=mode,
            layout=layout,
            grid_cols=grid_cols,
            frame_size=frame_size,
            use_color_markers=use_color_markers,
            use_trajectory=use_trajectory,
        )
