import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RifeNCNNOptions(BaseModel):
    model_path: Path = Field(..., description="Path to RIFE model directory")
    input_path: Path = Field(..., description="Path to source frames directory")
    output_path: Optional[Path] = Field(None, description="Path to output frames directory")
    num_frame: Optional[int] = Field(None, description="Number of frames to generate (default N*2)")
    time_step: float = Field(0.5, description="Time step for interpolation (default 0.5)", gt=0.0, le=1.0)
    gpu_id: Optional[int | list[int]] = Field(
        None, description="GPU ID(s) to use (default: auto, -1 for CPU)"
    )
    load_threads: int = Field(1, description="Number of threads for frame loading", gt=0)
    process_threads: int = Field(2, description="Number of threads used for frame processing", gt=0)
    save_threads: int = Field(2, description="Number of threads for frame saving", gt=0)
    spatial_tta: bool = Field(False, description="Enable spatial TTA mode")
    temporal_tta: bool = Field(False, description="Enable temporal TTA mode")
    uhd: bool = Field(False, description="Enable UHD mode")
    verbose: bool = Field(False, description="Enable verbose logging")

    def get_args(self, frame_multiplier: int = 7) -> list[str]:
        """Generate arguments to pass to rife-ncnn-vulkan.

        Frame multiplier is used to calculate the number of frames to generate, if num_frame is not set.
        """
        if self.output_path is None:
            self.output_path = self.input_path.joinpath("out")

        # calc num frames
        if self.num_frame is None:
            num_src_frames = len([x for x in self.input_path.glob("*.png") if x.is_file()])
            logger.info(f"Found {num_src_frames} source frames, using multiplier {frame_multiplier}")
            num_frame = num_src_frames * frame_multiplier
            logger.info(f"We will generate {num_frame} frames")
        else:
            num_frame = self.num_frame

        # GPU ID and process threads are comma-separated lists, so we need to convert them to strings
        if self.gpu_id is None:
            gpu_id = "auto"
            process_threads = self.process_threads
        elif isinstance(self.gpu_id, list):
            gpu_id = ",".join([str(x) for x in self.gpu_id])
            process_threads = ",".join([str(self.process_threads) for _ in self.gpu_id])
        else:
            gpu_id = str(self.gpu_id)
            process_threads = str(self.process_threads)

        # Build args list
        args_list = [
            "-i",
            f"{self.input_path.resolve()}/",
            "-o",
            f"{self.output_path.resolve()}/",
            "-m",
            f"{self.model_path.resolve()}/",
            "-n",
            num_frame,
            "-s",
            self.time_step,
            "-g",
            gpu_id,
            "-j",
            f"{self.load_threads}:{process_threads}:{self.save_threads}",
        ]

        # Add flags if set
        if self.spatial_tta:
            args_list.append("-x")
        if self.temporal_tta:
            args_list.append("-z")
        if self.uhd:
            args_list.append("-u")
        if self.verbose:
            args_list.append("-v")

        # Convert all args to strings and return
        return [str(x) for x in args_list]
