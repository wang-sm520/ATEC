"""CLI argument definitions for Task E demo collection.
"""


def add_collect_demo_args(parser) -> None:
    parser.add_argument(
        "--num_demos", type=int, default=50,
        help="Number of successful demos to collect.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./datasets/atec_task_e",
        help="Directory to save trajectory.hdf5 and trajectory.json.",
    )
    parser.add_argument(
        "--pick_objects", type=int, nargs="+", default=[1, 2, 3],
        metavar="N",
        help="Which objects to pick (subset of {1,2,3}). Default: all three.",
    )
    parser.add_argument(
        "--save_video", action="store_true", default=False,
        help="Save an MP4 per demo for visualization "
             "(requires: pip install imageio imageio-ffmpeg).",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Output directory for MP4 files. Defaults to <output_dir>/videos/.",
    )
    parser.add_argument(
        "--save_images", action="store_true", default=False,
        help="Save raw RGB frames into HDF5 under traj_N/images/rgb (T,H,W,3) "
             "for ACT RGBD training. Shares the camera with --save_video.",
    )
    parser.add_argument(
        "--only_success", action="store_true", default=False,
        help="Discard demos where not all picked objects ended up in the basket.",
    )
    parser.add_argument(
        "--full_order_123", action="store_true", default=False,
        help="Force Task E collection order to object_1, object_2, object_3 and require full-order success.",
    )
    parser.add_argument(
        "--optimized_grasp_flow", action="store_true", default=False,
        help="Use optimized pick-place dwell times for faster Task E grasp data collection.",
    )
    parser.add_argument(
        "--grasp_z_offsets", type=str, nargs="*", default=None,
        metavar="OBJECT_ID:OFFSET",
        help="Per-object grasp height offsets, for example: 1:0.080 2:0.085 3:0.075.",
    )
    parser.add_argument(
        "--grasp_offset_json", type=str, default=None,
        help="Path to sweep JSON containing best_offsets for object-specific grasp heights.",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=None,
        help="Maximum attempts before exiting non-zero if num_demos successful demos are not collected.",
    )
    parser.add_argument(
        "--tool_center_offset_local", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="Local xyz offset from gripper_base to jaw center, used to compensate fixed grasp bias.",
    )
