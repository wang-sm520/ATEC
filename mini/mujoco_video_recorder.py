import os
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np


class MuJoCoVideoRecorder:
    """Small multi-camera recorder used by the minimal G1 runner."""

    def __init__(self, fps=50, output_dir="log", resolution=(1920, 1080), streaming=True):
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.width, self.height = resolution
        self.streaming = streaming
        self.cameras = {}
        self.is_recording = False
        self.session_dir = None
        self.start_time = None
        self.frame_count_streaming = 0
        self.model = None
        self.renderer = None
        self.data_log = {
            "time": [],
            "qpos": [],
            "qvel": [],
            "ctrl": [],
            "xpos": [],
            "xquat": [],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def frames(self):
        if not self.cameras:
            return []
        return next(iter(self.cameras.values()))["frames"]

    def frame_count(self):
        if self.streaming:
            return self.frame_count_streaming * max(len(self.cameras), 1)
        return sum(len(camera["frames"]) for camera in self.cameras.values())

    def add_camera_config(self, name, config):
        camera = mujoco.MjvCamera()
        self._setup_camera(camera, config)
        self.cameras[name] = {"camera": camera, "frames": [], "writer": None, "video_path": None}

    def initialize_renderer(self, model):
        self.model = model

    def _ensure_renderer(self):
        if self.renderer is None:
            if self.model is None:
                raise RuntimeError("Video recorder model is not initialized.")
            self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self.renderer.scene.maxgeom = 2000

    def _setup_camera(self, camera, config):
        mode = config.get("mode", "fixed")
        if mode == "track":
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = config.get("trackbodyid", 1)
        else:
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE

        if "lookat" in config:
            camera.lookat[:] = config["lookat"]
        if "distance" in config:
            camera.distance = config["distance"]
        if "azimuth" in config:
            camera.azimuth = config["azimuth"]
        if "elevation" in config:
            camera.elevation = config["elevation"]

    def start_recording(self, trajectory_name="recording"):
        if self.is_recording:
            return False

        self.is_recording = True
        self.frame_count_streaming = 0
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

        for key in self.data_log:
            self.data_log[key] = []

        if self.streaming:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            for name, camera_data in self.cameras.items():
                suffix = f"_{name}" if len(self.cameras) > 1 else ""
                video_path = self.session_dir / f"{trajectory_name}{suffix}.mp4"
                camera_data["writer"] = cv2.VideoWriter(
                    str(video_path), fourcc, self.fps, (self.width, self.height)
                )
                camera_data["video_path"] = video_path
        else:
            for camera_data in self.cameras.values():
                camera_data["frames"] = []

        print(f"Recording started: {self.session_dir}")
        return True

    def stop_recording(self):
        if not self.is_recording:
            return False

        self.is_recording = False
        for camera_data in self.cameras.values():
            writer = camera_data.get("writer")
            if writer is not None:
                writer.release()
                camera_data["writer"] = None

        self.save_data_log()
        duration = time.time() - self.start_time if self.start_time else 0.0
        print(f"Recording stopped: {duration:.1f}s, {self.frame_count()} frames")
        return True

    def capture_frame(self, data, draw_callback=None):
        if not self.is_recording:
            return False
        self._ensure_renderer()
        self._log_mjdata(data)

        for camera_data in self.cameras.values():
            self.renderer.update_scene(data, camera=camera_data["camera"])
            if draw_callback is not None:
                draw_callback(self.renderer.scene)
            image = self.renderer.render()

            if self.streaming:
                writer = camera_data.get("writer")
                if writer is not None:
                    writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            else:
                camera_data["frames"].append(image.copy())

        if self.streaming:
            self.frame_count_streaming += 1
        return True

    def save_video(self, trajectory_name="recording"):
        if self.streaming:
            return True
        if self.session_dir is None:
            self.session_dir = self.output_dir

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for name, camera_data in self.cameras.items():
            frames = camera_data["frames"]
            if not frames:
                continue
            suffix = f"_{name}" if len(self.cameras) > 1 else ""
            video_path = self.session_dir / f"{trajectory_name}{suffix}.mp4"
            writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.width, self.height))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        self.save_data_log(trajectory_name=trajectory_name)
        return True

    def save_data_log(self, trajectory_name="recording"):
        if self.session_dir is None or not self.data_log["time"]:
            return False

        log_path = self.session_dir / f"{trajectory_name}_mjdata.npz"
        np.savez_compressed(
            log_path,
            time=np.array(self.data_log["time"]),
            qpos=np.array(self.data_log["qpos"]),
            qvel=np.array(self.data_log["qvel"]),
            ctrl=np.array(self.data_log["ctrl"]),
            xpos=np.array(self.data_log["xpos"]),
            xquat=np.array(self.data_log["xquat"]),
            fps=self.fps,
        )
        print(f"Data log saved: {log_path}")
        return True

    def _log_mjdata(self, data):
        self.data_log["time"].append(data.time)
        self.data_log["qpos"].append(data.qpos.copy())
        self.data_log["qvel"].append(data.qvel.copy())
        self.data_log["ctrl"].append(data.ctrl.copy())
        self.data_log["xpos"].append(data.xpos.copy())
        self.data_log["xquat"].append(data.xquat.copy())

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class VideoRecordingMixin:
    def init_video_recording(self, fps=50, output_dir="log", resolution=(1920, 1080)):
        self.video_recorder = MuJoCoVideoRecorder(fps=fps, output_dir=output_dir, resolution=resolution)

    def setup_video_recorder(self, model):
        self.video_recorder.initialize_renderer(model)

    def close_video_recorder(self):
        if hasattr(self, "video_recorder"):
            self.video_recorder.close()
