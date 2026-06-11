"""
Minimal default runner for G1 locomotion and manipulation.

This keeps only the default `main()` path from the full project:
MuJoCo viewer + command GUI + ONNX policy inference + optional recording.
"""

import time

import hydra
import mujoco
import mujoco.viewer
from omegaconf import DictConfig
from rich.console import Console

from command_gui import CommandGUI
from g1_robot import G1Mujoco


console = Console()


def print_banner() -> None:
    console.print("\n[bold cyan]===============================================[/bold cyan]")
    console.print("[bold yellow]G1 Humanoid Robot - Minimal Default Run[/bold yellow]")
    console.print("[bold cyan]===============================================[/bold cyan]\n")
    console.print("[green]Command GUI:[/green]")
    console.print("  - Sliders control velocity, base height, and hand targets")
    console.print("[green]Recording:[/green]")
    console.print("  - Press 'R' in the MuJoCo viewer to start/stop recording")
    console.print("  - Videos and logs are saved under ./log/session_*/\n")


@hydra.main(config_path="configs", config_name="g1_loco_manip", version_base="1.2")
def main(cfg: DictConfig) -> None:
    g1 = G1Mujoco(cfg)
    print_banner()

    console.print("[cyan]Starting command GUI...[/cyan]")
    g1.gui = CommandGUI(g1)
    g1.gui.start_in_thread()
    time.sleep(0.5)
    console.print("[green]GUI started.[/green]\n")

    recording_key_pressed = False
    last_key_time = 0.0

    def key_callback(key: int) -> None:
        nonlocal recording_key_pressed, last_key_time
        current_time = time.time()
        if key == ord("R") and (current_time - last_key_time) > 0.5:
            recording_key_pressed = True
            last_key_time = current_time

    with mujoco.viewer.launch_passive(g1.model, g1.data, key_callback=key_callback) as viewer:
        obs = g1.get_all_obs()
        frame_count = 0
        last_status_update = 0

        while viewer.is_running():
            step_start = time.time()
            frame_count += 1

            if recording_key_pressed:
                recording_key_pressed = False
                if g1.video_recorder.is_recording:
                    g1.video_recorder.stop_recording()
                else:
                    g1.video_recorder.start_recording()

            inference_start = time.time()
            action = g1.act(obs)
            inference_end = time.time()
            g1.inference_time = inference_end - inference_start
            g1.inference_fps = 1.0 / g1.inference_time if g1.inference_time > 0 else 0.0

            obs = g1.step(action)

            if g1.video_recorder.is_recording:
                g1.video_recorder.capture_frame(g1.data)

            if frame_count - last_status_update >= 50:
                console.clear()
                console.print(g1.create_status_display())
                last_status_update = frame_count

            viewer.sync()

            wait_time = cfg.simulation_dt * cfg.control_decimation - (time.time() - step_start)
            if wait_time > 0:
                time.sleep(wait_time)

    if g1.video_recorder.is_recording:
        g1.video_recorder.stop_recording()
    g1.close_video_recorder()


if __name__ == "__main__":
    main()
