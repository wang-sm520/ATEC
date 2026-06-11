"""
GUI控制面板
"""

import math
import tkinter as tk
from tkinter import ttk
import threading
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation as R
from rich.console import Console

if TYPE_CHECKING:
    from g1_robot import G1Mujoco

console = Console()


class CommandGUI:
    """GUI for adjusting robot commands with sliders"""

    def __init__(self, g1_robot: "G1Mujoco"):
        self.g1 = g1_robot
        self.window = None
        self.is_running = False
        self.sliders = {}  # 存储所有滑块的引用

    def create_slider(self, parent, label, min_val, max_val, initial_val, row, callback, slider_key=None):
        """Create a labeled slider"""
        # Label
        tk.Label(parent, text=label, font=("Arial", 10)).grid(row=row, column=0, sticky="w", padx=5, pady=5)

        # Value display
        value_label = tk.Label(parent, text=f"{initial_val:.3f}", font=("Arial", 10, "bold"), width=8)
        value_label.grid(row=row, column=1, padx=5)

        # Slider
        slider = tk.Scale(
            parent,
            from_=min_val,
            to=max_val,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=300,
            command=lambda val: self.on_slider_change(val, value_label, callback),
        )
        slider.set(initial_val)
        slider.grid(row=row, column=2, padx=5, pady=5)

        # 存储滑块引用
        if slider_key:
            self.sliders[slider_key] = slider

        return slider, value_label

    def on_slider_change(self, value, label, callback):
        """Handle slider value change"""
        val = float(value)
        label.config(text=f"{val:.3f}")
        callback(val)

    def update_hand_quaternion_from_rpy(self, hand, roll, pitch, yaw):
        """从RPY角度更新手部姿态的四元数"""
        quat = R.from_euler("xyz", [roll, pitch, yaw]).as_quat()
        quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]  # 转换为 w, x, y, z 格式

        if hand == "left":
            self.g1.left_hand_pose_command[3:7] = quat_wxyz
        else:
            self.g1.right_hand_pose_command[3:7] = quat_wxyz

    def run(self):
        """Run the GUI in a separate thread"""
        self.window = tk.Tk()
        self.window.title("🤖 G1 Robot Command Control")
        self.window.geometry("550x950")

        # Locomotion Commands
        loco_frame = ttk.LabelFrame(self.window, text="🚶 Locomotion Commands", padding=10)
        loco_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.create_slider(
            loco_frame,
            "Linear Vel X (m/s):",
            -1.0,
            1.0,
            self.g1.vel_command_b[0],
            0,
            lambda val: self.g1.vel_command_b.__setitem__(0, val),
            "loco_vel_x",
        )

        self.create_slider(
            loco_frame,
            "Linear Vel Y (m/s):",
            -1.0,
            1.0,
            self.g1.vel_command_b[1],
            1,
            lambda val: self.g1.vel_command_b.__setitem__(1, val),
            "loco_vel_y",
        )

        self.create_slider(
            loco_frame,
            "Angular Vel Z (rad/s):",
            -3.0,
            3.0,
            self.g1.vel_command_b[2],
            2,
            lambda val: self.g1.vel_command_b.__setitem__(2, val),
            "loco_vel_z",
        )

        self.create_slider(
            loco_frame,
            "Base Height (m):",
            0.3,
            0.9,
            self.g1.base_height_command[0],
            3,
            lambda val: self.g1.base_height_command.__setitem__(0, val),
            "base_height",
        )

        # Left Hand Commands
        left_hand_frame = ttk.LabelFrame(self.window, text="👈 Left Hand Commands", padding=10)
        left_hand_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Position
        self.create_slider(
            left_hand_frame,
            "Position X (m):",
            -0.2,
            0.6,
            self.g1.left_hand_pose_command[0],
            0,
            lambda val: self.g1.left_hand_pose_command.__setitem__(0, val),
            "left_pos_x",
        )

        self.create_slider(
            left_hand_frame,
            "Position Y (m):",
            -0.1,
            0.6,
            self.g1.left_hand_pose_command[1],
            1,
            lambda val: self.g1.left_hand_pose_command.__setitem__(1, val),
            "left_pos_y",
        )

        self.create_slider(
            left_hand_frame,
            "Position Z (m):",
            -0.2,
            0.65,
            self.g1.left_hand_pose_command[2],
            2,
            lambda val: self.g1.left_hand_pose_command.__setitem__(2, val),
            "left_pos_z",
        )

        # Orientation (RPY)
        self.left_rpy = [0.0, 0.0, 0.0]  # 存储当前RPY值

        self.create_slider(
            left_hand_frame,
            "Roll (rad):",
            -math.pi,
            math.pi,
            0.0,
            3,
            lambda val: self.update_left_rpy(0, val),
            "left_roll",
        )

        self.create_slider(
            left_hand_frame,
            "Pitch (rad):",
            -math.pi,
            math.pi,
            0.0,
            4,
            lambda val: self.update_left_rpy(1, val),
            "left_pitch",
        )

        self.create_slider(
            left_hand_frame,
            "Yaw (rad):",
            -math.pi,
            math.pi,
            0.0,
            5,
            lambda val: self.update_left_rpy(2, val),
            "left_yaw",
        )

        # Right Hand Commands
        right_hand_frame = ttk.LabelFrame(self.window, text="👉 Right Hand Commands", padding=10)
        right_hand_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Position
        self.create_slider(
            right_hand_frame,
            "Position X (m):",
            -0.2,
            0.6,
            self.g1.right_hand_pose_command[0],
            0,
            lambda val: self.g1.right_hand_pose_command.__setitem__(0, val),
            "right_pos_x",
        )

        self.create_slider(
            right_hand_frame,
            "Position Y (m):",
            -0.6,
            0.1,
            self.g1.right_hand_pose_command[1],
            1,
            lambda val: self.g1.right_hand_pose_command.__setitem__(1, val),
            "right_pos_y",
        )

        self.create_slider(
            right_hand_frame,
            "Position Z (m):",
            -0.2,
            0.65,
            self.g1.right_hand_pose_command[2],
            2,
            lambda val: self.g1.right_hand_pose_command.__setitem__(2, val),
            "right_pos_z",
        )

        # Orientation (RPY)
        self.right_rpy = [0.0, 0.0, 0.0]  # 存储当前RPY值

        self.create_slider(
            right_hand_frame,
            "Roll (rad):",
            -math.pi,
            math.pi,
            0.0,
            3,
            lambda val: self.update_right_rpy(0, val),
            "right_roll",
        )

        self.create_slider(
            right_hand_frame,
            "Pitch (rad):",
            -math.pi,
            math.pi,
            0.0,
            4,
            lambda val: self.update_right_rpy(1, val),
            "right_pitch",
        )

        self.create_slider(
            right_hand_frame,
            "Yaw (rad):",
            -math.pi,
            math.pi,
            0.0,
            5,
            lambda val: self.update_right_rpy(2, val),
            "right_yaw",
        )

        # Reset button
        reset_btn = tk.Button(
            self.window,
            text="🔄 Reset to Default",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.reset_commands,
            padx=20,
            pady=10,
        )
        reset_btn.pack(pady=10)

        # Info label
        info_label = tk.Label(
            self.window,
            text="💡 Adjust sliders to change robot commands in real-time",
            font=("Arial", 9),
            fg="gray",
        )
        info_label.pack(pady=5)

        self.is_running = True
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update_left_rpy(self, index, value):
        """更新左手RPY并转换为四元数"""
        self.left_rpy[index] = value
        self.update_hand_quaternion_from_rpy("left", *self.left_rpy)

    def update_right_rpy(self, index, value):
        """更新右手RPY并转换为四元数"""
        self.right_rpy[index] = value
        self.update_hand_quaternion_from_rpy("right", *self.right_rpy)

    def reset_commands(self):
        """Reset all commands to default values"""
        # 重置命令值
        self.g1.vel_command_b[:] = [0.0, 0.0, 0.0]
        self.g1.base_height_command[0] = 0.75
        self.g1.left_hand_pose_command[:3] = [0.3, 0.2, 0.0]
        self.g1.left_hand_pose_command[3:] = [1.0, 0.0, 0.0, 0.0]
        self.g1.right_hand_pose_command[:3] = [0.3, -0.2, 0.0]
        self.g1.right_hand_pose_command[3:] = [1.0, 0.0, 0.0, 0.0]

        # 重置RPY值
        self.left_rpy = [0.0, 0.0, 0.0]
        self.right_rpy = [0.0, 0.0, 0.0]

        # 更新所有滑块位置
        default_values = {
            "loco_vel_x": 0.0,
            "loco_vel_y": 0.0,
            "loco_vel_z": 0.0,
            "base_height": 0.75,
            "left_pos_x": 0.3,
            "left_pos_y": 0.2,
            "left_pos_z": 0.0,
            "left_roll": 0.0,
            "left_pitch": 0.0,
            "left_yaw": 0.0,
            "right_pos_x": 0.3,
            "right_pos_y": -0.2,
            "right_pos_z": 0.0,
            "right_roll": 0.0,
            "right_pitch": 0.0,
            "right_yaw": 0.0,
        }

        for key, value in default_values.items():
            if key in self.sliders:
                self.sliders[key].set(value)

        console.print("[green]✅ Commands reset to default values[/green]")

    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        self.window.destroy()

    def start_in_thread(self):
        """Start GUI in a separate thread"""
        gui_thread = threading.Thread(target=self.run, daemon=True)
        gui_thread.start()
        return gui_thread
