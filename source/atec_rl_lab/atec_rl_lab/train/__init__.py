
try:
    from .locomotion import *
except ModuleNotFoundError as exc:
    optional_modules = {"gymnasium", "isaaclab", "isaaclab_tasks", "isaaclab_rl"}
    if exc.name not in optional_modules:
        raise
