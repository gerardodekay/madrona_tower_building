import madrona_python
import madrona_simple_example_python
import torch
import time
import sys
import random

def error_usage():
    print("interactive.py (CUDA | CPU)")
    sys.exit(1)

if len(sys.argv) < 2:
    error_usage()

exec_mode_str = sys.argv[1]

if exec_mode_str == "CUDA":
    exec_mode = madrona_simple_example_python.ExecMode.CUDA
elif exec_mode_str == "CPU":
    exec_mode = madrona_simple_example_python.ExecMode.CPU
else:
    error_usage()

sim = madrona_simple_example_python.SimpleSimulator(
        exec_mode = exec_mode,
        gpu_id = 0,
        num_worlds = 8*1024,
        num_obstacles = 5,
        enable_render = False,
        render_width = 1536,
        render_height = 1024,
)

actions = sim.action_tensor().to_torch()
positions = sim.position_tensor().to_torch()
material_positions = sim.material_position_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
print(actions.shape, actions.dtype)
print(positions.shape, actions.dtype)
print(resets.shape, resets.dtype)

num_steps = 0
while True:
    action = random.randint(1, 8)

    actions[..., 0] = action

    start_time = time.time()
    sim.step()
    end_time = time.time()
    print("Elapsed time = %s s", end_time - start_time)
