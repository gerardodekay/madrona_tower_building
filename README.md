# madrona_tower_building

To use Madrona with GPU, you need a CUDA version of at least 12.0.

## Installation

```
conda create -n madrona python=3.10
conda activate madrona
pip install torch numpy tensorboard

git clone https://github.com/gerardodekay/madrona_tower_building.git
cd madrona_tower_building
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j
cd ..
```

NOTE: For cmake, you may need to specify the cuda tookit directory as follows:

```
cmake -D CUDAToolkit_ROOT=/usr/local/cuda-12.0 ..
```

You may also need to install VulkanSDK.

To simulate the example.py, run inside the build directory:
```
MADRONA_RENDER_DEBUG_PRESENT=0 PYTHONPATH=. python ../scripts/example.py CPU # Change to CUDA for GPU implementation
```
Set MADRONA_RENDER_DEBUG_PRESENT=1 to render the environment. 
