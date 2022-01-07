# nlm-cuda

This repository implements [Non-Local Means Denoising Method](https://ieeexplore.ieee.org/document/1467423/) on both CPU and CUDA to accelerate.

<p align="center">
    <img src="demo.png"></img>    
</p>

## FYI

- Not recommended for production use.

- Integral image optimization strategy is not implemented here.
- Only gray scale (single channel) image is supported now.

## Build

CMake is used to generate. RapidJSON is one of dependencies. CUDA is needed for GPU version.

- Check CMakeLists.txt in root directory. Set `RapidJSON_DIR` according to where your RapidJSON library is.
- Maybe you should modify CUDA BlockThread according to the Compute Capability of your NVIDIA GPU to achieve best concurrency. (https://github.com/z0gSh1u/nlm-cuda/blob/34ec8b877aa5bc07459bf6c0021d8b953ecc4f7f/nlm-cuda/nlm-cuda.cu#L218)
- Generate with CMake. And then compile to your platform.

## Usage

- A config file is needed. Refer to [NLMConfig.json](./NLMConfig.json) for a sample.

- Call

  ```sh
  path/to/nlm-cuda config.json
  ```

## License

MIT

