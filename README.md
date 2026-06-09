# tensorrt-builder

A small Python wrapper for building and running TensorRT engines from ONNX models or PyTorch modules.

**This is a learning project.** I wrote it to understand how TensorRT and ONNX work under the hood. If you need production-ready tools, look at [Torch-TensorRT](https://github.com/pytorch/TensorRT), [ONNX Runtime](https://onnxruntime.ai/), or NVIDIA's own [TensorRT samples](https://github.com/NVIDIA/TensorRT) — they cover the same ground and more.

## What it does

- Build TensorRT engines from `.onnx` files
- Build engines directly from a `torch.nn.Module` (exports to ONNX internally)
- Load and run pre-built `.engine` files
- Supports FP32, FP16, and INT8 (with post-training calibration)
- Dynamic batch sizes via optimization profiles

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- TensorRT >= 8.6
- PyTorch >= 2.0
- onnxruntime-gpu >= 1.16

```
pip install -r requirements.txt
```

## Usage

### From ONNX

```python
from tensorrt_engine import TensorRTEngine

engine = TensorRTEngine.from_onnx(
    onnx_path='model.onnx',
    engine_path='model.engine',
    input_shapes={
        'input': {
            'min': (1, 3, 32, 32),
            'opt': (32, 3, 32, 32),
            'max': (256, 3, 32, 32),
        }
    },
    fp=torch.float16,
)

output = engine(torch.rand(32, 3, 32, 32).cuda())
```

### From PyTorch

```python
engine = TensorRTEngine.from_pytorch(
    model=my_model,
    dummy_input=torch.rand(1, 3, 32, 32),
    engine_path='model.engine',
    input_shapes={
        'input': {
            'min': (1, 3, 32, 32),
            'opt': (32, 3, 32, 32),
            'max': (256, 3, 32, 32),
        }
    },
    fp=torch.float16,
)
```

### From a pre-built engine

```python
engine = TensorRTEngine.from_engine('model.engine', fp=torch.float16)
output = engine(torch.rand(1, 3, 32, 32).cuda())
```

### INT8 quantization

```python
from tensorrt_engine import TensorRTEngine, Int8Calibrator

calibrator = Int8Calibrator(train_dataloader, cache_path='calib.bin')
engine = TensorRTEngine.from_onnx(
    onnx_path='model.onnx',
    engine_path='model.engine',
    input_shapes={...},
    fp=torch.int8,
    calibrator=calibrator,
)
```

## Good to know

- Engine files are **not portable**. An engine built on one machine won't work on another with a different OS, GPU, or TensorRT version. Always build on the target machine.
- INT8 requires a calibrator with representative data from your training set.

## References

1. [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html)
2. [PyTorch ONNX Export](https://docs.pytorch.org/docs/2.12/onnx.html)
3. [ONNX Runtime](https://onnxruntime.ai/)

## License

MIT