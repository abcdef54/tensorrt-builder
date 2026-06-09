from pathlib import Path
from typing import Union, Optional
import logging
import numpy as np
import torch
import tensorrt as trt


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator for TensorRT post-training quantization.
    Feed it a DataLoader of representative samples from your dataset.

    Args:
        dataloader:  PyTorch DataLoader yielding (images, labels) batches.
        cache_path:  Path to save/load calibration cache. Reusing the cache
                     skips re-calibration on subsequent builds.

    Example:
        >>> calibrator = Int8Calibrator(train_dataloader, cache_path='calib.bin')
        >>> engine = TensorRTEngine(fp=torch.int8)
        >>> engine.from_onnx('model.onnx', 'model.engine', input_shapes={...}, calibrator=calibrator)
    """

    def __init__(self, dataloader, cache_path: str = './calib_cache.bin'):
        super().__init__()
        self.dataloader  = iter(dataloader)
        self.cache_path  = Path(cache_path)
        batch, _         = next(iter(dataloader))
        self._batch_size = batch.shape[0]
        self._input_buf  = torch.zeros_like(batch, device='cuda', dtype=torch.float32)

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: list) -> list:
        try:
            batch, _ = next(self.dataloader)
            self._input_buf.copy_(batch.to('cuda', dtype=torch.float32))
            return [self._input_buf.data_ptr()]
        except StopIteration:
            return []

    def read_calibration_cache(self) -> Optional[bytes]:
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.write_bytes(cache)


class TensorRTEngine:
    """
    TensorRT engine builder and inference runner.

    Supports FP32, FP16, and INT8 precisions with dynamic batch sizes.
    Works with .engine files, ONNX models, or PyTorch nn.Module directly.

    Three entry points:

        TensorRTEngine.from_engine(...)   — load a pre-built .engine file
        TensorRTEngine.from_onnx(...)     — build engine from an .onnx file
        TensorRTEngine.from_pytorch(...)  — build engine from a PyTorch nn.Module

    Example — from .engine:
        >>> engine = TensorRTEngine.from_engine('model.engine', fp=torch.float16)
        >>> output = engine(torch.rand(1, 3, 32, 32).cuda())

    Example — from ONNX:
        >>> engine = TensorRTEngine.from_onnx(
        ...     onnx_path='model.onnx',
        ...     engine_path='model.engine',
        ...     input_shapes={
        ...         'input': {
        ...             'min': (1, 3, 32, 32),
        ...             'opt': (32, 3, 32, 32),
        ...             'max': (256, 3, 32, 32)
        ...         }
        ...     },
        ...     fp=torch.float16
        ... )
        >>> output = engine(torch.rand(32, 3, 32, 32).cuda())

    Example — from PyTorch:
        >>> engine = TensorRTEngine.from_pytorch(
        ...     model=my_model,
        ...     dummy_input=torch.rand(1, 3, 32, 32),
        ...     engine_path='model.engine',
        ...     input_shapes={
        ...         'input': {
        ...             'min': (1, 3, 32, 32),
        ...             'opt': (32, 3, 32, 32),
        ...             'max': (256, 3, 32, 32)
        ...         }
        ...     },
        ...     fp=torch.float16
        ... )

    Example — INT8:
        >>> calibrator = Int8Calibrator(train_dataloader, cache_path='calib.bin')
        >>> engine = TensorRTEngine.from_onnx(
        ...     onnx_path='model.onnx',
        ...     engine_path='model.engine',
        ...     input_shapes={...},
        ...     fp=torch.int8,
        ...     calibrator=calibrator
        ... )

    Note:
        TensorRT engine files are platform-specific. An engine built on Linux
        will not run on Windows, and vice versa — even with identical hardware
        and CUDA versions. Always build the engine on the target deployment machine.
    """

    SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.int8}

    def __init__(self, fp: torch.dtype = torch.float32) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA is not available. TensorRT requires a CUDA-capable GPU.'
            )
        if fp not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f'Unsupported dtype: {fp}. '
                f'Supported dtypes: {self.SUPPORTED_DTYPES}'
            )
        self.fp = fp
        self.engine = None
        self.context = None
        self._input_names = []
        self._output_names= []
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._setup_logger()

    def _setup_logger(self) -> None:
        logging.basicConfig(level=logging.INFO, format='[TensorRTEngine] %(message)s')
        self.log = logging.getLogger('TensorRTEngine')


    @classmethod
    def from_engine(
        cls,
        engine_path: str,
        fp: torch.dtype = torch.float32
    ) -> 'TensorRTEngine':
        """
        Load a previously built .engine file.

        Args:
            engine_path:  Path to the .engine file.
            fp:           Precision dtype the engine was built with. Default: float32.

        Returns:
            Initialized TensorRTEngine ready for inference.

        Raises:
            FileNotFoundError: If engine_path does not exist.
            RuntimeError:      If deserialization fails (wrong platform, TRT version,
                               or GPU architecture mismatch).

        Example:
            >>> engine = TensorRTEngine.from_engine('model.engine', fp=torch.float16)
            >>> output = engine(torch.rand(1, 3, 32, 32).cuda())
        """
        instance = cls(fp=fp)
        instance._load(engine_path)
        return instance

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str,
        engine_path: str,
        input_shapes: dict[str, dict[str, tuple]],
        fp: torch.dtype = torch.float32,
        workspace_gb: float = 2.0,
        calibrator: Optional[Int8Calibrator] = None,
    ) -> 'TensorRTEngine':
        """
        Build a TensorRT engine from an ONNX model file.

        Args:
            onnx_path:      Path to the input .onnx file.
            engine_path:    Path to save the built .engine file.
                            Parent directories are created automatically.
            input_shapes:   Dict mapping each input tensor name to its
                            min / opt / max shapes for dynamic batching.
                            Example:
                            {
                                'input': {
                                    'min': (1, 3, 32, 32),
                                    'opt': (32, 3, 32, 32),
                                    'max': (256, 3, 32, 32)
                                }
                            }
            fp:             Precision. One of torch.float32, torch.float16,
                            torch.int8. Default: float32.
            workspace_gb:   Max GPU memory (GB) for the TensorRT build
                            workspace. Default: 2.0.
            calibrator:     Required when fp=torch.int8. Pass an
                            Int8Calibrator instance built from representative
                            training data.

        Returns:
            Initialized TensorRTEngine ready for inference.

        Raises:
            FileNotFoundError: If onnx_path does not exist.
            ValueError:        If input_shapes is empty or INT8 is requested
                               without a calibrator.
            RuntimeError:      If ONNX parsing or engine build fails.

        Example:
            >>> engine = TensorRTEngine.from_onnx(
            ...     onnx_path='model.onnx',
            ...     engine_path='model.engine',
            ...     input_shapes={
            ...         'input': {
            ...             'min': (1, 3, 32, 32),
            ...             'opt': (32, 3, 32, 32),
            ...             'max': (256, 3, 32, 32)
            ...         }
            ...     },
            ...     fp=torch.float16
            ... )
        """
        instance = cls(fp=fp)
        instance._build(onnx_path, engine_path, input_shapes, workspace_gb, calibrator)
        return instance

    @classmethod
    def from_pytorch(
        cls,
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        engine_path: str,
        input_shapes: dict[str, dict[str, tuple]],
        fp: torch.dtype = torch.float32,
        input_names: list[str] = None,
        output_names: list[str] = None,
        opset_version: int = 18,
        workspace_gb: float = 2.0,
        calibrator: Optional[Int8Calibrator] = None,
    ) -> 'TensorRTEngine':
        """
        Build a TensorRT engine directly from a PyTorch nn.Module.
        Internally exports to a temporary ONNX file then builds the engine.
        The temporary ONNX file is deleted after the build regardless of success.

        Args:
            model:          Trained PyTorch model. Will be set to eval() automatically.
            dummy_input:    Sample input tensor with correct shape and dtype.
                            Does not need to be on CUDA.
            engine_path:    Path to save the built .engine file.
            input_shapes:   Same format as from_onnx().
            fp:             Precision. One of torch.float32, torch.float16,
                            torch.int8. Default: float32.
            input_names:    ONNX input node names. Default: ['input'].
            output_names:   ONNX output node names. Default: ['output'].
            opset_version:  ONNX opset version. Default: 18.
            workspace_gb:   Max GPU memory (GB) for TensorRT build workspace. Default: 2.0.
            calibrator:     Required when fp=torch.int8.

        Returns:
            Initialized TensorRTEngine ready for inference.

        Example:
            >>> engine = TensorRTEngine.from_pytorch(
            ...     model=my_model,
            ...     dummy_input=torch.rand(1, 3, 32, 32),
            ...     engine_path='model.engine',
            ...     input_shapes={
            ...         'input': {
            ...             'min': (1, 3, 32, 32),
            ...             'opt': (32, 3, 32, 32),
            ...             'max': (256, 3, 32, 32)
            ...         }
            ...     },
            ...     fp=torch.float16
            ... )
        """
        input_names  = input_names  or ['input']
        output_names = output_names or ['output']

        instance = cls(fp=fp)
        tmp_onnx = Path(engine_path).with_suffix('.tmp.onnx')

        try:
            instance.log.info('Exporting PyTorch model to ONNX...')
            model.eval()
            torch.onnx.export(
                model,
                dummy_input,
                str(tmp_onnx),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    name: {0: 'batch_size'}
                    for name in input_names + output_names
                },
                opset_version=opset_version
            )
            instance.log.info(f'ONNX export complete → {tmp_onnx}')
            instance._build(
                str(tmp_onnx), engine_path, input_shapes, workspace_gb, calibrator
            )
        finally:
            if tmp_onnx.exists():
                tmp_onnx.unlink()

        return instance


    def _build(
        self,
        onnx_path: str,
        engine_path: str,
        input_shapes: dict[str, dict[str, tuple]],
        workspace_gb: float,
        calibrator: Optional[Int8Calibrator],
    ) -> None:
        onnx_path   = Path(onnx_path)
        engine_path = Path(engine_path)

        if not onnx_path.exists():
            raise FileNotFoundError(f'ONNX file not found: {onnx_path}')
        if not input_shapes:
            raise ValueError('input_shapes must be provided.')
        if self.fp == torch.int8 and calibrator is None:
            raise ValueError(
                'INT8 quantization requires a calibrator. '
                'Pass an Int8Calibrator instance via the calibrator argument.'
            )

        self.log.info(f'Building engine from {onnx_path}')
        self.log.info(f'Precision: {self._dtype_str()}')

        builder = trt.Builder(self._logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self._logger)

        if not parser.parse_from_file(str(onnx_path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError('ONNX parse failed:\n' + '\n'.join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(workspace_gb * (1 << 30))
        )

        if self.fp == torch.float16:
            if not builder.platform_has_fast_fp16:
                self.log.warning(
                    'GPU does not support fast FP16. Falling back to FP32.'
                )
            else:
                config.set_flag(trt.BuilderFlag.FP16)

        elif self.fp == torch.int8:
            if not builder.platform_has_fast_int8:
                self.log.warning(
                    'GPU does not support fast INT8. Falling back to FP32.'
                )
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calibrator

        profile = builder.create_optimization_profile()
        for input_name, shapes in input_shapes.items():
            profile.set_shape(
                input_name,
                min=shapes['min'],
                opt=shapes['opt'],
                max=shapes['max']
            )
        config.add_optimization_profile(profile)

        self.log.info('Building engine — this may take several minutes...')
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(
                'Engine build failed. '
                'Ensure your ONNX model uses supported ops and that the opset '
                'version is compatible with your TensorRT installation.'
            )

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(serialized)
        self.log.info(f'Engine saved to {engine_path}')

        runtime = trt.Runtime(self._logger)
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self._init_context()

    def _load(self, engine_path: str) -> None:
        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f'Engine file not found: {engine_path}')

        self.log.info(f'Loading engine from {engine_path}')
        runtime = trt.Runtime(self._logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())

        if self.engine is None:
            raise RuntimeError(
                f'Failed to deserialize engine: {engine_path}\n'
                'Common causes: engine built on a different platform, '
                'TensorRT version, or GPU architecture.'
            )
        self._init_context()
        self.log.info('Engine loaded successfully.')

    def _init_context(self) -> None:
        self.context       = self.engine.create_execution_context()
        self._input_names  = []
        self._output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

        self.log.info(f'Inputs:  {self._input_names}')
        self.log.info(f'Outputs: {self._output_names}')


    def __call__(
        self,
        *inputs: torch.Tensor
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Run inference.

        Args:
            *inputs: One tensor per model input, in order.
                     Tensors are automatically moved to CUDA and cast to
                     the engine's dtype.

        Returns:
            Single np.ndarray if the model has one output.
            List of np.ndarray if the model has multiple outputs.

        Raises:
            RuntimeError: If the engine is not initialized.
            ValueError:   If the number of inputs doesn't match the model.

        Example:
            >>> output = engine(torch.rand(32, 3, 32, 32))
            >>> predicted_classes = output.argmax(axis=1)
        """
        if self.context is None:
            raise RuntimeError(
                'Engine not initialized. '
                'Use from_engine(), from_onnx(), or from_pytorch() to create an engine.'
            )
        if len(inputs) != len(self._input_names):
            raise ValueError(
                f'Expected {len(self._input_names)} input(s), got {len(inputs)}. '
                f'Input names: {self._input_names}'
            )

        # bind inputs
        for name, x in zip(self._input_names, inputs):
            x = x.to(device='cuda', dtype=self.fp).contiguous()
            self.context.set_input_shape(name, x.shape)
            self.context.set_tensor_address(name, x.data_ptr())

        # allocate and bind outputs
        output_tensors = []
        for name in self._output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            out   = torch.empty(shape, dtype=self.fp, device='cuda')
            self.context.set_tensor_address(name, out.data_ptr())
            output_tensors.append(out)

        self.context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream
        )
        torch.cuda.synchronize()

        results = [t.cpu().numpy() for t in output_tensors]
        return results[0] if len(results) == 1 else results


    def _dtype_str(self) -> str:
        return {
            torch.float32: 'FP32',
            torch.float16: 'FP16',
            torch.int8:    'INT8',
        }.get(self.fp, str(self.fp))

    def __repr__(self) -> str:
        status = 'loaded' if self.context else 'not loaded'
        return (
            f'TensorRTEngine(\n'
            f'  precision = {self._dtype_str()},\n'
            f'  status    = {status},\n'
            f'  inputs    = {self._input_names},\n'
            f'  outputs   = {self._output_names}\n'
            f')'
        )