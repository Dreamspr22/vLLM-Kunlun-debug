"""kunlun"""
import psutil
import torch

from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum, _Backend
from typing import Optional, Union
import vllm.envs as envs
from vllm.logger import init_logger


logger = init_logger(__name__)

class KunlunPlatform(Platform):
    """KunlunPlatform"""
    _enum = PlatformEnum.CUDA 
    dist_backend:str = "nccl"
    ray_device_key: str = "GPU"
    device_name: str = "xpu"

    @property
    def device_type(self):
        """Returns the device type, which is fixed as 'cuda'.
        """
        return "cuda"

    def is_kunlun(self) -> bool:
        """is_kunlun"""
        return self._enum == PlatformEnum.CUDA

    def is_cuda(self) -> bool:
        """is_cuda"""
        return False

    def is_rocm(self) -> bool:
        """is_rocm"""
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        """is_tpu"""
        return self._enum == PlatformEnum.TPU

    def is_hpu(self) -> bool:
        """is_hpu"""
        return self._enum == PlatformEnum.HPU

    def is_xpu(self) -> bool:
        """is_xpu"""
        return self._enum == PlatformEnum.XPU

    def is_cpu(self) -> bool:
        """is_cpu"""
        return self._enum == PlatformEnum.CPU

    def is_neuron(self) -> bool:
        """is_neuron"""
        return self._enum == PlatformEnum.NEURON

    def is_out_of_tree(self) -> bool:
        """is_out_of_tree"""
        return self._enum == PlatformEnum.OOT

    def is_cuda_alike(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    def is_sleep_mode_available(self) -> bool:
        """is_sleep_mode_available"""
        return self._enum == PlatformEnum.CUDA

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Returns the device name, which defaults to "kunlun".

        Args:
            device_id (int, optional): The device ID, default is 0. Ignored in this method. Defaults to 0.

        Returns:
            str: The device name, which is fixed as "kunlun".
        """
        return "kunlun"

    @classmethod
    def get_piecewise_backend_cls(cls) -> str:
        return "vllm.compilation.cuda_piecewise_backend.CUDAPiecewiseBackend"  # noqa

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"  # noqa

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Returns the total memory size of the device in bytes (B). Defaults to the total memory size of the first device.
        If the `device_id` parameter is not an integer or exceeds the available device range, a ValueError will be raised.

        Args:
            device_id (int, optional): The device ID, default is 0. Defaults to 0.

        Raises:
            ValueError: If the `device_id` parameter is not an integer or exceeds the available device range, this exception is raised.

        Returns:
            int: The total memory size of the device in bytes (B).
        """
        return psutil.virtual_memory().total

    @classmethod
    def inference_mode(cls):
        """Returns a context manager that disables gradient computation.
        """
        return torch.no_grad()

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        """get_device_capability"""
        major, minor = torch.cuda.get_device_capability()
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Updates the default values of various components based on the configuration.
        If not specified, automatically selects the worker class based on certain conditions.
        If the block size is not set in the cache configuration, it is set to 16.
        If using MLA and `VLLM_ATTENTION_BACKEND` is not set or is set to "FLASHMLA",
        the cache block size is set to 64.
        If running in DeepEP high throughput backend, data parallelism greater than 1, and CUDA graph mode,
        it forces the use of eager mode, as DP + DeepEP high throughput kernels are not CUDA graph compatible,
        and using DeepEP low latency kernels can resolve this issue.

        Args:
            vllm_config (VllmConfig): VLLM configuration object.

        Raises:
            NotImplementedError: If multi-step scheduling is used on vLLM V1, this exception is raised.
            Please remove the --num-scheduler-steps argument from the command line.
            NotImplementedError: If MLA is used on vLLM V1, this exception is raised.
            Please ensure that the `VLLM_ATTENTION_BACKEND` environment variable is set before using MLA.

        Returns:
            None: No return value.
        """
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            if vllm_config.speculative_config:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                    parallel_config.sd_worker_cls = \
                        "vllm.worker.worker.Worker"
            else:
                print(f"envs.VLLM_USE_V1 = {envs.VLLM_USE_V1}")
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                            "vllm.v1.worker.gpu_worker.Worker"
                else:
                    parallel_config.worker_cls = "vllm.worker.worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        if model_config is not None and model_config.use_mla:
            # if `VLLM_ATTENTION_BACKEND` is not set and we are using MLA, then
            # we default to FlashMLA backend, so we need to force the blocksize
            # here
            use_flashmla = (envs.VLLM_ATTENTION_BACKEND is None \
                or envs.VLLM_ATTENTION_BACKEND == "FLASHMLA")
            from vllm.attention.ops.flashmla import is_flashmla_supported
            if use_flashmla and is_flashmla_supported()[0] \
                and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLA backend.")

        if (envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput"
                and parallel_config.data_parallel_size > 1
                and vllm_config.compilation_config.use_cudagraph):
            logger.info(
                "Data Parallel: Forcing enforce eager to be True since DP "
                "with DeepEP high-throughput kernels are not CUDA Graph "
                "compatible. The DeepEP low-latency kernels are CUDA Graph "
                "compatible. Set the all_to_all backend to deepep_low_latency "
                "to use those kernels instead.")
            vllm_config.compilation_config.use_cudagraph = False
            vllm_config.model_config.enforce_eager = True
            # TODO (varun): Turning this ON gives incorrect results for the
            # Deepseek-V2-lite model.
            vllm_config.compilation_config.use_inductor = False
        if vllm_config.compilation_config.use_cudagraph and envs.VLLM_USE_V1:
            vllm_config.compilation_config.custom_ops = ["all"]
            vllm_config.compilation_config.pass_config.enable_fusion = False
            vllm_config.compilation_config.use_inductor = False

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                             kv_cache_dtype, block_size, use_v1, use_mla,use_sink):
        """
            Returns the class of attention backend based on the selected backend and other parameters.
        
        Args:
            selected_backend (str): Selected backend name. Currently supported backends are 'kunlun' and 'default'.
            head_size (int): Size of the attention heads.
            dtype (torch.dtype): Data type of the input tensor.
            kv_cache_dtype (torch.dtype): Data type of the key-value cache.
            block_size (int): Block size used in the attention computation.
            use_v1 (bool, optional): Whether to use v1 version of the backend. Defaults to False.
            use_mla (bool, optional): Whether to use MLA version of the backend. Defaults to False.
        
        Returns:
            str: Class name of the attention backend.
        """
        if use_v1:
            return "vllm_kunlun.v1.attention.backends.kunlun_attn.KunlunAttentionBackend"
        elif not use_mla:                     
            return "vllm_kunlun.ops.attention.backends.kunlun_attn.KunlunAttentionBackend"
        else:
            return "vllm_kunlun.attention.backends.kunlun_mla.KunlunMLAAttentionBackend"

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        """Gets the current memory usage of the device, including allocated and max allocated.
        If no device is specified, defaults to the current context's device.

        Args:
            device (Optional[torch.types.Device], optional): Optional device object, defaults to None. Defaults to the current context's device.

        Returns:
            float: Returns a float representing the current memory usage of the device, in bytes.

            Raises:
                None.
        """
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """Checks if asynchronous output is supported.
        By default, Kunlun does not support asynchronous output.

        Args:
            enforce_eager (Optional[bool], optional): Whether to enforce eager execution. Defaults to None.
                None means not to force eager execution, but to automatically select based on the current environment.

        Returns:
            bool: True means asynchronous output is supported, False means asynchronous output is not supported.
        """
        # Assume Kunlun does not support asynchronous output
        return False

    @classmethod
    def supports_v1(cls, model_config: "ModelConfig") -> bool:
        """
            Check if the model config is supported by this class in v1.
        
        Args:
            model_config (ModelConfig): Model configuration to be checked.
        
        Returns:
            bool: Whether the model config is supported in v1. Always returns True for this class.
        """
        return True

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        '''
       communicator
       '''
        return "vllm_kunlun.distributed.kunlun_communicator.KunlunCommunicator"

    @classmethod
    def get_punica_wrapper(cls):
        return "vllm_kunlun.lora.punica_wrapper.punica_kunlun.PunicaWrapperKunlun"
