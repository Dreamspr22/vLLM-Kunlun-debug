#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Author: Xinyu Dong
# Email: dongxinyu03@baidu.com
# This file is a part of the vllm-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vllm kunlun init"""
from .platforms import current_platform
import sys
import importlib
import warnings
import builtins
import os
import time
import vllm.envs as envs

OLD_IMPORT_HOOK = builtins.__import__


def _custom_import(module_name, globals=None, locals=None, fromlist=(), level=0):
    try:
        start_time = time.time()

        # Module mapping table
        module_mappings = {
            "vllm.model_executor.layers.fused_moe.layer": "vllm_kunlun.ops.fused_moe.layer",
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe": "vllm_kunlun.ops.quantization.compressed_tensors_moe",
            "vllm.compilation.wrapper": "vllm_kunlun.compilation.wrapper",
        }

        # Keep the original imported modules
        original_imports = [
            "vllm.model_executor.layers.fused_moe.base",
            "vllm.model_executor.layers.fused_moe.config",
            "vllm.model_executor.layers.fused_moe.layer",
        ]

        if module_name in original_imports:
            if module_name == "vllm.model_executor.layers.fused_moe.layer" and fromlist:
                if "FusedMoEMethodBase" in fromlist:
                    return OLD_IMPORT_HOOK(
                        module_name,
                        globals=globals,
                        locals=locals,
                        fromlist=fromlist,
                        level=level,
                    )

        if module_name in module_mappings:
            if module_name in sys.modules:
                return sys.modules[module_name]
            target_module = module_mappings[module_name]
            module = importlib.import_module(target_module)
            sys.modules[module_name] = module
            sys.modules[target_module] = module
            return module

        relative_mappings = {
            (
                "compressed_tensors_moe",
                "compressed_tensors",
            ): "vllm_kunlun.ops.quantization.compressed_tensors_moe",
            ("layer", "fused_moe"): "vllm_kunlun.ops.fused_moe.layer",
        }

        if level == 1:
            parent = globals.get("__package__", "").split(".")[-1] if globals else ""
            key = (module_name, parent)
            if key in relative_mappings:
                if module_name in sys.modules:
                    return sys.modules[module_name]
                target_module = relative_mappings[key]
                module = importlib.import_module(target_module)
                sys.modules[module_name] = module
                sys.modules[target_module] = module
                return module

    except Exception:
        pass

    return OLD_IMPORT_HOOK(
        module_name, globals=globals, locals=locals, fromlist=fromlist, level=level
    )


def import_hook():
    """Apply import hook for VLLM Kunlun"""
    if not int(os.environ.get("DISABLE_KUNLUN_HOOK", "0")):
        builtins.__import__ = _custom_import

        try:
            modules_to_preload = [
                "vllm_kunlun.ops.quantization.compressed_tensors_moe",
                "vllm_kunlun.ops.fused_moe.custom_ops",
                "vllm_kunlun.ops.fused_moe.layer",
                "vllm_kunlun.ops.quantization.fp8",
            ]
            for module_name in modules_to_preload:
                importlib.import_module(module_name)
        except Exception:
            pass


def register():
    """Register the Kunlun platform"""
    from .utils import redirect_output
    from .vllm_utils_wrapper import (
        direct_register_custom_op,
        patch_annotations_for_schema,
    )

    import_hook()
    if envs.VLLM_USE_V1:
        patch_V1blockTable()
        patch_V1top_p_K()
        patch_V1penalties()
    else:
        patch_sampler()
    return "vllm_kunlun.platforms.kunlun.KunlunPlatform"


def register_model():
    """Register models for training and inference"""
    from .models import register_model as _reg

    _reg()


def patch_sampler():
    try:
        custom_sampler = importlib.import_module("vllm_kunlun.ops.sample.sampler")
        sys.modules["vllm.model_executor.layers.sampler"] = custom_sampler
        print("[vllm_kunlun] sampler patched ->", custom_sampler.__file__)
    except Exception as e:
        warnings.warn(f"[vllm_kunlun] sampler patch failed: {e!r}")


def patch_V1top_p_K():
    try:
        custom_sampler = importlib.import_module(
            "vllm_kunlun.v1.sample.ops.topk_topp_sampler"
        )
        sys.modules["vllm.v1.sample.ops.topk_topp_sampler"] = custom_sampler
        print("[vllm_kunlun] V1sampler top p & k patched ->", custom_sampler.__file__)
    except Exception as e:
        warnings.warn(f"[vllm_kunlun] V1 sampler top p & k patch failed: {e!r}")


def patch_V1penalties():
    try:
        custom_sampler = importlib.import_module("vllm_kunlun.v1.sample.ops.penalties")
        sys.modules["vllm.v1.sample.ops.penalties"] = custom_sampler
        print("[vllm_kunlun] V1sampler penalties patched ->", custom_sampler.__file__)
    except Exception as e:
        warnings.warn(f"[vllm_kunlun] V1 sampler penalties patch failed: {e!r}")


def patch_V1blockTable():
    try:
        custom_sampler = importlib.import_module("vllm_kunlun.v1.worker.block_table")
        sys.modules["vllm.v1.worker.block_table"] = custom_sampler
        print("[vllm_kunlun] V1 block table patched ->", custom_sampler.__file__)
    except Exception as e:
        warnings.warn(f"[vllm_kunlun] V1 block table patch failed: {e!r}")


# Automatically apply patches when modules are imported
import_hook()
