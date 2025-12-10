# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_MULTI_LOGPATH: str = ("./log",)
    ENABLE_VLLM_MULTI_LOG: bool = (False,)
    ENABLE_VLLM_INFER_HOOK: bool = (False,)
    ENABLE_VLLM_OPS_HOOK: bool = (False,)
    ENABLE_VLLM_MODULE_HOOK: bool = False


def maybe_convert_int(value: Optional[str]) -> Optional[int]:
    """
    If the value is None, return None; otherwise, convert the string to an integer and return it.

    Args:
        value (Optional[str], optional): The optional string to convert. Defaults to None.

    Returns:
        Optional[int]: If the value is None, return None; otherwise, convert the string to an integer and return it.
    """
    if value is None:
        return None
    return int(value)


# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

xvllm_environment_variables: dict[str, Callable[[], Any]] = {
    # path to the logs of redirect-output, abstrac of related are ok
    "VLLM_MULTI_LOGPATH": lambda: os.environ.get("VLLM_MULTI_LOGPATH", "./logs"),
    # turn on / off multi-log of multi nodes & multi cards
    "ENABLE_VLLM_MULTI_LOG": lambda: (
        os.environ.get("ENABLE_VLLM_MULTI_LOG", "False").lower() in ("true", "1")
    ),
    # turn on / off XVLLM infer stage log ability
    "ENABLE_VLLM_INFER_HOOK": lambda: (
        os.environ.get("ENABLE_VLLM_INFER_HOOK", "False").lower() in ("true", "1")
    ),
    # turn on / off XVLLM infer_ops log ability
    "ENABLE_VLLM_OPS_HOOK": lambda: (
        os.environ.get("ENABLE_VLLM_OPS_HOOK", "False").lower() in ("true", "1")
    ),
    "ENABLE_VLLM_MODULE_HOOK": lambda: (
        os.environ.get("ENABLE_VLLM_MODULE_HOOK", "False").lower() in ("true", "1")
    ),
    # fuse sorted op with fused_moe kernel
    "ENABLE_VLLM_MOE_FC_SORTED": lambda: (
        os.environ.get("ENABLE_VLLM_MOE_FC_SORTED", "False").lower() in ("true", "1")
    ),
    # enable custom dpsk scaling rope
    "ENABLE_CUSTOM_DPSK_SCALING_ROPE": lambda: (
        os.environ.get("ENABLE_CUSTOM_DPSK_SCALING_ROPE", "False").lower()
        in ("true", "1")
    ),
    # fuse qkv split & qk norm & qk rope
    # only works for qwen3 dense and qwen3 moe models
    "ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE": lambda: (
        os.environ.get("ENABLE_VLLM_FUSED_QKV_SPLIT_NORM_ROPE", "False").lower()
        in ("true", "1")
    ),
}

# end-env-vars-definition


def __getattr__(name: str):
    """
    This function is called when an attribute that doesn't exist is accessed.
    If the attribute is one of the xvllm_environment_variables, return the corresponding value.
    Otherwise, raise an AttributeError.

    Args:
        name (str): The name of the attribute to retrieve.

    Raises:
        AttributeError (Exception): If the attribute is not one of xvllm_environment_variables, this exception is raised.

    Returns:
        Any, optional: If the attribute is one of xvllm_environment_variables, the corresponding value is returned; otherwise, None is returned.
    """
    # lazy evaluation of environment variables
    if name in xvllm_environment_variables:
        return xvllm_environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """
    Returns a list of all visible variable names.

    Returns:
        list: A list of all visible variable names, which are defined through the `xvllm_environment_variables` dictionary.

    Returns:
        List[str]: A list of all visible variable names.
                   These variables are defined through the `xvllm_environment_variables` dictionary.
    """
    return list(xvllm_environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in xvllm_environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
