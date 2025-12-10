# Quantization Guide
>Note: This feature is currently experimental. In future versions, there may be behavioral changes around configuration, coverage, performance improvement.

Like vLLM, we now support quantization methods such as compressed-tensors, AWQ, and GPTQ, enabling various precision configurations including W8A8, W4A16, and W8A16. These can help reduce memory consumption and accelerate inference while preserving model accuracy.


## Usages

### Compressed-tensor
To run a `compressed-tensors` model with vLLM-kunlun, you should first add the below configuration to the model's `config.json`:

```Bash
"quantization_config": {
    "quant_method": "compressed-tensors"
  }
```

Then you run `Qwen/Qwen3-30B-A3B` with dynamic W8A8 quantization with the following command:

```Bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B \
    --quantization compressed-tensors
```

### AWQ

To run an `AWQ` model with vLLM-kunlun, you can use `Qwen/Qwen3-32B-AWQ` with the following command:

```Bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-AWQ \
    --quantization awq
```

### GPTQ

To run a `GPTQ` model with vLLM-kunlun, you can use `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` with the following command:

```Bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --quantization gptq
```

