![vLLM Kunlun Logo](vllm_kunlun/patches/vLLM_Kunlun.jpg)

<p align="center">
  <a href="./docs/_build/html/documentation.html"><b>Documentation</b></a> |
  <a href=""><b>Users Forum</b></a> |
  <a href="join.slack.com/t/vllm-kunlun/shared_invite/zt-3iinb8u5z-FcqZKbNNdMJ_32fHmipzvwjoin.slack.com/t/vllm-kunlun/shared_invite/zt-3iinb8u5z-FcqZKbNNdMJ_32fHmipzvw"><b>slack</b></a> |
</p>

---

## Latest News🔥
- [2025/11] 
- [2025/11] 
- [2025/11] 
- [2025/11] Initial release of vLLM Kunlun

---

# Overview

vLLM Kunlun (vllm-kunlun) is a community-maintained hardware plugin designed to seamlessly run vLLM on the Kunlun XPU. It is the recommended approach for integrating the Kunlun backend within the vLLM community, adhering to the principles outlined in the [RFC]: Hardware pluggable. This plugin provides a hardware-pluggable interface that decouples the integration of the Kunlun XPU with vLLM.

By utilizing the vLLM Kunlun plugin, popular open-source models, including Transformer-like, Mixture-of-Expert, Embedding, and Multi-modal LLMs, can run effortlessly on the Kunlun XPU.

---
## Prerequisites

- **Hardware**: Kunlun3 P800 
- **OS**: Ubuntu 22.04 
- **Software**:
  - Python >=3.10
  - PyTorch ≥ 2.5.1
  - vLLM (same version as vllm-kunlun)

---
## Supported Models
<style>
  table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    border-radius: 8px;
    overflow: hidden;
  }
  
  th {
    background: linear-gradient(135deg, #0E7DC6 0%, #0A5BA8 100%);
    color: white;
    padding: 14px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.5px;
    border: none;
  }
  
  td {
    padding: 12px;
    border-bottom: 1px solid #e8e8e8;
    font-size: 13px;
    color: #333;
  }
  
  tr:last-child td {
    border-bottom: none;
  }
  
  tbody tr {
    transition: background-color 0.2s ease;
  }
  
  tbody tr:hover {
    background-color: #f5faff;
  }
  
  tbody tr:nth-child(even) {
    background-color: #fafbfc;
  }
  
  tbody tr:nth-child(even):hover {
    background-color: #f0f7fc;
  }
  
  .status-support {
    color: #22863a;
    font-weight: 600;
    font-size: 14px;
  }
  
  .status-progress {
    color: #f6a909;
    font-weight: 600;
    font-size: 14px;
  }
  
  .status-coming {
    color: #999;
    font-size: 12px;
    background-color: #f5f5f5;
    padding: 2px 6px;
    border-radius: 3px;
    display: inline-block;
  }
  
  .model-name {
    font-weight: 500;
    color: #1e40af;
  }

  h3 {
    color: #1e40af;
    font-size: 16px;
    margin-top: 30px;
    margin-bottom: 15px;
    font-weight: 600;
  }

  h3:first-of-type {
    margin-top: 0;
  }
</style>

<h3>Generaltive Models</h3>
<table>
  <thead>
    <tr>
      <th width="20%">Model</th>
      <th width="12%">Support</th>
      <th width="15%">Quantization</th>
      <th width="10%">LoRA</th>
      <th width="20%">Piecewise Kunlun Graph</th>
      <th width="23%">Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="model-name">Qwen2/2.5</td>
      <td class="status-support">✅</td>
      <td></td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3</td>
      <td class="status-support">✅</td>
      <td></td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3-Moe/Coder</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">QwQ-32B</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">LLama2/3/3.1</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">GLM-4.5/Air</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3next</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">comming soon</span></td>
    </tr>
    <tr>
      <td class="model-name">Gpt oss</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">comming soon</span></td>
    </tr>
    <tr>
      <td class="model-name">Deepseek v3/3.2</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">comming soon</span></td>
    </tr>
  </tbody>
</table>

<h3>Multimodal Language Models</h3>
<table>
  <thead>
    <tr>
      <th width="20%">Model</th>
      <th width="12%">Support</th>
      <th width="15%">Quantization</th>
      <th width="10%">LoRA</th>
      <th width="20%">Piecewise Kunlun Graph</th>
      <th width="23%">Note</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="model-name">Qianfan-VL</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen2.5VL</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">InternVL2.5/3/3.5</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">InternVL3.5</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">InternS1</td>
      <td class="status-support">✅</td>
      <td></td>
      <td></td>
      <td class="status-support">✅</td>
      <td></td>
    </tr>
    <tr>
      <td class="model-name">Qwen2.5 omini</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">comming soon</span></td>
    </tr>
    <tr>
      <td class="model-name">Qwen3vl</td>
      <td class="status-progress">⚠️</td>
      <td></td>
      <td></td>
      <td></td>
      <td><span class="status-coming">comming soon</span></td>
    </tr>
  </tbody>
</table>



## Performance Visualization 🚀
### High-performance computing at work: How different models perform on the Kunlun3 P800.

Current environment: 16-way concurrency, input/output size 2048.


![Models and tgs](./vllm_kunlun/patches/performance.png)

## Getting Started

Please use the following recommended versions to get started quickly:

| Version | Release type | Doc |
|----------|---------------|-----|
| v0.10.1.1 | Latest stable version | [QuickStart](./docs/_build/html/quick_start.html) and [Installation](./docs/_build/html/installation.html) for more details |

---

## Contributing

See [CONTRIBUTING]() for more details, which is a step-by-step guide to help you set up the development environment, build, and test.

We welcome and value any contributions and collaborations:
- Open an [Issue]() if you find a bug or have a feature request

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.