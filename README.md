# RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping

<h5>
  
[![arXiv](https://img.shields.io/badge/RaZeR-2308.13137-b31b1b.svg?logo=arXiv)](https://arxiv.org/html/2501.04052v2)
 <br>
 
</h5>

<p align="center">
  <img src="imgs/razer.png" alt="razer" width="700"/>
</p>

RaZeR extends the standard NVFP4 format by remapping the redundant FP4 zero as an additional, special quantization value. By carefully selecting the set of allowed special values, each NVFP4 block can be quantized with the basic FP4 values and a useful special value, thereby reducing per-block quantization error. Moreover, RaZeR exploits redundancy in the NVFP4 block scale to encode the metadata for storing and indexing special values. Consequantly, RaZeR maintains the same memory footprint as NVFP4 while achieving much higher accuracy.


## News
- [2026/01] 🔥 [RaZeR](https://arxiv.org/abs/2501.04052v2) is available on arXiv.


## Getting Started
1. Clone the GitHub repository and set up conda environment.
```
git clone https://github.com/yc2367/NVFP4-RaZeR/
cd NVFP4-RaZeR
conda env create -f env.yml
conda activate razer
```

2. Go to `scripts/`. For every bash script, change the `hOME_DIR` variable at the beginning.
```
HOME_DIR="Your/Home/Directory"
```

3. Run experiments using the bash script under `scripts/`. For example, to run the perplexity evaluation:
```
bash scripts/test_ppl.sh
```
Currently, we only supoprt Llama and Qwen3 models. If you want to evaluate other models, go to `models/` and add your own `qmodule_<xxx>.py` to evaluate them.
Some python command parameters for evaluation are described below.
- `--model_name`: The model nickname, e.g., "llama_2_7b" used to specify an LLM. The full model path will be mapped according to `model2path.json`.
- `--use_fp16`: If set, then evaluate the original FP16 or BP16 baseline model.
- `--output_dir`: Output path the store the results.
- `--w_bits`: Weight precision.
- `--w_groupsize`: Weight group size.
- `--w_dtype`: Weight data format, e.g., "nvfp4".
- `--w_outlier`: The second special value pair for RaZeR weights. E.g., if set to `8.0`, then the four special values for RaZeR weights will be {$\pm5, \pm8$}. Refer to Appendix B.2 of our paper for more details.
- `--a_bits`: Activation precision.
- `--a_groupsize`: Activation group size.
- `--a_dtype`: Activation data format, e.g., "nvfp4".
- `--kv_quant`: Whether to quantize KV-cache.
- `--k_bits`: Key precision.
- `--k_groupsize`: Key group size.
- `--k_dtype`: Key data format, e.g., "nvfp4".
- `--v_bits`: Value precision.
- `--v_groupsize`: Value group size.
- `--v_dtype`: Value data format, e.g., "nvfp4".


## Supported Data Formats
|  **Data Format**              | Definition                                     |
| --------------------------- | ---------------------------------------------- |
|  **fp16**                 | The baseline FP16 without quantization   |
|  **mxfp4**                | The 4-bit [Microscaling FP4](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) |
|  **nf4**                  | The 4-bit NormalFloat format in [QLoRA](https://arxiv.org/abs/2305.14314) |
|  **nvfp4**                | NVIDIA [NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) |
|  **nvfp4_4over6**         | The [4over6](https://arxiv.org/abs/2512.02010) format on top of NVFP4 |
|  **nvfp4_razer_e3m3**     | The RaZeR format with E3M3 block scale and 4 special values for weight quantization |
|  **nvfp4_razer_e4m3**     | The RaZeR format with E4M3 block scale and 2 special values for activation quantization |



## Perplexity Results
RaZeR significantly outperforms existing 4-bit LLM quantization methods.
![ppl](imgs/ppl_results.png)


## Citation
```bibtex
@article{chen2026razer,
  title={{RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping}},
  author={Yuzong Chen and Xilai Dai and Jake Hyun and Chi-Chih Chang and Wonsuk Jang and Yuheng Wu and Thierry Tambe and Jae-sun Seo and Mohamed S. Abdelfattah},
  journal={arXiv preprint arXiv:2501.04052v2},
  year={2026}
}
```

-----------------

_This work is subject to a patent application filed by Cornell University._
