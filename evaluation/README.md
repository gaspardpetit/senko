# Senko Evaluation via OpenBench
The following benchmarks on standard datasets were performed using [OpenBench](https://github.com/argmaxinc/OpenBench) by [Argmax](https://www.argmaxinc.com). Using OpenBench allows for standardized, reproducible evaluation and comparison with other diarization systems.

Default settings were used (which in OpenBench are `collar=0.0` and `skip_overlap=False`), meaning the results below for Senko can be compared with those of other diarization systems and services benchmarked by Argmax and compiled in [BENCHMARKS.md](https://github.com/argmaxinc/OpenBench/blob/main/BENCHMARKS.md) in the OpenBench repo.

Some important notes:
- Definitions of `DER`, `SF`, `SCA` can be found in the OpenBench `BENCHMARKS.md`
- For a fair comparison of `Speed Factor (SF)` results, be sure to look at the hardware used to conduct the benchmarks in OpenBench's `BENCHMARKS.md` (listed under "Benchmarked Systems"), and compare that with the hardware used to evaluate Senko below.

See the instructions after the benchmarks below for reproducing these results.

### RTX 5090 + Ryzen 9 9950X (GPU clustering)

| Dataset | Diarization Error Rate (DER) | Speed Factor (SF) | Speaker Count Accuracy (SCA) |
|:-------:|:---:|:------------:|:---:|
| AISHELL-4 | 0.133 | 693.4x | 95.0% |
| AMI-IHM | 0.266 | 724.7x | 75.0% |
| AMI-SDM | 0.321 | 734.6x | 50.0% |
| AVA-AVD | 0.722 | 961.7x | 13.0% |
| AliMeeting | 0.314 | 656.9x | 70.0% |
| Earnings-21 | 0.209 | 777.5x | 43.2% |
| ICSI | 0.378 | 705.0x | 44.0% |
| VoxConverse | 0.177 | 669.0x | 40.9% |

### RTX 5090 + Ryzen 9 9950X (CPU clustering)

| Dataset | Diarization Error Rate (DER) | Speed Factor (SF) | Speaker Count Accuracy (SCA) |
|:-------:|:---:|:------------:|:---:|
| AISHELL-4 | 0.133 | 317.3x | 95.0% |
| AMI-IHM | 0.265 | 346.3x | 87.5% |
| AMI-SDM | 0.297 | 355.7x | 56.2% |
| AVA-AVD | 0.713 | 800.6x | 14.8% |
| AliMeeting | 0.305 | 283.1x | 80.0% |
| Earnings-21 | 0.209 | 518.1x | 47.7% |
| ICSI | 0.375 | 505.1x | 44.0% |
| VoxConverse | 0.135 | 465.6x | 41.8% |

### Apple M3

| Dataset | Diarization Error Rate (DER) | Speed Factor (SF) | Speaker Count Accuracy (SCA) |
|:-------:|:---:|:------------:|:---:|
| AISHELL-4 | 0.136 | 285.3x | 90.0% |
| AMI-IHM | 0.275 | 269.7x | 50.0% |
| AMI-SDM | 0.328 | 274.6x | 68.8% |
| AVA-AVD | 0.722 | 834.3x | 18.5% |
| AliMeeting | 0.304 | 281.6x | 80.0% |
| Earnings-21 | 0.211 | 433.1x | 45.5% |
| ICSI | 0.375 | 385.9x | 50.7% |
| VoxConverse | 0.139 | 392.7x | 44.8% |

## Evaluation Setup
Create a Python virtual environment and activate it
```
uv venv --python 3.11.13 .venv
source .venv/bin/activate
```
Install dependencies
```sh
# For NVIDIA GPUs with CUDA compute capability >= 7.5 (~GTX 16 series and newer)
uv pip install -r requirements.txt "git+https://github.com/narcotic-sh/senko.git[nvidia]"

# For NVIDIA GPUs with CUDA compute capability < 7.5 (~GTX 10 series and older)
uv pip install -r requirements.txt "git+https://github.com/narcotic-sh/senko.git[nvidia-old]"

# For Mac (macOS 14+) and CPU execution on all other platforms
uv pip install -r requirements.txt "git+https://github.com/narcotic-sh/senko.git"
```
Run the evaluation script
```sh
# Evaluate on specific dataset
python evaluate.py --dataset voxconverse

# Evaluate on all datasets
python evaluate.py --all-datasets

# [--device, --vad, --clustering, --model-dir, --num-samples] options also available; see evaluate.py
```
Note: Datasets will be automatically downloaded from Hugging Face. If you run into rate limiting, it helps to add your Hugging Face token as an environment variable `export HF_TOKEN="..."`.
