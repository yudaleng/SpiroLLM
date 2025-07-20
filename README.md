<h1 align="center">SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting</h1>

<p align="center">
<a href="https://github.com/yudaleng/SpiroLLM">
<img src="https://img.shields.io/badge/GitHub-Code-blue?style=for-the-badge&logo=github" alt="GitHub Repository">
</a>
<a href="https://huggingface.co/yudaleng/SpiroLLM">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow?style=for-the-badge" alt="Hugging Face Model">
</a>
<a href="#">

[//]: # (<img src="https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper">)
</a>
<img src="https://img.shields.io/badge/CC%20BY-NC-SA?style=for-the-badge" alt="License: CC BY-NC-SA 4.0">
</p>

<p align="center">
<strong><a href="#">Shuhao Mei</a> [1, 2, 7], <a href="#">Yongchao Long</a> [2], <a href="#">Shan Cao</a> [3], <a href="#">Xiaobo Han</a> [4], <a href="#">Shijia Geng</a> [5],</strong><br>
<strong><a href="#">Jinbo Sun</a> [1]<em>*, <a href="#">Yuxi Zhou</a> [2, 6]<em>*, <a href="https://hsd1503.github.io/">Shenda Hong</a> [7]<em>* </strong>
<br>
<br>
<small>
[1] <a href="https://www.xidian.edu.cn/">Xidian University</a>;
[2] <a href="https://www.tjut.edu.cn/">Tianjin University of Technology</a>;
[3] <a href="https://www.tjmush.com.cn/index.shtml">The Second Hospital of Tianjin Medical University</a>;
[4] <a href="https://www.301hospital.com.cn/index.html">Chinese PLA General Hospital</a>;
</small><br>
<small>
[5] <a href="https://www.heartvoice.com.cn/about.html">HeartVoice Medical Technology</a>;
[6] <a href="https://www.tsinghua.edu.cn/">Tsinghua University</a>;
[7] <a href="https://www.pku.edu.cn/">Peking University</a>
</small><br>
<small><em>* Corresponding Author</em></small>
</p>


---

## Introduction
SpiroLLM is the **first** multimodal large language model specifically designed to interpret spirogram time-series data, providing diagnostic support for Chronic Obstructive Pulmonary Disease (COPD). By integrating raw spirometry signals with demographic information, SpiroLLM generates comprehensive and clinically relevant diagnostic reports.

---

## Quickstart

### 1. Setup Environment
First, create and activate a Conda virtual environment, then install the required dependencies.

```bash
# Create and activate the environment
conda create -n SpiroLLM python=3.11 -y
conda activate SpiroLLM

# Install all dependencies
pip install -r requirements.txt
````

### 2\. Prepare Demo Data

Run the provided script to automatically download the example spirometry data from the UK Biobank website. The data will be saved to the `data/` directory.

```bash
python generate_ukbb_demo_data.py
```

### 3\. Run Inference

Once the environment is set up and the data is downloaded, run the main inference script with the patient's information.

```bash
python main.py \
    --csv_path ./data/example.csv \
    --age 69 \
    --sex Male \
    --height_cm 176.0 \
    --is_smoker
```

The generated report will be printed to the console and saved to the output file specified in your `config.yaml`.

-----

## System Requirements

  - **Python**: 3.11
  - **PyTorch**: \>= 2.0
  - **GPU**: A CUDA-enabled GPU with at least **16 GB of VRAM** is required for the model to run properly.

-----

## Usage

The `main.py` script is the primary entry point for running inference. It requires the following command-line arguments:

| Argument | Type | Description | Required |
| :--- | :--- | :--- | :---: |
| `--csv_path` | `str` | Path to the patient's raw spirometry data file. | **Yes** |
| `--age` | `int` | The age of the patient in years. | **Yes** |
| `--sex` | `str` | The sex of the patient (`Male` or `Female`). | **Yes** |
| `--height_cm` | `float` | The height of the patient in centimeters. | **Yes** |
| `--is_smoker` | `flag` | Include this flag if the patient is a smoker. | No |
| `--ethnicity` | `str` | Patient's ethnicity. Defaults to `Caucasian`. | No |
| `--config` | `str` | Path to the configuration YAML file. | No |

-----

## Data Source

The data used in this project is sourced from the **UK Biobank**, a large-scale biomedical database and research resource. Access to the data is available to approved researchers upon application. For more information, please visit the [UK Biobank website](https://www.ukbiobank.ac.uk).

-----

## Relation to Prior Work
The `DeepSpiro` feature extractor, a key component of this project, is based on our prior work published in npj systems biology and applications:
```
Mei S, Li X, Zhou Y, et al. Deep learning for detecting and early predicting chronic obstructive pulmonary disease from spirogram time series[J]. npj Systems Biology and Applications, 2025, 11(1): 18.
```
The original implementation is available at the <a href="https://github.com/yudaleng/COPD-Early-Prediction">COPD-Early-Prediction GitHub repository</a>.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or see the LICENSE file.