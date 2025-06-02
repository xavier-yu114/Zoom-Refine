#  <img src="asset/logo.png" alt="Example Image" width="45" style="margin-bottom:-50px"/> Zoom-Refine: Boosting High-Resolution Multimodal Understanding via Localized Zoom and Self-Refinement

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

This repository contains the official implementation for the paper: **Zoom-Refine: Boosting High-Resolution Multimodal Understanding via Localized Zoom and Self-Refinement**.

Anonymous Author(s)
NeurIPS 2025 

## Abstract

Multimodal Large Language Models (MLLMs) often struggle to interpret high-resolution images accurately, where fine-grained details are crucial for complex visual understanding. We introduce Zoom-Refine, a novel training-free method that enhances MLLM capabilities to address this issue. Zoom-Refine operates through a synergistic process of **Localized Zoom** and **Self-Refinement**. In the *Localized Zoom* step, Zoom-Refine leverages the MLLM to provide a preliminary response to an input query and identifies the most task-relevant image region by predicting its bounding box coordinates. During the *Self-Refinement* step, Zoom-Refine then integrates fine-grained details from the high-resolution crop (identified by Localized Zoom) with its initial reasoning to re-evaluate and refine its preliminary response. Our method harnesses the MLLM's inherent capabilities for spatial localization, contextual reasoning, and comparative analysis without requiring additional training or external experts. Comprehensive experiments demonstrate the efficacy of Zoom-Refine on two challenging high-resolution multimodal benchmarks.

## 👀Framework Overview

Zoom-Refine enhances MLLM understanding of high-resolution images in a two-step, training-free process:

1.  **Localized Zoom:** The MLLM first processes a downsampled version of the image and the textual query to provide an initial answer and predict bounding box coordinates for the most task-relevant region.
2.  **Self-Refinement:** A high-resolution crop is extracted from the original image based on the predicted bounding box. This crop, along with the initial context (original query, downsampled image, initial answer), is fed back to the MLLM, which then re-evaluates and refines its answer.


## 📦 Preparation
### 1.MLLM checkpoints
Our experiments primarily use models from the InternVL series, you could download these checkpoints before running.
* [InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B)
* [InternVL3-14B](https://huggingface.co/OpenGVLab/InternVL3-14B)
* [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)
* [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B)
* [InternVL2.5-78B](https://huggingface.co/OpenGVLab/InternVL2_5-78B)

### 2.Evaluation benchmarks
Our experiments utilize the following publicly available benchmarks,you could download these benchmarks before evaluation.
* [MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld)
* [HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench)

After downloading the datasets, their JSON files need to be converted into a standardized format for evaluation with our framework. We provide scripts for this conversion:
For MME-RealWorld use the `scripts/convert.py`
For HR-Bench，you can use the `scripts/convert_parquet.py` to download the benchmark and use the `scripts/build_json.py` to convert to the expected format.

**Standardized MCQA Format:**

The preprocessing scripts will convert the benchmark data into the following standardized Multiple-Choice Question-Answering (MCQA) format, which is then used to prompt the MLLM for evaluation:
```
[Image] [Question] The choices are listed below:
(A) [Choice A]
(B) [Choice B]
(C) [Choice C]
(D) [Choice D]
(E) [Choice E]
Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. 
The best answer is:
```
## 🛠️Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/xavier-yu114/NIPS2025.git
    cd Zoom-Refine
    ```

2.  **Create a conda environment and activate it:**
    ```bash
    conda create -n zoom_refine python=3.9 
    conda activate zoom_refine
    ```

3.  **Install dependencies:**
    Install dependencies using `requirements.txt:`
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file includes the following dependencies:
    - `-r requirements/internvl_chat.txt`
    - `-r requirements/streamlit_demo.txt`
    - `-r requirements/classification.txt`
    - `-r requirements/segmentation.txt`

    we use  `flash-attn==2.3.6`

  ```bash
  pip install flash-attn==2.3.6 --no-build-isolation
  ```

4.  **Model Setup:**    
*   API-based models (e.g., InternVL3-78B, InternVL2.5-78B via InternVL API):
    *   To obtain an API key and for details on client library installation and configuration, please refer to the official InternVL API documentation: [https://internlm.intern-ai.org.cn/api/document](https://internlm.intern-ai.org.cn/api/document).
    *   Once you have your API key, you can set it as an environment variable or include it in a configuration file.
      ```bash
      # export API_KEY="YOUR_API_KEY"
      ```
*   Locally-run models (e.g., InternVL3-14B/8B/2B from Hugging Face):
    For locally run models, download the weights from their official sources (e.g., Hugging Face).Ensure your environment is configured as described in the "Install Dependencies" section above and update model paths in the configuration files or scripts.

## 🚀 Evaluation
This section outlines how to evaluate the Zoom-Refine method on the MME-RealWorld and HR-Bench benchmarks using the provided scripts.

**A. Evaluating Baseline Performance**

**1. Using API-based Models :**
   To evaluate the baseline performance of an API-based model:
   ```bash
   python evaluation/API.py     
   ```

**2. Using Locally-run Models :**
   To evaluate the baseline performance of a locally-run model:
   ```bash
   python evaluation/base.py
   ```
After you get your evaluation results,you can use `evaluation/eval_acc.py` to calculate the accuracy from the baseline prediction files.

**B. Evaluating with Zoom-Refine**

**1. Using API-based Models :**
   To evaluate with Zoom-Refine using an API-based model:
   ```bash
   python zoomrefine/ZoomRefine_API.py 
   ```

**2. Using Locally-run Models:**
   To evaluate with Zoom-Refine using a locally-run model:
   ```bash
   python zoomrefine/ZoomRefine_Base.py
   ```
After you get your evaluation results,you can use `zoomrefine/zoomrefine_acc.py` to calculate the accuracy from the Zoom-Refine prediction files.
