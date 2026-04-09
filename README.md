# Hinglish Sentiment Analysis: PEFT vs. Full Fine-Tuning

This repository contains a comprehensive pipeline for fine-tuning Transformer models on **Hinglish** (Hindi-English code-mixed) text. The project specifically explores and compares two different training strategies: **Full Fine-Tuning** and **Parameter-Efficient Fine-Tuning (PEFT)**.

---

## 🚀 Project Overview

The core Jupyter Notebook provides a step-by-step guide to:

- **Data Preparation:** Processing and tokenizing code-mixed Hinglish review datasets.
- **Full Fine-Tuning:** Updating all model weights using the Hugging Face `Trainer` API.
- **PEFT with LoRA:** Implementing Low-Rank Adaptation (LoRA) to fine-tune only a tiny fraction of the parameters, significantly reducing memory and storage requirements.
- **Comparative Evaluation:** Assessing both methods using accuracy metrics and visual Confusion Matrices.
- **Inference:** Custom functions to test the model on real-world Hinglish phrases like *"Value for money laga"* or *"Experience kaafi acha raha"*.

---

## 📊 Key Comparison: PEFT vs. Full Fine-Tuning

A major feature of this notebook is the side-by-side comparison:

| Aspect | PEFT (LoRA) | Full Fine-Tuning |
|--------|-------------|------------------|
| **Parameters Updated** | < 1% of model parameters | All model weights |
| **Resource Usage** | Low memory & storage | High memory & storage |
| **Speed** | Faster training | Slower training |
| **Performance** | Evaluated against full fine-tuning | Baseline benchmark |
| **Visualization** | Per-class sentiment heatmaps | Per-class sentiment heatmaps |

---

## 🛠️ Installation

To run this project, you will need a GPU environment (e.g., Google Colab T4) and the following dependencies:

```bash
pip install torch transformers datasets peft scikit-learn matplotlib seaborn
```

---

## 📋 Usage

1. **Load the Notebook:** Open `PyTorch_Code_Switched_Sentiment_Analysis.ipynb` in your preferred environment.
2. **Dataset:** The notebook automatically handles the loading of the Hinglish sentiment dataset.
3. **Training:** Follow the cells to run both the Full Fine-Tuning and PEFT/LoRA training loops.
4. **Analysis:** Review the generated metrics and confusion matrices to see the comparison results.

---

## 🏗️ Built With

| Library | Purpose |
|---------|---------|
| [PyTorch](https://pytorch.org/) | Backend deep learning framework |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Pre-trained model access |
| [Hugging Face PEFT](https://huggingface.co/docs/peft) | Parameter-Efficient Fine-Tuning library |
| [Seaborn](https://seaborn.pydata.org/) / [Matplotlib](https://matplotlib.org/) | Results visualization |
