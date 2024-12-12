# Memory-Efficient Mental Health Question-Answering System

## Project Overview
This project develops a memory-efficient mental health question-answering system using Large Language Models (LLMs). It is a comparative study baed on Secondary Empirical methodology. It leverages parameter-efficient fine-tuning and memory-efficient quantization (Q-LoRA) techniques for optimized performance. The models are evaluated using ROUGE, BLEU, and BERT scores.

## Models Used
- [**Flan-T5**](https://huggingface.co/google/flan-t5-base)
- [**Tiny-Llama**](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) 
- [**Llama-2** ](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [**Gemma-2**](https://huggingface.co/google/gemma-2-2b-it) 
- [**GPT-Neo**](https://huggingface.co/EleutherAI/gpt-neo-2.7B) 

## Hardware Requirements
- **Development Environment**: 2020 MacBook M1, MacOS Sonoma 14.3.1
- **Training Environment**: Google Colaboratory Pro
  - T4 GPU: 12.7GB RAM, 15.0GB GPU RAM
  - L4 GPU: 53.0GB RAM, 22.5GB GPU RAM
  - A100 GPU: 83.5GB RAM, 40.0GB GPU RAM

## Software Requirements
- **Python Version**: 3.10.12
- **Required Libraries**:
  ```bash
  pip install bitsandbytes==0.400.2 trl==0.4.7 peft==0.4.0 accelerate==0.21.0 transformers==4.31.0 datasets torch evaluate bert-score rouge-score

  ## Steps to Fine-Tune and Evaluate Models

1. **Load and Transform Dataset**:
   - First, load the dataset from the [HuggingFace repository of MentalChat16k](https://huggingface.co/datasets/ShenLab/MentalChat16K).
   - Transform the dataset using the `transform` function to split it into `train`, `test`, and `validation` sets. Ensure the data is formatted properly in `DatasetDict` format to serve the model effectively.

2. **Fine-Tuning the Models**:
   - Open any of the 5 provided `LLM-1epoch.ipynb` files to fine-tune the models using Q-LoRA for 1 epoch. 
   - This script trains the model and evaluates the test set using evaluation metrics such as **ROUGE**, **BERT**, and **BLEU** scores.

3. **Model Selection**:
   - Due to the hallucination tendency of **Tiny Llama** and the poor response generation quality of **Flan T5**, I proceeded with the other three better-performing models:
     - **Llama-2**
     - **Gemma-2**
     - **GPT-Neo**

4. **Fine-Tuning on 10 Epochs**:
   - These three models were further fine-tuned on 10 epochs to check their consistency and performance under resource-constrained conditions (as observed during the 1-epoch fine-tuning).

   This approach ensures the evaluation of model stability and improvements over extended fine-tuning.

