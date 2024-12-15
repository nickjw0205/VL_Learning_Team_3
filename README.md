# VL_Learning_Team_3

Official repository for the **Visual Representation Learning PBL lecture**.

This repository contains the necessary scripts and models for fine-tuning and evaluating a LongCLIP model on a construction site risk assessment dataset. It is based on the [CLIP](https://github.com/openai/CLIP) and [LongCLIP](https://github.com/beichenzbc/Long-CLIP/blob/main/README.md) frameworks.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nickjw0205/VL_Learning_Team_3.git
   cd VL_Learning_Team_3
   ```

2. Set up the environment for **CLIP** and **LongCLIP**:
   - Follow the installation instructions provided in the [CLIP](https://github.com/openai/CLIP) and [LongCLIP](https://github.com/beichenzbc/Long-CLIP/blob/main/README.md) repositories.
   - Install additional dependencies:
     ```bash
     pip install -r requirements.txt
     ```

---

## Dataset Preparation

This project uses the [Aihub 건설 현장 위험 상태 판단 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71407). Follow these steps to set up the dataset:

1. **Download the dataset**:
   - Place the dataset in the `data/` directory.

2. **Format the dataset for fine-tuning LongCLIP**:
   - Use the provided script to preprocess the dataset and convert it into the required format:
     ```bash
     python generate_json.py \
         --base_path "data/images/" \
         --output_json_path "data/image_data.json" \
         --rules_json_path "data/aihub_rules_pair_en.json"
     ```
   - Replace the paths with the correct locations for your dataset.

---

## Training

To fine-tune the LongCLIP model, use the `train.py` script:

```bash
python train.py --exp_name <experiment_name> --soft-prompt True
```

### Notes:
- `--soft-prompt True` must be used during training to use SupCon Loss.

---

## Evaluation

To evaluate the fine-tuned LongCLIP model, use the following command:

```bash
python script.py \
    --model_path "/path/to/Finetuned_longclip_model.pt" \
    --output_path "/path/to/output.json" \
    --rules_path "data/aihub_rules_pair_en.json" \
    --image_root "data/images/"
```

### Output:
- The evaluation results will be printed in the format:
  ```
  Recall@K: <accuracy>
  ```

---

## Usage Example

To train and evaluate the model on your custom dataset:
1. Format the dataset:
   ```bash
   python generate_json.py \
       --base_path "data/custom_images/" \
       --output_json_path "data/custom_image_data.json" \
       --rules_json_path "data/custom_rules.json"
   ```

2. Train the model:
   ```bash
   python train.py --exp_name "custom_experiment" --soft-prompt True
   ```

3. Evaluate the model:
   ```bash
   python script.py \
       --model_path "checkpoints/custom_model.pt" \
       --output_path "results/custom_results.json" \
       --rules_path "data/custom_rules.json" \
       --image_root "data/custom_images/"
   ```

---
