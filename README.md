# AUTO1 Car Part Segmentation – Technical Assignment

This repository contains my end‑to‑end solution for the AUTO1 
*Senior Computer Vision Engineer – Technical Assignment: Car Part Segmentation*.  
The project includes a full training pipeline, inference CLI, DeepLabV3‑based model, experiment tracking with TensorBoard and a script for computing class weights based on training & validation mask statistics.

---

## 1. Environment Setup

### **Requirements**
- Python **3.11+**
- PyTorch **2.x**
- Minimal additional dependencies: see 'requirements.txt' for full requirements list.
The requirements.txt is in strict form ('==' instead of '>=') to avoid mismatched between package versions.

### **Install**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate   # macOS / Linux
# .\.venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

```

---

## 2. Training the Model

### **Command**
```bash
python3 src/train.py
```

All training‑related hyperparameters (learning rate, batch size, class_weights, etc.) can be configured in `config.py`.
Data augmentation parameters can be configured in `data_structures.py` under `AugmentationProbabilities`

### **Experiment Notes**
Due to time constraints, I did not perform a full hyperparameter search.  
Instead, I evaluated:
- **3 learning rates:** 1e-4, 2e-4 & ***5e-5***
- **2 augmentation pipelines:** all avialable augmentations & ***only flipping***
- **2 batch sizes:** 4 & ***8***
- **2 augmentation sets:** All implemented augmentation & ***Only horizontal and vertical flips***

The selected configurations (in bold) are the best observed under these constraints.

---

## 3. Inference on Test Images

### **Command**
```bash
python3 src/inference.py --input path/to/test_images --output path/to/predictions
```

The script:
- Loads the selected checkpoint from `checkpoint.pth`
- Performs preprocessing + normalization
- Runs inference under `torch.no_grad()`
- Saves masks as **single‑channel PNGs** with the exact class values `{0, 32, 64, 96, 128, 160}`
- In debug mode it also:
  1. Creates GIFs alternating between test_image and prediction map
  2. Creates a legend for simply understanding what color represents which class

---

## 4. Model Choice and Justification

I chose **DeepLabV3 (ResNet‑50 backbone)** as implemented in `torchvision.models.segmentation`.  
Paper: https://arxiv.org/pdf/1706.05587v3

To match it to the task at hand, I replaced the classifying fully connected layer 
at the end of the model and replaced it with a fully connected layer that outputs a vector with `config.num_classes` channels.

### **Reasons**
1. **Strong baseline**: DeepLabV3 is a well‑established architecture for semantic segmentation with competitive performance across many datasets.
2. **Production‑friendly**: It is efficient, well‑tested, easy to optimize, and meets the assignment constraints for:
   - Model size (<180 MB)
   - Inference speed (<1s per image on RTX 5090 equivalent)
3. **Time constraints**: Designing a custom architecture or testing many alternatives was outside the allowed time. DeepLabV3 offers an optimal balance of **accuracy vs. engineering time**.
4. **Good performance on small datasets**: The atrous spatial pyramid pooling (ASPP) captures multi‑scale context, which helps with car‑part segmentation.

Also chosen were the following **training configurations**:
- **Input size**: 512×512 pixels, balancing detail, memory, training time and inference latency constraints
- **Loss function**: Weighted Cross‑Entropy Loss, to address class imbalance (especially for rare classes like door handles).
- **Pretrained weights**: Using DeepLabV3‑pretrained backbone speeds up convergence and improves generalization.
- **Optimizer**: AdamW, which generally performs well for segmentation tasks.

### **Relevant SOTA References for DeepLabV3**

| Dataset | mIoU (DeepLabV3, ResNet‑50) | Notes |
|--------|-------------------------------|-------|
| **PASCAL VOC 2012** | ~0.87 | Standard benchmark for mid‑level segmentation |
| **Cityscapes** | ~0.81 | Urban driving scenes; similar object structure complexity |

These are used as the **only available baselines**, since no public baseline exists for AUTO1's private dataset.

---

## 5. Metrics, Results, Latency & Business Considerations

My **validation metrics** at step 4,600 (best validation IoU) are:

- **IoU** ≈ 0.735  
- **Dice** ≈ 0.842  

- **Average Inference Time (on test images)** = ~0.25 seconds (not including first warmup pass)  
- **Model Size** = ~160 MB (saved as `checkpoint.pth`)
The inference time was recorder on an Apple M4 (16Gb) chip and should be faster with an NVIDIA RTX 5090 

The selected checkpoint was chosen according to **mean IoU**.  
However, depending on the *business objective*, another metric may become more important

### Examples of metric vs. business priority:
- **Mean Dice** if overall overlap quality is prioritized.
- **Recall‑oriented** if missing a damaged car part has a high cost.
- **Precision‑oriented** if false positives trigger expensive manual review.
- **Per‑class IoU** if some classes (e.g., door handles) are more critical than others.
- **Hard metrics** such as % of objects detected above a certain pixel threshold, if there is such a 
threshold that sets the business needs. 

Depending on the business requirements, the ranking of checkpoints may differ.

---

## 6. TensorBoard Logging

Training and validation metrics are logged to TensorBoard:

### **Logged Global Metrics**
- IoU  
- Dice  
- Precision  
- Recall  
- Loss

### **Per‑class Metrics**
Logged for each of the 6 classes:
- IoU  
- Dice  
- Recall  
- Precision

### **Purpose**

1. Debugging segmentation failures (e.g., small objects collapsing).  
2. Ensuring the final checkpoint performs well for *all* classes.  
3. Detecting class‑specific collapse early, especially for rare classes.

To launch TensorBoard:
```bash
tensorboard --logdir outputs/
```
While it is running:
Navigate to `http://localhost:6006` in your browser.

---

## 7. Class Weight Computation Script

This repository includes a helper script:

`count_label_distribution.py`

It scans all training masks and produces:
- Pixel count per class  
- Class distribution histogram  
- Automatically computed **inverse‑frequency and mean-frequency class weights**

These weights were used for:
- **Weighted Cross‑Entropy Loss**
- Helping the model avoid collapsing on rare classes (e.g., door handles)

### **Usage**
```bash
python3 count_label_distribution.py --masks path/to/masks
```

Output includes:
- Raw counts  
- Normalized frequencies  
- Suggested PyTorch weight tensor  

Example:
```python
class_weights = [0.130084, 0.722525, 0.835331, 3.642700, 1.245533, 21.490470]
```

You can manually paste the resulting weights into `config.py`.

---

## 8. Project Structure

```
auto1_car_part_segmentation/
├── .venv
├── data/
    ├── test/
        ├── images/
            ├── 000.jpg
            ├── 001.jpg
            ├── ...
    ├── train/
       ├── images/ 
           ├── 000.jpg
           ├── 001.jpg
           ├── ...
       ├── masks/
           ├── 000.png
           ├── 001.png
           ├── ...
├── predictions/
    ├── 000.png
    ├── 001.png
    ├── ...
├── src/
    ├── augmenter.py
    ├── config.py
    ├── count_label_distribution.py
    ├── data_structures.py
    ├── dataset.py
    ├── inference.py
    ├── logger.py
    ├── metrics.py
    ├── model.py
    ├── metrics.py
    ├── preprocessors.py
    ├── tensorboard_logger.py
    ├── train.py
├── checkpoint.pth
├── requirements.txt
├── README.md



```

---

## 9. Reproducibility

- Deterministic seeds - Complete determinism was not tested as it was not required for the assignment, 
but seeds are set for all relevant libraries.
- Fixed versions in `requirements.txt`  
- All training parameters documented in `config.py`  
- Single‑command inference and training  

---

## 10. Hardware

The project was created and tested on a MacBook Pro with an Apple M4 chip with 16Gb unified memory
Please NOTICE that I did not have an Nvidia GPU to test this on, therefore it has not been tested on GPU, 
but should work seamlessly in theory. The project has been tested on CPU and it works, 
but of course this slows the entire process down by a large margin.

---

## 11. Unit Tests
Each module (except for `Config`) includes basic unit tests that can be run with:
```bash
python3 src/<module_name>.py
```
The tests themselves are located at the bottom of each module file, under the
`if __name__ == "__main__":` clause.

## 12. Future Work
If more time were available, I would explore several improvements to enhance model performance, robustness and production readiness:

### **Model & Training Improvements**
- Run a more systematic hyperparameter search (learning-rate schedules, optimizers, batch sizes, regularization).
- Test alternative architectures such as DeepLabV3+, HRNet, or SegFormer.
- Improve handling of small objects using higher-resolution training and losses like Focal or Lovász-Softmax.
- Add hard metrics (for example how many door handles did we segment at least a th% of their pixels) 
to get a better picture of performance, as well as align to business metrics depending on the details of the business task

### **Data & Augmentation Enhancements**
- Use stronger augmentations (s.a. color jitter, noise, compression, perspective transforms) - 
already implemented but seems to degrade results, it needs some more testing and param tuning.
- Improve sampling by oversampling rare classes or mining patches rich in small objects like door handles.

### **Evaluation & Error Analysis**
- Add uncertainty estimation, per-class calibration, and deeper failure analysis.

### **Production-Level Improvements**
- Export the model to ONNX and optimize with TensorRT for faster inference (if low latency is a business requirment)
- Add structured model versioning, monitoring and drift detection.
- Build an automated training/retraining pipeline for continuous improvements.

## 12. Final Notes

This repository satisfies all requirements from the assignment, including:
- Model file under 180MB
- Inference under 1s per image (test on Apple M4)
- CLI inference tool
- Full documentation
- Justified design choices

