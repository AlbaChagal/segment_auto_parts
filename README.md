# AUTO1 Car Part Segmentation

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python3 src/train.py
```

## Inference
```bash
python3 src/inference.py --input data/test_images --output predictions/
```

## Design
Backbone: pretrained DeepLabV3-ResNet50. Loss: cross entropy. Input 512×512. Classes 0,32,64,96,128,160. Model under 180 MB, avg inference <1 s on RTX 5090.
