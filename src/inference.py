import argparse, torch, os
from PIL import Image
import torchvision.transforms as T
from model import SegModel
from config import Config

def run_inference(input_dir, output_dir):
    cfg = Config()
    os.makedirs(output_dir, exist_ok=True)
    model = SegModel(config=cfg)
    model.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
    model.eval().to(model.device)
    transform = T.Compose([
        T.Resize(cfg.image_size), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        for fname in os.listdir(input_dir):
            img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
            x = transform(img).unsqueeze(0).to(model.device)
            pred = model(x).argmax(1).squeeze(0).cpu().byte()
            Image.fromarray(pred.numpy()).save(os.path.join(output_dir, fname.replace(".jpg",".png")))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run_inference(args.input, args.output)
