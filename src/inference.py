import argparse, torch, os
from time import perf_counter

import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import SegModel
from config import Config
from logger import Logger
from preprocessors import ImagePreprocessor


class Inference_manager(object):
    def __init__(self, config: Config,
                 input_dir: str,
                 output_dir: str,
                 model_path: str,
                 debug_level: str = 'info'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_path = model_path

        self.config = config
        self.logger = Logger(name=self.__class__.__name__,
                             logging_level=debug_level)

        os.makedirs(self.output_dir, exist_ok=True)

        # Modules
        self.preprocessor = ImagePreprocessor(config=self.config)
        self.model = SegModel(config=self.config)
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval().to(self.model.device)

    def run_inference(self):
        with torch.no_grad():
            for fname in os.listdir(self.input_dir):
                t_load_data_start = perf_counter()
                img = Image.open(os.path.join(self.input_dir, fname)).convert("RGB")
                orig_shape = img.size

                t_preprocess_start = perf_counter()
                x = self.preprocessor(img).unsqueeze(0).to(self.model.device)
                t_infer_start = perf_counter()
                pred = self.model(x).argmax(1).squeeze(0) * 32.
                t_post_process_start = perf_counter()
                postprocess = T.Resize(orig_shape,
                                       interpolation=T.InterpolationMode.NEAREST)
                pred_full_size = postprocess(pred.unsqueeze(0).cpu())
                pred_numpy = pred_full_size.numpy().astype(np.uint8)
                t_main_end = perf_counter()
                Image.fromarray(pred_numpy.squeeze()).save(os.path.join(self.output_dir, fname))

                self.logger.info(f'Inference timings for {fname}: '
                                 f'load_data: {t_preprocess_start - t_load_data_start:.4f}, '
                                 f'preprocess: {t_infer_start - t_preprocess_start:.4f}, '
                                 f'inference: {t_post_process_start - t_infer_start:.4f}s, '
                                 f'postprocess: {t_main_end - t_post_process_start:.4f}s, '
                                 f'total: {t_main_end - t_load_data_start:.4f}s')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model_path_main = 'outputs/20251111_150841/weights/11.pth'
    config = Config()
    inference_manager = Inference_manager(config=config,
                                          input_dir=args.input,
                                          output_dir=args.output,
                                          model_path=model_path_main)
    inference_manager.run_inference()
