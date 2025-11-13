import argparse
import os
import torch
from time import perf_counter
from typing import Tuple

import numpy as np
import torchvision.transforms as T
from PIL import Image

from config import Config
from logger import Logger
from model import SegModel
from preprocessors import ImagePreprocessor


class InferenceManager(object):
    """
    A class to manage the inference process for image segmentation.
    """
    def __init__(self, config: Config,
                 input_dir: str,
                 output_dir: str,
                 model_path: str,
                 debug_level: str = 'info'):
        """
        Initialize the InferenceManager.
        :param config: The configuration object.
        :param input_dir: The directory containing input images.
        :param output_dir: The directory to save output segmentation masks.
        :param model_path: The path to the trained model weights.
        :param debug_level: The logging level.
        """
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.model_path: str = model_path

        self.config: Config = config
        self.logger: Logger = Logger(name=self.__class__.__name__,
                                     logging_level=debug_level)

        os.makedirs(self.output_dir, exist_ok=True)

        # Modules
        self.preprocessor: ImagePreprocessor = ImagePreprocessor(config=self.config)
        self.model: SegModel = SegModel(config=self.config)
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval().to(self.model.device)

    def run_inference(self):
        """
        Run inference on all images in the input directory and save the segmentation masks.
        :return: None
        """
        with torch.no_grad():
            for fname in os.listdir(self.input_dir):
                t_load_data_start: float = perf_counter()
                img: Image = Image.open(os.path.join(self.input_dir, fname)).convert("RGB")
                orig_shape: Tuple[int, int] = img.size

                t_preprocess_start: float = perf_counter()
                x: torch.Tensor = self.preprocessor(img).unsqueeze(0).to(self.model.device)
                t_infer_start: float = perf_counter()
                pred: torch.Tensor = self.model(x).argmax(1).squeeze(0) * 32.
                t_post_process_start: float = perf_counter()
                postprocess: T.Resize = T.Resize(orig_shape[::-1],
                                                 interpolation=T.InterpolationMode.NEAREST)

                pred_full_size: torch.Tensor = postprocess(pred.unsqueeze(0).cpu())
                pred_numpy: np.ndarray = pred_full_size.numpy().astype(np.uint8)
                final_pred: Image.Image = Image.fromarray(pred_numpy.squeeze())
                assert final_pred.size == orig_shape, \
                    f'Expected final prediction size {orig_shape}, got {final_pred.size}'
                t_main_end: float = perf_counter()
                Image.fromarray(pred_numpy.squeeze()).save(os.path.join(self.output_dir, fname))

                self.logger.info(f'Inference timings for {fname}: '
                                 f'load_data: {t_preprocess_start - t_load_data_start:.4f}, '
                                 f'preprocess: {t_infer_start - t_preprocess_start:.4f}, '
                                 f'inference: {t_post_process_start - t_infer_start:.4f}s, '
                                 f'postprocess: {t_main_end - t_post_process_start:.4f}s, '
                                 f'total: {t_main_end - t_load_data_start:.4f}s')


if __name__ == "__main__":
    """
    Run inference on a set of images using a trained segmentation model
    and save the output segmentation masks.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    model_path_main = '/Users/shaharheyman/PycharmProjects/auto1_segmentation/outputs/20251111_190327/'
    config = Config()
    inference_manager = InferenceManager(config=config,
                                         input_dir=args.input,
                                         output_dir=args.output,
                                         model_path=model_path_main)
    inference_manager.run_inference()
