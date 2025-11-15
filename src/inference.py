import argparse
import os
import torch
from time import perf_counter
from typing import Tuple, List

import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from config import Config
from logger import Logger
from model import SegModel
from preprocessors import ImagePreprocessor


class InferenceManager(object):
    """
    A class to manage the inference process for image segmentation.
    """
    def __init__(self,
                 config: Config,
                 input_dir: str,
                 output_dir: str,
                 model_path: str,
                 logging_level: str = 'info'):
        """
        Initialize the InferenceManager.
        :param config: The configuration object.
        :param input_dir: The directory containing input images.
        :param output_dir: The directory to save output segmentation masks.
        :param model_path: The path to the trained model weights.
        :param logging_level: The logging level.
        """
        self.input_dir: str = input_dir
        self.output_dir: str = output_dir
        self.model_path: str = model_path

        self.config: Config = config
        self.logger: Logger = Logger(name=self.__class__.__name__,
                                     logging_level=logging_level)
        self.is_create_gifs: bool = logging_level == 'debug'

        os.makedirs(self.output_dir, exist_ok=True)

        # Modules
        self.preprocessor: ImagePreprocessor = ImagePreprocessor(config=self.config)
        self.model: SegModel = SegModel(config=self.config)
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval().to(self.model.device)

    @staticmethod
    def _change_name_to_png(filename: str) -> str:
        """
        Change the file extension of the given filename to .png
        :param filename: The original filename.
        :return: The filename with .png extension.
        """
        base, _ = os.path.splitext(filename)
        return f'{base}.png'

    def _create_alternating_gif(self,
                                img: Image.Image,
                                mask: Image.Image,
                                save_path: str,
                                duration: int = 1000):
        """
        Create an alternating GIF between the original image and the segmentation mask.
        :param img: The original image.
        :param mask: The segmentation mask.
        :param save_path: The path to save the GIF.
        :param duration: Duration of each frame in milliseconds.
        :return: None
        """
        frames: List[Image.Image] = [img, mask.convert("RGB")]
        frames[1].putalpha(128)  # Make mask semi-transparent
        frames = [frames[0], frames[1], frames[0]]  # Alternate frames
        frames[0].save(save_path,
                       save_all=True,
                       append_images=frames[1:]*20,  # Repeat to make longer GIF
                       duration=duration)
        self.logger.info(f'Saved alternating GIF to {save_path}')

    def _plot_legend(self) -> None:
        """
        Create a legend to understand visualization better
        :return: None
        """
        plt.figure()
        legend: np.ndarray = np.zeros(shape=[512, 512])
        height_factor: int = legend.shape[0] // len(self.config.class_names_to_labels)
        w_center: int = legend.shape[0] // 2
        text_and_center: List[Tuple[str, int]] = []
        h_min: int
        h_max: int
        h_center: int
        for i, (name, value) in enumerate(self.config.class_names_to_labels.items()):
            h_min = i * height_factor
            h_max = (i + 1) * height_factor
            h_center = h_min + (height_factor // 2)
            legend[h_min: h_max, :] = value
            text_and_center.append((name, h_center))

        plt.imshow(legend, vmin=0, vmax=255, cmap='gray')
        color: Tuple[str, int] = ('C3', 1)  # Red - visible on all grayscale backgrounds
        for text, center in text_and_center:
            plt.text(x=w_center, y=center, s=text, color=color, horizontalalignment='center')
        path: str = f'{self.output_dir}/../legend.jpg'
        plt.savefig(path)
        self.logger.debug(f'_plot_legend - saved legend plot to {path}')

    def run_inference(self):
        """
        Run inference on all images in the input directory and save the segmentation masks.
        :return: None
        """
        times = []
        test_files: List[str] = os.listdir(self.input_dir)
        test_files = [test_files[0]] + test_files  # Duplicate the first file for warm-up

        self.logger.info(f'Running inference on {len(test_files)} images from {self.input_dir}')
        with torch.no_grad():
            for i, fname in enumerate(os.listdir(self.input_dir)):

                t_load_data_start: float = perf_counter()
                img: Image = Image.open(os.path.join(self.input_dir, fname)).convert("RGB")
                orig_shape: Tuple[int, int] = img.size

                t_preprocess_start: float = perf_counter()
                x: torch.Tensor = self.preprocessor(img).unsqueeze(0).to(self.model.device)

                t_infer_start: float = perf_counter()
                pred: torch.Tensor = self.model(x).argmax(1).squeeze(0) * 32.  # * 32 to match labels

                t_post_process_start: float = perf_counter()
                postprocess: T.Resize = T.Resize(orig_shape[::-1],
                                                 interpolation=T.InterpolationMode.NEAREST)

                pred_full_size: torch.Tensor = postprocess(pred.unsqueeze(0).cpu())
                pred_numpy: np.ndarray = pred_full_size.numpy().astype(np.uint8)
                final_pred: Image.Image = Image.fromarray(pred_numpy.squeeze())
                assert final_pred.size == orig_shape, \
                    f'Expected final prediction size {orig_shape}, got {final_pred.size}'
                t_main_end: float = perf_counter()

                if i == 0:
                    self.logger.info(f'Warm-up inference completed for {fname}, skipping time logging.')
                    continue

                times.append(t_main_end - t_load_data_start)

                # Save prediction mask
                fname_png: str = self._change_name_to_png(fname)
                final_pred.save(os.path.join(self.output_dir, fname_png))
                if self.is_create_gifs:
                    gif_path: str = os.path.join(self.output_dir,
                                                 f'{os.path.splitext(fname)[0]}_alt.gif')
                    self._create_alternating_gif(img, final_pred, gif_path)
                    self._plot_legend()

                self.logger.info(f'Inference timings for {fname}: '
                                 f'load_data: {t_preprocess_start - t_load_data_start:.4f}, '
                                 f'preprocess: {t_infer_start - t_preprocess_start:.4f}, '
                                 f'inference: {t_post_process_start - t_infer_start:.4f}s, '
                                 f'postprocess: {t_main_end - t_post_process_start:.4f}s, '
                                 f'total: {t_main_end - t_load_data_start:.4f}s')
        # Exclude first time (warm-up)
        avg_time: float = sum(times) / len(times) if times else 0.0
        self.logger.info(f'Average inference time per image: {avg_time:.4f}s')


if __name__ == "__main__":
    """
    Run inference on a set of images using a trained segmentation model
    and save the output segmentation masks.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    logging_level = 'info'
    config = Config()
    model_path_main = config.final_checkpoint_path
    inference_manager = InferenceManager(config=config,
                                         input_dir=args.input,
                                         output_dir=args.output,
                                         model_path=model_path_main,
                                         logging_level=logging_level)
    inference_manager.run_inference()
