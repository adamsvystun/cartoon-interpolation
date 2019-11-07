import argparse

import numpy as np
import Augmentor
from PIL import Image

from src.utils import batch


def get_triplets(path):
    return []


def get_images(paths):
    return [[np.asarray(Image.open(y)) for y in x] for x in paths]


def save_images(output_path, images):
    pass


def apply_augmentation_pipeline(p):
    p.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p.flip_top_bottom(0.5)
    p.zoom_random(1, percentage_area=0.5)
    return p


def augment(options):
    all_triplets = get_triplets(options.path)
    for triplets_batch in batch(all_triplets):
        images = get_images(triplets_batch)
        p = Augmentor.DataPipeline(images)
        p = apply_augmentation_pipeline(p)
        augmented_images = p.sample(int(len(images) * options.augmentation_mulitplier))
        save_images(options.output_path, augmented_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dir', '-i', type=str, required=True, help='Directory where input scene images are located'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, required=True, help='Directory where to save output frames'
    )
    parser.add_argument(
        '--input-dataset-file-name', '-idfm', type=str, required=True,
        help='Name of the input dataset csv file describing each group of frames'
    )
    parser.add_argument(
        '--output-dataset-file-name', '-odfm', type=str, required=True,
        help='Name of the output dataset csv file describing each group of frames'
    )
    parser.add_argument(
        '--augmentation-multiplier', '-am', type=int, required=True, default=4,
        help='By how many times to increase the size of the dataset'
    )

    args = parser.parse_args()
    augment(args)
