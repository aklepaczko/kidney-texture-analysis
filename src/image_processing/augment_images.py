"""Augment images for classification

Usage:
    augment_images.py [options] <input-dir> <output-dir>
    augment_images.py -h | --help
    augment_images.py --version
    
Arguments:
    <input-dir>     Path to the input directory with images.
    <output-dir>    Path to the output directory with augmented images.
    
Options:
    --num-augs <value>      Number of augmentations per image [default: 10].
    --resolution <value>    Output image width and height in pixels [default: 128].
    -h --help               Show this screen.
    --version               Show version.
"""
from pathlib import Path
from typing import Any, Dict, Optional

import albumentations as A
import cv2
from docopt import docopt
import numpy as np


def _get_image(file_path: str | Path, ext):
    if ext == '.npy':
        image = np.load(str(file_path))
    elif ext in ['.jpg', '.jpeg', '.png']:
        image = cv2.imread(str(file_path))         
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / np.max(image).item()
        image *= 255
        image = image.astype(np.uint8)
    else:
        raise ValueError('Unsupported image format')
    return image


def perform_augmentation(resolution: int, num_augmentations: int, image_dir: Path, output_dir: Path) -> None:
    transform = A.Compose([
        A.SmallestMaxSize(max_size=200),
        A.ShiftScaleRotate(shift_limit=0,
                           scale_limit=0.05,
                           rotate_limit=30,
                           p=0.5,
                           border_mode=cv2.BORDER_REFLECT101),
        A.RandomCrop(width=resolution, height=resolution),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),
    ])

    ext = '.png'
    image_fns = [x for x in image_dir.iterdir() if x.suffix == ext]

    for i in image_fns:
        image = _get_image(i, ext)
        for j in range(num_augmentations):
            transformed = transform(image=image)
            transformed_image = transformed['image']
            out_path = output_dir / f'{i.stem}_{j}{ext}'
            # cv2.imwrite(str(out_path), transformed_image, (cv2.IMWRITE_JPEG_QUALITY, 100))
            cv2.imwrite(str(out_path), transformed_image, (cv2.IMWRITE_PNG_COMPRESSION, 0))
            # np.save(out_path, transformed_image)


def main(args: Dict[str, Optional[Any]]):
    res = int(args['--resolution'])
    path_dataset = Path(args['<input-dir>'])
    out_dir = Path(args['<output-dir>'])
    out_dir.mkdir(parents=True, exist_ok=True)
    num_augmentations = int(args['--num-augs'])
    
    perform_augmentation(resolution=res,
                         num_augmentations=num_augmentations,
                         image_dir=path_dataset,
                         output_dir=out_dir)    


if __name__=='__main__':
   main(docopt(__doc__, version='augment_images.py 0.1.0'))
