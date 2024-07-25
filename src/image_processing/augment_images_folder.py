from pathlib import Path
from typing import Final

from augment_images import perform_augmentation


INPUT_ROOT: Final[Path] = Path(R'D:\TexturesKTH-44-48-60\3')
OUTPUT_ROOT: Final[Path] = Path(R'D:\TexturesKTH-44-48-60-Augmented')

RES: Final[int] = 112

# NUM_AUGMENTATIONS: Final[dict] = {'1': 10, '2': 5, '3': 10}
NUM_AUGMENTATIONS: Final[dict] = {'1': 10, '2': 10, '3': 10}

for patient in INPUT_ROOT.iterdir():

    class_label = INPUT_ROOT.name

    output_dir = OUTPUT_ROOT / class_label / patient.stem
    output_dir.mkdir(exist_ok=True, parents=True)

    perform_augmentation(resolution=RES,
                         num_augmentations=NUM_AUGMENTATIONS[class_label],
                         image_dir=patient,
                         output_dir=output_dir)
