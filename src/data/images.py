from typing import List, Tuple

from PIL import Image


def load_images(files: List[str], labels: List[int]) -> Tuple[List[Image.Image], List[int]]:
    """
    Load images and transform them according to the model to be used.
    :param files: List of paths to images.
    :param labels: List of labels corresponding to the images.
    :return: Array of images after transformation.
    """
    inputs = []
    input_labels = []
    for path, lab in zip(files, labels):
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
            inputs.append(image)
            input_labels.append(lab)
        except:
            print("Could not load image {}".format(path))
    return inputs, input_labels
