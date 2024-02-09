import os
from typing import NoReturn

import matplotlib.pyplot as plt
from PIL import Image


def plot_images(image_paths: str) -> NoReturn:
    """Plot each page in a document as an image

    Args:
        image_paths (str): Image path

    Returns:
        NoReturn:
    """
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(3, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break

    plt.show()
