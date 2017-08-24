import argparse
import json
import os

import scipy
import scipy.ndimage
import skimage.data


def fit_square(lbl, i):
    """Fits a square around region i in the given label image.
    Returns center, radius, and rotation."""
    x_arr, y_arr = scipy.where(lbl == i)
    assert len(x_arr) == len(y_arr)
    pixel_count = len(x_arr)
    points = scipy.array([x_arr, y_arr])
    m = scipy.mean(points, axis=1)
    distances = points - scipy.repeat(m.reshape(len(m), 1), pixel_count, 1)
    distances = scipy.linalg.norm(distances, axis=0)
    max_index = scipy.argmax(distances)
    max_distance = distances[max_index]
    most_distant_point = points[:, max_index]
    angle_vector = most_distant_point - m
    angle = scipy.math.atan2(angle_vector[1], angle_vector[0])
    return tuple(m), max_distance, angle


def extract_rectangles(img_filename: str, output_filename: str) -> None:
    """Loads the given image, performs a marker extraction, and stores the centroids in the given output file."""
    # Load image from disk.
    img_raw = skimage.data.imread(img_filename, as_grey=True)

    # Convert image to black and white.
    img = img_raw[:]
    img[img < 0.5] = 0
    img[img >= 0.5] = 1

    # Invert colors because label image needs black background.
    img = 1-img
    assert isinstance(img, scipy.ndarray)

    # We assume that all seats are separated.
    # Create image labels.
    lbl, lbl_count = scipy.ndimage.label(img)

    # Fit rectangle around each label.
    squares = [fit_square(lbl, i) for i in range(1, lbl_count)]

    # Store results in output file.
    results = {
        "original_image_size": img.shape,
        "squares": squares
    }
    with open(output_filename, "w") as f:
        json.dump(results, f)


def initialize_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initializes the given argument parser."""
    parser.description = "Extracts markers from the given image and stores their centroids in the output file."
    parser.add_argument("-i", "--image", type=str, required=True, help="file name of input image")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file name")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output file if it already exists")
    return parser


def main(args: argparse.Namespace=None) -> None:
    """Calls extract_rectangles() with the given arguments.

    Parses the command line arguments if args is None. If args is not None it should be obtained from a parser that was
    set up with initialize_arg_parser() from this module.
    """
    if args is None:
        parser = argparse.ArgumentParser()
        initialize_arg_parser(parser)
        args = parser.parse_args()
    if os.path.isfile(args.output) and not args.overwrite:
        raise RuntimeError("Output file already exists:", args.output)
    extract_rectangles(args.image, args.output)


if __name__ == "__main__":
    main()
