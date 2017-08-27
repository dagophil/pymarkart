import argparse
import json
import os
from typing import Generator

import scipy
import scipy.ndimage
import skimage.data

from dto import DTOEncoder, Marker, Point


def create_marker_from_region(lbl: scipy.ndarray, i: int) -> Marker:
    """Returns position and radius of the marker in region i of the given label image."""
    x_arr, y_arr = scipy.where(lbl == i)
    assert len(x_arr) == len(y_arr)
    pixel_count = len(x_arr)
    points = scipy.array([x_arr, y_arr])
    m = scipy.mean(points, axis=1)
    distances = points - scipy.repeat(m.reshape(len(m), 1), pixel_count, 1)
    distances = scipy.linalg.norm(distances, axis=0)
    max_index = scipy.argmax(distances)
    max_distance = distances[max_index]
    return Marker(
        position=Point(m[0], m[1]),
        radius=max_distance
    )


def extract_markers(img_raw: scipy.ndarray) -> Generator[Marker, None, None]:
    """Performs a marker extraction on the given image and returns all found markers.

    It is assumed that all markers are clearly separated. A connected black region is treated as single marker.
    """
    # Make a copy so that the original is unchanged.
    img = img_raw[:]

    # Invert colors because marker extraction needs bright markers on dark background.
    img = 1-img
    assert isinstance(img, scipy.ndarray)

    # Create image labels.
    img[img < 0.5] = 0
    img[img >= 0.5] = 1
    lbl, lbl_count = scipy.ndimage.label(img)

    # Extract marker of each region.
    for i in range(1, lbl_count):
        yield create_marker_from_region(lbl, i)


def initialize_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initializes the given argument parser with the arguments for this module."""
    parser.description = "Extracts markers from the given image and stores their centroids in the output file."
    parser.add_argument("-i", "--image", type=str, required=True, help="file name of input image")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file name")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output file if it already exists")
    return parser


def main(args: argparse.Namespace=None) -> None:
    """Opens the input image, performs a marker extraction, and stores the markers in the output file.

    Parses the command line arguments if args is None. If args is not None it should be obtained from a parser that was
    set up with initialize_arg_parser() from this module.
    """
    if args is None:
        parser = argparse.ArgumentParser()
        initialize_arg_parser(parser)
        args = parser.parse_args()
    if os.path.isfile(args.output) and not args.overwrite:
        raise RuntimeError("Output file already exists:", args.output)

    img = skimage.data.imread(args.image, as_grey=True)
    markers = list(extract_markers(img))
    results = {"markers": markers}
    with open(args.output, "w") as f:
        json.dump(results, f, cls=DTOEncoder)


if __name__ == "__main__":
    main()
