import argparse
import json
import os
from typing import Iterable, Tuple

import pyx
import scipy
import scipy.spatial


Point = Tuple[float, float]
Marker = Tuple[Point, float]


def fill_square(canvas: pyx.canvas.canvas, centroid: Point, radius: float, angle: float) -> None:
    """Draws a rotated square with the given (diagonal) radius at the given position into the canvas."""
    # Get vector (vx, vy) from centroid to first corner.
    angle += scipy.math.pi/4.0
    vx = radius * scipy.math.cos(angle)
    vy = radius * scipy.math.sin(angle)

    # Get the square corners.
    p0 = centroid[0]+vx, centroid[1]+vy
    p1 = centroid[0]-vy, centroid[1]+vx
    p2 = centroid[0]-vx, centroid[1]-vy
    p3 = centroid[0]+vy, centroid[1]-vx

    # Draw the square.
    p = pyx.path.line(p0[0], p0[1], p1[0], p1[1]) \
        << pyx.path.line(p1[0], p1[1], p2[0], p2[1]) \
        << pyx.path.line(p2[0], p2[1], p3[0], p3[1])
    p.append(pyx.path.closepath())
    canvas.stroke(p, [pyx.deco.filled([pyx.color.rgb.black])])


def create_image(markers: Iterable[Marker]) -> pyx.canvas.canvas:
    """Draws the given markers into a canvas and returns it."""
    # Find mean radius.
    mean_radius = scipy.mean(list(m[1] for m in markers))
    assert isinstance(mean_radius, float)

    # Compute distance matrix.
    centroids = scipy.array(list(m[0] for m in markers))
    distance_matrix = scipy.spatial.distance.pdist(centroids)
    distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
    scipy.fill_diagonal(distance_matrix, scipy.inf)
    nearest_neighbors = scipy.argmin(distance_matrix, axis=0)

    # Compute angles as angle between nearest neighbors.
    orientations = centroids - centroids[nearest_neighbors]
    angles = list(scipy.math.atan2(v[1], v[0]) for v in orientations)

    # Draw the markers as rotated squares into a canvas.
    canvas = pyx.canvas.canvas()
    for i, m in enumerate(markers):
        fill_square(canvas, m[0], mean_radius, angles[i])
    return canvas


def initialize_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initializes the given argument parser."""
    parser.description = "Draws the squares specified in the input file into the output image."
    parser.add_argument("-i", "--input", type=str, required=True, help="input file name")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file name")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output file if it already exists")
    return parser


def main(args: argparse.Namespace=None) -> None:
    """Loads markers from the input file, draws them into an image, and saves the image as EPS file.

    Parses the command line arguments if args is None. If args is not None it should be obtained from a parser that was
    set up with initialize_arg_parser() from this module.
    """
    if args is None:
        parser = argparse.ArgumentParser()
        initialize_arg_parser(parser)
        args = parser.parse_args()
    if os.path.isfile(args.output) and not args.overwrite:
        raise RuntimeError("Output file already exists:", args.output)

    with open(args.input, "r") as f:
        input_data = json.load(f)
    canvas = create_image(input_data["markers"])
    canvas.writeEPSfile(args.output)


if __name__ == "__main__":
    main()
