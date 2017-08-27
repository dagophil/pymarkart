import argparse
import json
from typing import Generator, Sequence

import scipy
import scipy.linalg
import scipy.spatial

from dto import DTODecoder, DTOEncoder, Marker


def find_angles(markers: Sequence[Marker]) -> Generator[float, None, None]:
    """Computes the angles of the given markers and returns them."""
    # Compute distance matrix.
    positions = scipy.array(list((m.position.x, m.position.y) for m in markers))
    distance_matrix = scipy.spatial.distance.pdist(positions)
    distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    num_markers = distance_matrix.shape[0]

    # Find nearest neighbors.
    scipy.fill_diagonal(distance_matrix, scipy.inf)
    nearest_neighbors = scipy.argmin(distance_matrix, axis=1)

    # Use direction to nearest neighbor as angle.
    for i in range(num_markers):
        n = nearest_neighbors[i]
        v = positions[n] - positions[i]
        yield scipy.math.atan2(v[1], v[0])


def initialize_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initializes the given argument parser with the arguments for this module."""
    parser.description = "Finds the angles between the given markers."
    parser.add_argument("-i", "--input", type=str, required=True, help="input json file")
    return parser


def main(args: argparse.Namespace=None) -> None:
    """Reads the positions from the input file, computes their orientations, and stores them in the same file.

    Parses the command line arguments if args is None. If args is not None it should be obtained from a parser that was
    set up with initialize_arg_parser() from this module.
    """
    if args is None:
        parser = argparse.ArgumentParser()
        initialize_arg_parser(parser)
        args = parser.parse_args()

    with open(args.input, "r") as f:
        input_data = json.load(f, cls=DTODecoder)
    markers = input_data["markers"]
    angles = find_angles(markers)
    for marker, angle in zip(markers, angles):
        assert isinstance(marker, Marker)
        marker.orientation = angle
    with open(args.input, "w") as f:
        json.dump(input_data, f, cls=DTOEncoder)


if __name__ == "__main__":
    main()
