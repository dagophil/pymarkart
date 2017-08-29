import argparse
import json
import os
from typing import Iterable, Tuple

import pyx
import scipy
import scipy.spatial

from dto import DTODecoder, Marker, Point


def fill_square(canvas: pyx.canvas.canvas, position: Point, radius: float, angle: float) -> None:
    """Draws a rotated square with the given (diagonal) radius at the given position into the canvas."""
    # Get vector (vx, vy) from centroid to first corner.
    angle += scipy.math.pi/4.0
    vx = radius * scipy.math.cos(angle)
    vy = radius * scipy.math.sin(angle)

    # Get the square corners.
    p0 = position.x+vx, position.y+vy
    p1 = position.x-vy, position.y+vx
    p2 = position.x-vx, position.y-vy
    p3 = position.x+vy, position.y-vx

    # Draw the square.
    p = pyx.path.line(p0[0], p0[1], p1[0], p1[1]) \
        << pyx.path.line(p1[0], p1[1], p2[0], p2[1]) \
        << pyx.path.line(p2[0], p2[1], p3[0], p3[1])
    p.append(pyx.path.closepath())
    canvas.stroke(p, [pyx.deco.filled([pyx.color.rgb.black])])


def get_bounding_box(canvas: pyx.canvas.canvas) \
        -> Tuple[pyx.unit.length, pyx.unit.length, pyx.unit.length, pyx.unit.length]:
    """Returns a tuple (x0, x1, y0, y1) with the bounding box of the given canvas."""
    b = canvas.bbox()
    return b.left(), b.right(), b.bottom(), b.top()


def get_scale_from_ratio(ratio: float, width: pyx.unit.length, height: pyx.unit.length) -> Tuple[float, float]:
    """Returns a pair (scale_x, scale_y) that, when multiplied with (width, height), complies with the given ratio.

    Mathematically spoken, the returned values satisfy the equation (width*scale_x)/(height*scale_y) == ratio.
    The following constraints are fulfilled:
        * The scales are enlarging, not shrinking (scale_x >= 1.0, scale_y >= 1.0).
        * The scales are minimal (scale_x == 1.0 or scale_y == 1.0).
    """
    desired_width = height * ratio
    if width < desired_width:
        return desired_width / width, 1.0
    desired_height = width / ratio
    if height < desired_height:
        return 1.0, desired_height / height
    return 1.0, 1.0


def apply_scale_centered(scale: float, a: pyx.unit.length, b: pyx.unit.length) \
        -> Tuple[pyx.unit.length, pyx.unit.length]:
    """Returns a scaled version (x, y) of (a, b) such that the difference of a and b is scaled by the given scale but
    their mid point stays unchanged.

    Mathematically spoken, the returned values satisfy the following equations:
        (1) y-x == scale*(b-a)
        (2) (x+y)/2 == (a+b)/2
    """
    a_scaled = (a*(1+scale) + b*(1-scale)) / 2
    b_scaled = (a*(1-scale) + b*(1+scale)) / 2
    return a_scaled, b_scaled


def draw_frame(canvas: pyx.canvas.canvas,
               frame_ratio: float=None,
               frame_scale: float=1.0,
               double_line: bool=False) -> None:
    """Draws a frame around the content of the given canvas.

    Sets the frame ratio to frame_ratio if it is not None.
    Sets the frame size to frame_scale * content_size.
    Draws two line with different line widths if double_line is True.
    """
    # Get bounding box of current canvas content.
    x0, x1, y0, y1 = get_bounding_box(canvas)
    assert x0 <= x1
    assert y0 <= y1

    # Get the scaling from the given scale and ratio.
    scale_x = frame_scale
    scale_y = frame_scale
    if frame_ratio is not None:
        ratio_scale_x, ratio_scale_y = get_scale_from_ratio(frame_ratio, x1-x0, y1-y0)
        scale_x *= ratio_scale_x
        scale_y *= ratio_scale_y

    # Apply the scaling to the frame coordinates.
    x0, x1 = apply_scale_centered(scale_x, x0, x1)
    y0, y1 = apply_scale_centered(scale_y, y0, y1)

    # Get a line width relative to the content size.
    line_width = min(x1-x0, y1-y0) / 200

    # Draw the inner frame.
    p = pyx.path.line(x0, y0, x1, y0) \
        << pyx.path.line(x1, y0, x1, y1) \
        << pyx.path.line(x1, y1, x0, y1)
    p.append(pyx.path.closepath())
    canvas.stroke(p, [pyx.style.linewidth(line_width)])

    # Draw the outer frame.
    if double_line:
        x0 -= 2*line_width
        x1 += 2*line_width
        y0 -= 2*line_width
        y1 += 2*line_width
        p = pyx.path.line(x0, y0, x1, y0) \
            << pyx.path.line(x1, y0, x1, y1) \
            << pyx.path.line(x1, y1, x0, y1)
        p.append(pyx.path.closepath())
        canvas.stroke(p, [pyx.style.linewidth(line_width/2)])


def create_image(markers: Iterable[Marker],
                 use_mean_radius: bool=False,
                 frame_ratio: float=None,
                 frame_scale: float=1.0,
                 double_frame_line: bool=False) -> pyx.canvas.canvas:
    """Draws the given markers into a canvas and returns the canvas."""
    # Find mean radius.
    mean_radius = None
    if use_mean_radius:
        mean_radius = scipy.mean(list(m.radius for m in markers))
        assert isinstance(mean_radius, float)

    # Draw the markers as rotated squares into a canvas.
    canvas = pyx.canvas.canvas()
    for i, m in enumerate(markers):
        r = mean_radius or m.radius
        fill_square(canvas, m.position, r, m.orientation)

    # Draw the frame.
    draw_frame(canvas, frame_ratio=frame_ratio, frame_scale=frame_scale, double_line=double_frame_line)

    return canvas


def initialize_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initializes the given argument parser with the arguments for this module."""
    parser.description = "Draws the squares specified in the input file into the output image."
    parser.add_argument("-i", "--input", type=str, required=True, help="input file name")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file name")
    parser.add_argument("--overwrite", action="store_true", help="overwrite output file if it already exists")
    parser.add_argument("--use_mean_radius", action="store_true", help="use mean marker radius for all markers")
    parser.add_argument("--frame_ratio", type=float, default=None, help="use given ratio for the frame")
    parser.add_argument("--frame_scale", type=float, default=1.0, help="size of frame relative to content size")
    parser.add_argument("--double_frame_line", action="store_true", help="draw the frame with two lines")
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
        input_data = json.load(f, cls=DTODecoder)
    canvas = create_image(input_data["markers"],
                          use_mean_radius=args.use_mean_radius,
                          frame_ratio=args.frame_ratio,
                          frame_scale=args.frame_scale,
                          double_frame_line=args.double_frame_line)

    page = pyx.document.page(canvas, fittosize=True, paperformat=pyx.document.paperformat.A4)
    doc = pyx.document.document([page])
    doc.writeEPSfile(args.output)


if __name__ == "__main__":
    main()
