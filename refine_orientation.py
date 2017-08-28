import argparse
import json
from typing import Any, Iterable, List, Tuple

from matplotlib.backend_bases import MouseEvent, PickEvent
import matplotlib.pyplot as plt
import scipy

from dto import DTODecoder, DTOEncoder, Marker


def find_limits(markers: Iterable[Marker]) -> Tuple[float, float, float, float]:
    """Returns minimum and maximum values of the marker positions.
    The values are scaled by 2% so that they can be used as limits in a plot.

    Returns:
        x_min, x_max, y_min, y_max
    """
    x_min = min(m.position.x for m in markers)
    x_max = max(m.position.x for m in markers)
    x_delta = 0.02 * (x_max - x_min)
    x_min -= x_delta
    x_max += x_delta
    y_min = min(m.position.y for m in markers)
    y_max = max(m.position.y for m in markers)
    y_delta = 0.02 * (y_max - y_min)
    y_min -= y_delta
    y_max += y_delta
    return x_min, x_max, y_min, y_max


def create_circles(markers: Iterable[Marker], color: Any) -> List[plt.Circle]:
    """Creates a circle around each marker and returns them."""
    return [plt.Circle((m.position.x, m.position.y), m.radius, color=color) for m in markers]


def create_arrows(markers: Iterable[Marker], color: Any, width: float) -> List[plt.Arrow]:
    """Creates an arrow on each marker that represents the marker orientation and returns them."""
    arrows = []
    for m in markers:
        x = m.position.x
        y = m.position.y
        dx = m.radius * scipy.cos(m.orientation)
        dy = m.radius * scipy.sin(m.orientation)
        arrows.append(plt.Arrow(x-dx, y-dy, 2*dx, 2*dy, color=color, width=width))
    return arrows


class MarkerRefinementGui(object):
    """Creates a GUI that shows the markers with their orientation and refines them with user input."""

    ARROW_COLOR = (0, 0, 0, 1)
    CIRCLE_COLOR = (0, 0, 0, 0.2)
    CIRCLE_ACTIVE_COLOR = (1, 0, 0, 0.4)

    def __init__(self, markers: Iterable[Marker]) -> None:
        """Creates a plot from the given markers and updates their orientation from user input."""
        # Create the plot.
        x_min, x_max, y_min, y_max = find_limits(markers)
        self._fig, self._ax = plt.subplots()
        self._ax.set_xlim((x_min, x_max))
        self._ax.set_ylim((y_min, y_max))
        self._ax.set_aspect("equal")
        plt.subplots_adjust(bottom=0.2)

        self._markers = [m for m in markers]
        self._angles = [m.orientation for m in markers]
        self._arrows = create_arrows(markers, color=self.ARROW_COLOR, width=5.0)
        self._circles = create_circles(markers, color=self.CIRCLE_COLOR)
        self._selected_index = None
        self._ignore_next_click = False
        self._fig_background = None
        btn_ax = plt.axes([0.4, 0.02, 0.2, 0.08])
        self._btn_accept = plt.Button(btn_ax, "Save")
        self._btn_accept.on_clicked(self.on_accept)
        self._accepted = False

        for i, (arrow, circle) in enumerate(zip(self._arrows, self._circles)):
            circle.__dict__["user_index"] = i
            circle.set_picker(True)
            self._ax.add_artist(circle)
            self._ax.add_artist(arrow)

        self._fig.canvas.mpl_connect("resize_event", self.store_background)
        self._fig.canvas.mpl_connect("scroll_event", self.store_background)
        self._fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self._fig.canvas.mpl_connect("button_press_event", self.on_click)
        self._fig.canvas.callbacks.connect("pick_event", self.on_pick)

    @property
    def accepted(self) -> bool:
        """Returns whether the accept button has been clicked."""
        return self._accepted

    @property
    def angles(self) -> List[float]:
        """Returns the angle list."""
        return self._angles

    def show(self):
        """Shows the gui."""
        plt.show()

    def store_background(self, *args, **kwargs) -> None:
        """Draws all objects in the current figure and stores the canvas."""
        self._fig.canvas.draw()
        self._fig_background = self._fig.canvas.copy_from_bbox(self._ax.bbox)

    def on_accept(self, *args, **kwargs) -> None:
        """Closes the figure and sets this handler to accepted."""
        self._accepted = True
        plt.close(self._fig)

    def on_mouse_move(self, event: MouseEvent) -> None:
        """Computes orientation of marker, depending on mouse position, and updates the respective arrow."""
        # Early out if nothing is selected or if the cursor is not on the canvas.
        if self._selected_index is None or event.xdata is None or event.ydata is None:
            return

        # Compute angle from marker position to mouse.
        # Store delta to previous angle. Rotation in plot can not be set with absolute values, only relative ones.
        m = self._markers[self._selected_index]
        x = m.position.x
        y = m.position.y
        dx = event.xdata - x
        dy = event.ydata - y
        angle = scipy.math.atan2(dy, dx)
        angle_delta = angle - self._angles[self._selected_index]
        self._angles[self._selected_index] = angle

        # Rotate the arrow and draw it.
        self._fig.canvas.restore_region(self._fig_background)
        arrow = self._arrows[self._selected_index]
        arrow.get_patch_transform().rotate_around(x, y, angle_delta)
        self._ax.draw_artist(arrow)
        self._fig.canvas.blit(self._ax.bbox)

    def on_click(self, event: MouseEvent) -> None:
        """Resets the selected marker."""
        if self._ignore_next_click:
            self._ignore_next_click = False
            return
        if self._selected_index is None:
            return

        # Reset selection.
        circle = self._circles[self._selected_index]
        circle.set_color(self.CIRCLE_COLOR)
        arrow = self._arrows[self._selected_index]
        arrow.set_animated(False)
        self._fig.canvas.draw()
        self._selected_index = None

    def on_pick(self, event: PickEvent) -> None:
        """Sets the clicked marker as selected."""
        # Early out if there already is a selected marker. Let on_click handle the reset.
        if self._selected_index is not None:
            return

        # Make sure that the next click event does not reset the just selected marker.
        self._ignore_next_click = True

        # Get the circle index.
        circle = event.artist
        assert isinstance(circle, plt.Circle)
        self._selected_index = circle.__dict__["user_index"]

        # Update circle and arrow and store the current canvas. The canvas can then be reused for efficient drawing.
        circle.set_color(self.CIRCLE_ACTIVE_COLOR)
        arrow = self._arrows[self._selected_index]
        arrow.set_animated(True)
        self.store_background()


def refine_markers(markers: Iterable[Marker]) -> Tuple[bool, List[float]]:
    """Shows a plot with the markers and their orientation and asks the user to refine their orientation.

    Returns a tuple t where t[0] is a bool that indicates whether the user accepted the input and t[1] is a list with
    the refined angles.
    """
    refinement_gui = MarkerRefinementGui(markers)
    refinement_gui.show()
    return refinement_gui.accepted, refinement_gui.angles


def initialize_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initializes the given argument parser with the arguments for this module."""
    parser.description = "Asks for user input to refine the angles of the given markers."
    parser.add_argument("-i", "--input", type=str, required=True, help="input json file")
    return parser


def main(args: argparse.ArgumentParser=None) -> None:
    """Reads the marker positions and orientations from the input file and asks the user to refine them.

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
    accepted, angles = refine_markers(markers)
    if accepted:
        for marker, angle in zip(markers, angles):
            assert isinstance(marker, Marker)
            marker.orientation = angle
        with open(args.input, "w") as f:
            json.dump(input_data, f, cls=DTOEncoder)


if __name__ == "__main__":
    main()
