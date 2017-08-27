"""
Contains a collection of data transfer objects.
The classes can be serialized to and from json using the classes DTOEncoder and DTODecoder.
"""

import builtins
import json
from typing import Any


class DTOEncoder(json.JSONEncoder):
    """This json encoder dumps dto objects as dicts. The created dicts can be loaded by the DTODecoder.

     The encoder creates a dict that contains the dto type name and the object's internal __dict__.
     """

    def default(self, obj: Any) -> dict:
        """Returns a dict that represents obj if obj is a dto type. Raises a TypeError if obj is not a dto type."""
        if obj.__module__ == __name__:
            return {
                "py/type": obj.__module__ + "." + obj.__class__.__name__,
                "py/data": obj.__dict__
            }
        json.JSONEncoder.default(self, obj)


def get_type_from_string(type_name: str) -> type:
    """Returns the type that is represented by the given string.

    Example:
        get_type_from_string("list") returns the list type.
        get_type_from_string("dto.Point") returns the dto.Point type.
    """
    components = type_name.split(".")
    if len(components) == 1:
        return getattr(builtins, components[0])
    t = __import__(components[0])
    for component in components[1:]:
        t = getattr(t, component)
    return t


class DTODecoder(json.JSONDecoder):
    """This json decoder loads strings that were created from a DTOEncoder."""

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the decoder by setting the object_hook."""
        super().__init__(object_hook=DTODecoder.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(obj: dict) -> Any:
        """Creates a dto from obj and returns it.

        The dto is created using obj["py/type"] as type and obj["py/data"] as internal dict.
        If the given dict does not contain any of those keys it is returned as is.
        """
        if "py/type" in obj and "py/data" in obj:
            obj_type_string = obj["py/type"]
            obj_type = get_type_from_string(obj_type_string)
            obj_data = obj["py/data"]
            return obj_type(**obj_data)
        return obj


class Point(object):
    """Point with x and y coordinates."""

    def __init__(self, x: float, y: float) -> None:
        """Sets x and y."""
        self.x = x
        self.y = y


class Marker(object):
    """Marker with position, radius, and orientation."""

    def __init__(self, position: Point=None, radius: float=None, orientation: float=None) -> None:
        """Sets position, radius, and orientation."""
        self.position = position
        self.radius = radius
        self.orientation = orientation
