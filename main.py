import argparse

import create_image
import extract_markers


def initialize_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initializes the given argument parser with the arguments for this module."""
    parser.description = "Executes the selected workflow."
    subparsers = parser.add_subparsers(dest="workflow", help="the executed workflow")
    subparsers.required = True
    extract_markers_parser = subparsers.add_parser("extract_markers")
    extract_markers.initialize_arg_parser(extract_markers_parser)
    create_image_parser = subparsers.add_parser("create_image")
    create_image.initialize_arg_parser(create_image_parser)
    return parser


def main(args: argparse.Namespace=None) -> None:
    """Executes the selected workflow with the given arguments.

    Parses the command line arguments if args is None. If args is not None it should be obtained from a parser that was
    set up with initialize_arg_parser() from this module."""
    if args is None:
        parser = argparse.ArgumentParser()
        initialize_arg_parser(parser)
        args = parser.parse_args()
    if args.workflow == "extract_markers":
        extract_markers.main(args)
    elif args.workflow == "create_image":
        create_image.main(args)
    else:
        raise RuntimeError("Unknown workflow:", args.workflow)


if __name__ == "__main__":
    main()
