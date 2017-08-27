import argparse
import collections

import create_image
import extract_markers


Workflow = collections.namedtuple("Workflow", ["main", "init_parser"])

WORKFLOWS = {
    "extract_markers": Workflow(extract_markers.main, extract_markers.initialize_arg_parser),
    "create_image": Workflow(create_image.main, create_image.initialize_arg_parser)
}


def initialize_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Initializes the given argument parser with the arguments for this module."""
    parser.description = "Executes the selected workflow."
    subparsers = parser.add_subparsers(dest="workflow", help="the executed workflow")
    subparsers.required = True
    for workflow_name, workflow in WORKFLOWS.items():
        assert isinstance(workflow, Workflow)
        subparser = subparsers.add_parser(workflow_name)
        workflow.init_parser(subparser)
    return parser


def main(args: argparse.Namespace=None) -> None:
    """Executes the selected workflow with the given arguments.

    Parses the command line arguments if args is None. If args is not None it should be obtained from a parser that was
    set up with initialize_arg_parser() from this module."""
    if args is None:
        parser = argparse.ArgumentParser()
        initialize_arg_parser(parser)
        args = parser.parse_args()

    if args.workflow not in WORKFLOWS:
        raise RuntimeError("Unknown workflow:", args.workflow)
    w = WORKFLOWS[args.workflow]
    return w.main(args)


if __name__ == "__main__":
    main()
