"""
Package for processing segmentation data
Modified from @Tobias Frick, github.com/FrickTobias/nd2tools
"""
import sys
import logging
import pkgutil
import importlib
import timeit
from argparse import ArgumentParser

import segprocess.cli as cli_package

logger = logging.getLogger(__name__)


def main(commandline_arguments=None) -> int:

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s: %(message)s")

    parser = ArgumentParser(description=__doc__, prog="segprocess")
    

    subparsers = parser.add_subparsers()

    # Each subcommand is implemented as a module in the cli subpackage.
    # It needs to implement an add_arguments() and a main() function.
    modules = pkgutil.iter_modules(cli_package.__path__)

    for _, module_name, _ in modules:
        module = importlib.import_module("." + module_name, cli_package.__name__)
        help = module.__doc__.strip().split("\n", maxsplit=1)[0]
        subparser = subparsers.add_parser(
            module_name,
            help=help,
            description=module.__doc__
        )
        subparser.set_defaults(module=module)

        module.add_arguments(subparser)

    args = parser.parse_args(commandline_arguments)


    if not hasattr(args, "module"):
        parser.error("Please provide the name of a subcommand to run")
    else:
        module = args.module
        del args.module

        # Print settings for module
        module_name = module.__name__.split('.')[-1]
        sys.stderr.write(f"SETTINGS FOR: {module_name} \n")
        for object_variable, value in vars(args).items():
            sys.stderr.write(f" {object_variable}: {value}\n")

        # Set timer
        tic = timeit.default_timer()
        logger.info(f"Starting timer ({module_name})")

        # Run submodule
        module.main(args)

        # Stop timer
        toc = timeit.default_timer()
        processing_time = toc - tic
        logger.info(f"Elapsed time ({module_name}): {round(processing_time, 4)}s")
        logger.info("Done")
    return 0



if __name__ == "__main__":
    sys.exit(main())
