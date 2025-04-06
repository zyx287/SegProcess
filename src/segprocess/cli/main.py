"""
Main entry point for the segprocess command-line interface.

This module provides a central command-line interface for the various
functionalities provided by the segprocess package.
"""

import sys
import logging
import pkgutil
import importlib
import timeit
from argparse import ArgumentParser
from typing import List, Optional, Dict, Any, Union

import segprocess.cli as cli_package

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main(commandline_arguments: Optional[List[str]] = None) -> int:
    """
    Main entry point for the segprocess CLI.
    
    Args:
        commandline_arguments: Command line arguments. If None, sys.argv[1:] is used.
        
    Returns:
        Exit code.
    """
    parser = ArgumentParser(description=__doc__, prog="segprocess")
    
    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest="command")

    # Each subcommand is implemented as a module in the cli subpackage
    modules = pkgutil.iter_modules(cli_package.__path__)

    for _, module_name, _ in modules:
        # Skip __init__.py and internal modules
        if module_name.startswith('_'):
            continue
            
        try:
            # Import the module
            module = importlib.import_module("." + module_name, cli_package.__name__)
            
            # Get help text from module docstring
            if module.__doc__:
                help_text = module.__doc__.strip().split("\n", maxsplit=1)[0]
            else:
                help_text = f"{module_name} command"
                
            # Create subparser for this command
            subparser = subparsers.add_parser(
                module_name,
                help=help_text,
                description=module.__doc__
            )
            
            # Add arguments specific to this command
            if hasattr(module, "add_arguments"):
                module.add_arguments(subparser)
                
            # Set the module as a default for this command
            subparser.set_defaults(module=module)
            
        except Exception as e:
            logger.error(f"Error loading module {module_name}: {e}")

    # Parse arguments
    args = parser.parse_args(commandline_arguments)

    # Handle the case where no command is specified
    if not hasattr(args, "module"):
        parser.print_help()
        return 0
        
    # Get the module for the selected command
    module = args.module
    del args.module

    # Print settings for module
    module_name = module.__name__.split('.')[-1]
    sys.stderr.write(f"SETTINGS FOR: {module_name} \n")
    for object_variable, value in vars(args).items():
        sys.stderr.write(f" {object_variable}: {value}\n")

    # Set timer
    tic = timeit.default_timer()
    logger.info(f"Starting {module_name}")

    # Run command
    exit_code = 0
    try:
        if hasattr(module, "main"):
            result = module.main(args)
            if isinstance(result, int):
                exit_code = result
        else:
            logger.error(f"Module {module_name} has no main function")
            exit_code = 1
    except Exception as e:
        logger.error(f"Error executing {module_name}: {e}")
        exit_code = 1

    # Stop timer
    toc = timeit.default_timer()
    processing_time = toc - tic
    logger.info(f"Elapsed time ({module_name}): {round(processing_time, 4)}s")
    
    if exit_code == 0:
        logger.info(f"Command {module_name} completed successfully")
    else:
        logger.error(f"Command {module_name} failed with exit code {exit_code}")
        
    return exit_code


if __name__ == "__main__":
    sys.exit(main())