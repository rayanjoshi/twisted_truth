"""
Main script for executing a sequence of data processing scripts.

This module configures logging and runs a predefined list of Python scripts
located in the src directory. It captures and logs the output or errors
from each script execution, ensuring robust error handling and logging.
"""
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
logger = logging.getLogger("run.py")


def main():
    """Execute a series of Python scripts in the src directory and log their output.

    This function locates the source directory relative to the current script,
    defines a list of scripts to run, and executes each one using subprocess.
    It logs the output or errors for each script execution.

    Raises:
        SystemExit: If a subprocess.CalledProcessError occurs, exits with status code 1.
    """
    repo_root = Path(__file__).parent
    src_dir = Path(repo_root / "src").resolve()

    scripts = [
            ('Data Loader', 'data_loader.py'),
            ('Data Visualiser', 'data_visualiser.py'),
            ('Correlation Calculator', 'calculating_correlation.py')
        ]

    try:
        for script_name, script_file in scripts:
            logger.info("Running %s...", script_name)

            script_path = src_dir / script_file
            command = [sys.executable, str(script_path)]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            if result.stdout:
                logger.info("Output from %s: %s", script_name, result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logger.error("Error occurred while running %s: %s", script_name, e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
