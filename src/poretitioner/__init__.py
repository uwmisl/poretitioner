import sys
import os
from pathlib import Path

from .poretitioner import main


# project_root_dir = Path(__file__).parent.parent.resolve()
# PROJECT_DIR_LOCATION = str(project_root_dir)


# def add_poretitioner_to_path():
#     poretitioner_directory = str(Path(PROJECT_DIR_LOCATION, "poretitioner"))
#     sys.path.append(poretitioner_directory)
#     os.chdir(PROJECT_DIR_LOCATION)


# add_poretitioner_to_path()
# ./result/bin/poretitioner segment --config ../config.toml --file "./src/tests/data/bulk_fast5_dummy.fast5" --output-dir "."