from .getargs import ARG, COMMAND, get_args
from .poretitioner import main

if __name__ == "__main__":
    # To test the application with pre-configured command line arguments,
    # set `use_fake_command_line` to True and/or modify the `command_line` list
    # with whatever arguments you'd like:
    use_fake_command_line = True
    if use_fake_command_line:
        command_line = [
            "segment",
            "--file=./tests/data/bulk_fast5_dummy.fast5",
            "--output-dir=./out/data/",
            "-vvvvv",
        ]
        args = get_args(command_line)
    else:
        args = get_args()
    # test_fast5()
    main(args)
