from argparse import ArgumentParser
from simclr.experiment import Experiment
from simclr.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = Experiment()
    match args["command"]:
        case "train":
            module.train(save_path=args["save_path"], config_path=args["config_path"])

        case _:
            return False

    return True


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Start experiment with the SimCLR framework")

    parser = setup_parser(parser)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    if not execute(args_dict):
        parser.print_usage()


if __name__ == "__main__":
    main()
