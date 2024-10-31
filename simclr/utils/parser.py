from argparse import ArgumentParser


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--save-path",
        help="where to store checkpoints and save train metrics. Defaults to None.",
        dest="save_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--config-path",
        help='from where to load the config. Defaults to "./configs/simclr_config.yaml".',
        dest="config_path",
        type=str,
        default="./configs/simclr_config.yaml",
    )
    return parser


def setup_experiment_parser(parser: ArgumentParser) -> ArgumentParser:
    command_subparser = parser.add_subparsers(dest="command", title="command")
    train = command_subparser.add_parser("train", help="train a SimCLR model")
    train = add_train_args(train)
    return parser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser = setup_experiment_parser(parser)
    return parser
