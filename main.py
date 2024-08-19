from argparse import ArgumentParser


def cli() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "option",
        type=str,
        choices=[
            "MoCVAE",
            "preprocess"
        ],
        help="option",
    )
    return parser


def main():
    parser = cli()
    args, _ = parser.parse_known_args()

    if args.option == "MoCVAE":
        from runs.run_MoCVAE import MoCVAE_cli, MoCVAE_main

        args = MoCVAE_cli(parser).parse_args()
        MoCVAE_main(args)


    elif args.option == "preprocess":
        from runs.run_preprocess import preprocess_cli, preprocess_main

        args = preprocess_cli(parser).parse_args()
        preprocess_main(args)


if __name__ == "__main__":
    main()
