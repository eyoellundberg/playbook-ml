"""Autoforge CLI argument parsing and dispatch.

Core commands cover the full loop:
bootstrap/new, calibrate/validate, run, export/status, and pack/install/eval/import.

Environment overrides:
  AUTOFORGE_DIRECTOR_MODEL
  AUTOFORGE_LIBRARY_MODEL
  AUTOFORGE_EXTRACT_MODEL
"""

import argparse

from commands.bootstrap import cmd_bootstrap, cmd_new
from commands.run_cmd import cmd_run
from commands.tools import cmd_calibrate, cmd_validate, cmd_export, cmd_status, cmd_pack, cmd_install, cmd_eval, cmd_import, cmd_tail, cmd_train


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Autoforge — autonomous strategy learning system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bootstrap
    p_bootstrap = subparsers.add_parser("bootstrap", help="Generate a new domain from a description using AI")
    p_bootstrap.add_argument("domain", help="Domain name (e.g. GrainMarketing)")
    p_bootstrap.add_argument("--description", required=True, help="Natural language description of the domain")
    p_bootstrap.add_argument("--manual-ai", action="store_true", help="Use file-based manual AI responses instead of Anthropic")

    # new
    p_new = subparsers.add_parser("new", help="Scaffold a new domain from template")
    p_new.add_argument("domain", help="Domain name (e.g. GrainMarketing)")

    # run
    p_run = subparsers.add_parser("run", help="Run the tournament")
    p_run.add_argument("--domain",  required=True, help="Domain subfolder name")
    p_run.add_argument("--batches", type=int, default=8,   help="Batches per year (default 8)")
    p_run.add_argument("--rounds",  type=int, default=200, help="Rounds per batch (default 200)")
    p_run.add_argument("--years",   type=int, default=1,   help="Years to run (default 1)")
    p_run.add_argument("--brain",   action="store_true",   help="Stage 2: AI archetypes")
    p_run.add_argument("--auto",    action="store_true",   help="Stage 1 until saturated, then auto-promote to Stage 2")
    p_run.add_argument("--manual-ai", action="store_true", help="Use file-based manual AI responses instead of Anthropic")
    p_run.add_argument("--workers", type=int, default=1,
        help="Parallel workers for simulation (default 1, use os.cpu_count() for max)")

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Show scenario distributions and score stats")
    p_cal.add_argument("--domain", required=True, help="Domain subfolder name")
    p_cal.add_argument("--n", type=int, default=500, help="Scenarios to sample (default 500)")

    # validate
    p_validate = subparsers.add_parser("validate", help="Sanity-check a domain's simulation.py")
    p_validate.add_argument("--domain", required=True, help="Domain subfolder name")

    # export
    p_export = subparsers.add_parser("export", help="Export Stage 3 training data")
    p_export.add_argument("--domain", required=True, help="Domain subfolder name")

    # status
    p_status = subparsers.add_parser("status", help="Show domain state")
    p_status.add_argument("--domain", required=True, help="Domain subfolder name")

    # pack
    p_pack = subparsers.add_parser("pack", help="Bundle a domain into a shareable .zip")
    p_pack.add_argument("domain", help="Domain name")

    # install
    p_install = subparsers.add_parser("install", help="Install a domain pack from a .zip file")
    p_install.add_argument("pack", help="Path to .zip pack file")
    p_install.add_argument("--force", action="store_true", help="Overwrite if domain already exists")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run champion against eval scenarios")
    p_eval.add_argument("--domain", required=True, help="Domain subfolder name")

    # import
    p_import = subparsers.add_parser("import", help="Import real production decisions into training data")
    p_import.add_argument("--domain", required=True, help="Domain subfolder name")
    p_import.add_argument("--file",   required=True, help="Path to JSONL file of real decisions")

    # tail
    p_tail = subparsers.add_parser("tail", help="Show live snapshot of an in-progress run")
    p_tail.add_argument("--domain", required=True, help="Domain subfolder name")

    # train
    p_train = subparsers.add_parser("train", help="Train a Stage 3 model from exported data")
    p_train.add_argument("--domain", required=True, help="Domain subfolder name")

    args = parser.parse_args()

    if args.command == "bootstrap":
        cmd_bootstrap(args)
    elif args.command == "new":
        cmd_new(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "pack":
        cmd_pack(args)
    elif args.command == "install":
        cmd_install(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "import":
        cmd_import(args)
    elif args.command == "tail":
        cmd_tail(args)
    elif args.command == "train":
        cmd_train(args)
