import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from .data import read_records, validate, write_records
from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Genomic foundation model representation benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "embed"):
        p = sub.add_parser(command)
        p.add_argument("config", type=Path)
        p.add_argument("--models", nargs="+")
    p = sub.add_parser("prepare", help="Convert OpenGenome/local data to canonical JSONL")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sequence-column", default="sequence")
    p.add_argument("--id-column", default="id")
    p.add_argument("--metadata", type=Path, help="Canonical CSV/JSONL with id,label,split,group; sequence optional")
    p.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command in {"run", "embed"}:
        config = yaml.safe_load(args.config.read_text())
        run(config, args.config.resolve().parent, args.models, args.command == "embed")
    else:
        records = list(read_records(args.input, {"sequence": args.sequence_column, "id": args.id_column}, args.limit))
        if args.metadata:
            import csv
            import json
            with args.metadata.open() as handle:
                rows = list(csv.DictReader(handle) if args.metadata.suffix == ".csv" else (json.loads(line) for line in handle if line.strip()))
            metadata = {str(row["id"]): row for row in rows}
            if len(metadata) != len(rows):
                raise ValueError("Duplicate ids in metadata")
            if any(r.id not in metadata for r in records):
                raise ValueError("Missing metadata for one or more sequence ids")
            records = [replace(r, label=str(metadata[r.id]["label"]), split=metadata[r.id]["split"],
                               group=str(metadata[r.id].get("group", ""))) for r in records]
        validate(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        write_records(records, args.output)
        print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
