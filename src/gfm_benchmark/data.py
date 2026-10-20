"""Small, explicit interchange format for public and private datasets."""

import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def normalize(sequence):
    sequence = "".join(sequence.split()).upper()
    if not sequence or set(sequence) - set("ACGTN"):
        raise ValueError("Expected nonempty DNA containing only A,C,G,T,N; remove tags explicitly")
    return sequence


@dataclass(frozen=True)
class Record:
    id: str
    sequence: str
    split: str = ""
    label: str = ""
    group: str = ""


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_records(path, columns=None, limit=None):
    """Stream FASTA(.gz), CSV(.gz), or JSONL(.gz); no automatic split inference."""
    columns = columns or {}
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    suffix = str(path).removesuffix(".gz").lower()
    with open_text(path) as handle:
        if suffix.endswith((".fa", ".fasta", ".fna")):
            rows = _fasta(handle)
        elif suffix.endswith(".csv"):
            rows = csv.DictReader(handle)
        elif suffix.endswith(".jsonl"):
            rows = (json.loads(line) for line in handle if line.strip())
        else:
            raise ValueError("Supported formats: FASTA, CSV, JSONL (optionally gzip)")
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                break
            def field(name):
                value = row.get(columns.get(name, name), "")
                return "" if value is None else str(value)
            yield Record(field("id") or f"record_{i}", normalize(field("sequence")),
                         field("split"), field("label"), field("group"))


def _fasta(handle):
    name, chunks = None, []
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                yield {"id": name, "sequence": "".join(chunks)}
            name, chunks = line[1:].split()[0], []
        else:
            if name is None:
                raise ValueError("FASTA sequence before header")
            chunks.append(line)
    if name is not None:
        yield {"id": name, "sequence": "".join(chunks)}


def validate(records, supervised=False):
    if not records:
        raise ValueError("Empty dataset")
    ids, sequences, groups = set(), {}, {}
    for r in records:
        if r.id in ids:
            raise ValueError(f"Duplicate id: {r.id}")
        ids.add(r.id)
        if supervised and (r.split not in {"train", "validation", "test"} or r.label == ""):
            raise ValueError("Supervised evaluation requires label and train/validation/test split")
        key = min(r.sequence, reverse_complement(r.sequence))
        for mapping, value, description in ((sequences, key, "sequence/reverse complement"),
                                             (groups, r.group, "group")):
            if value and value in mapping and mapping[value] != r.split:
                raise ValueError(f"Cross-split leakage: {description} ({r.id})")
            mapping[value] = r.split
    if supervised and {r.split for r in records} != {"train", "validation", "test"}:
        raise ValueError("All three splits must be nonempty")


def write_records(records, path):
    with Path(path).open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")
