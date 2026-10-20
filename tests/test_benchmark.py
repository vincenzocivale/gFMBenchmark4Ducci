from dataclasses import replace

import numpy as np
import pytest

from gfm_benchmark.data import Record, read_records, reverse_complement, validate
from gfm_benchmark.encoders import KmerEncoder
from gfm_benchmark.evaluation import evaluate
from gfm_benchmark.runner import extract, fingerprint, run


def test_kmer_frequencies_and_ambiguity():
    encoder = KmerEncoder(k=2, canonical=False)
    values = encoder.encode(["AACNAA", "N", "A"])
    assert values[0, encoder.index["AA"]] == pytest.approx(2 / 3)
    assert values[0, encoder.index["AC"]] == pytest.approx(1 / 3)
    assert np.all(values[1:] == 0)


def test_reverse_complement_invariance():
    encoder = KmerEncoder(k=3)
    sequence = "AACGTTGCCANNN"
    np.testing.assert_array_equal(*encoder.encode([sequence, reverse_complement(sequence)]))


@pytest.mark.parametrize("change", ["sequence", "group", "id"])
def test_leakage_and_duplicate_checks(change):
    first = Record("a", "AAAAC", "train", "0", "g1")
    second = Record("b", "ACCCC", "test", "1", "g2")
    if change == "sequence":
        second = replace(second, sequence=reverse_complement(first.sequence))
    elif change == "group":
        second = replace(second, group=first.group)
    else:
        second = replace(second, id=first.id)
    with pytest.raises(ValueError):
        validate([first, second])


def test_streaming_read_and_normalization(tmp_path):
    path = tmp_path / "input.fa"
    path.write_text(">a description\naacg\nTT\n>b\nNNAC\n")
    assert [r.sequence for r in read_records(path)] == ["AACGTT", "NNAC"]
    assert len(list(read_records(path, limit=1))) == 1
    path = tmp_path / "input.jsonl"
    path.write_text('{"text":"ACGT<eos>"}\n')
    with pytest.raises(ValueError, match="remove tags"):
        list(read_records(path, {"sequence": "text"}))


def test_length_weighted_windowing():
    class LengthEncoder:
        def encode(self, seqs):
            return np.array([[len(s)] for s in seqs])
    result = extract([Record("a", "A" * 10)], LengthEncoder(), window_bp=4, batch_size=2)
    assert result[0, 0] == pytest.approx(3.6)


def test_cache_identity():
    records = [Record("a", "AAAA")]
    options = {"backend": "kmer", "k": 2}
    key = fingerprint(records, options, 128)
    assert key != fingerprint(records, dict(options, k=3), 128)
    assert key != fingerprint([replace(records[0], sequence="AAAC")], options, 128)
    assert key != fingerprint(records, options, 256)


def test_end_to_end_and_cached_run(tmp_path, monkeypatch):
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data/examples/toy.jsonl"
    config = {"dataset": {"path": str(path)}, "output_dir": str(tmp_path),
              "models": {"kmer": {"backend": "kmer", "k": 3}}}
    result = run(config)
    assert result["kmer"]["test"]["accuracy"] >= 0.75
    def fail(*args):
        raise AssertionError("Cache should avoid model loading")
    monkeypatch.setattr("gfm_benchmark.runner.create_encoder", fail)
    assert run(config) == result
    assert len(list(tmp_path.glob("kmer/*/manifest.json"))) == 1


def test_test_labels_do_not_affect_probe_selection():
    from pathlib import Path
    records = list(read_records(Path(__file__).resolve().parents[1] / "data/examples/toy.jsonl"))
    features = KmerEncoder(k=2).encode([r.sequence for r in records])
    first, _ = evaluate(features, records)
    changed = [replace(r, label=str(1 - int(r.label))) if r.split == "test" else r for r in records]
    second, _ = evaluate(features, changed)
    assert first["strength"] == second["strength"]
    assert first["validation_score"] == second["validation_score"]


def test_regression():
    records = [Record(str(i), "A" * (i + 1), split, str(i))
               for i, split in enumerate(["train"] * 6 + ["validation"] * 2 + ["test"] * 2)]
    metrics, pred = evaluate(np.arange(10).reshape(-1, 1), records, task="regression")
    assert metrics["test"]["rmse"] < 0.1
    assert len(pred) == 2
