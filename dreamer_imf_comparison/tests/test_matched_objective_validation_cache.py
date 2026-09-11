from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock


from dreamer_imf_compare.artifacts import write_json_atomic
import dreamer_imf_compare.matched_objective_benchmark as benchmark


class MatchedObjectiveValidationCacheTests(unittest.TestCase):
    @staticmethod
    def _context(root: Path):
        cell = {"stage": "dataset", "cell_id": "dataset-cache-test"}
        matrix = {"matrix_sha256": "a" * 64, "cells": [cell]}
        protocol = {"protocol": "cache-test"}
        directory = benchmark.stage_directory(root, cell)
        directory.mkdir(parents=True)
        result = {"matrix_sha256": matrix["matrix_sha256"], "value": 1}
        result_path = directory / "result.json"
        write_json_atomic(result_path, result)
        (directory / "payload.bin").write_bytes(b"AAAA")
        return cell, matrix, protocol, result, result_path, directory

    def test_public_verifiers_have_no_cache_injection_parameter(self) -> None:
        for verifier in (
            benchmark.verify_output_root,
            benchmark.verify_pilot_hpo_output_root,
        ):
            parameters = inspect.signature(verifier).parameters
            self.assertNotIn("_validated_results", parameters)
            self.assertNotIn("_validation_cache", parameters)
            with self.assertRaises(TypeError):
                verifier(Path("/unused"), _validated_results={})

    def test_arbitrary_mapping_and_direct_cache_construction_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell, matrix, protocol, result, result_path, _ = self._context(root)
            with self.assertRaisesRegex(TypeError, "verifier-private"):
                benchmark._completed_result_for_cell(
                    root,
                    matrix,
                    protocol,
                    cell,
                    cache={cell["cell_id"]: (result, result_path)},
                )
            with self.assertRaisesRegex(TypeError, "verifier-private"):
                benchmark._ValidatedResultCache(
                    root, matrix, protocol, _token=object()
                )

            class ForgedCache(benchmark._ValidatedResultCache):
                def contains(self, *args, **kwargs):
                    return True

                def lookup(self, *args, **kwargs):
                    return result, result_path

            with self.assertRaisesRegex(TypeError, "verifier-private"):
                benchmark._completed_result_for_cell(
                    root,
                    matrix,
                    protocol,
                    cell,
                    cache=object.__new__(ForgedCache),
                )

    def test_cache_is_bound_to_root_matrix_and_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            cell, matrix, protocol, result, result_path, _ = self._context(first_root)
            self._context(second_root)
            cache = benchmark._new_validation_cache(first_root, matrix, protocol)
            cache.record(
                first_root,
                matrix,
                protocol,
                cell,
                result,
                result_path,
                validated_fingerprint=benchmark._cell_directory_fingerprint(
                    benchmark.stage_directory(first_root, cell)
                ),
            )
            cache.seal(first_root, matrix, protocol)
            observed, observed_path = cache.lookup(
                first_root, matrix, protocol, cell
            )
            self.assertEqual(observed, result)
            self.assertEqual(observed_path, result_path)

            with self.assertRaisesRegex(ValueError, "another verification context"):
                cache.lookup(second_root, matrix, protocol, cell)
            changed_matrix = {**matrix, "matrix_sha256": "b" * 64}
            with self.assertRaisesRegex(ValueError, "another verification context"):
                cache.lookup(first_root, changed_matrix, protocol, cell)
            changed_protocol = {**protocol, "protocol": "changed"}
            with self.assertRaisesRegex(ValueError, "another verification context"):
                cache.lookup(first_root, matrix, changed_protocol, cell)

    def test_cache_rejects_in_place_context_mutation(self) -> None:
        for context_name in ("matrix", "protocol"):
            with self.subTest(context=context_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cell, matrix, protocol, result, result_path, _ = self._context(root)
                cache = benchmark._new_validation_cache(root, matrix, protocol)
                cache.record(
                    root,
                    matrix,
                    protocol,
                    cell,
                    result,
                    result_path,
                    validated_fingerprint=benchmark._cell_directory_fingerprint(
                        benchmark.stage_directory(root, cell)
                    ),
                )
                cache.seal(root, matrix, protocol)
                if context_name == "matrix":
                    matrix["matrix_sha256"] = "b" * 64
                else:
                    protocol["protocol"] = "changed-in-place"
                with self.assertRaisesRegex(ValueError, "another verification context"):
                    cache.lookup(root, matrix, protocol, cell)

    def test_cache_hit_detects_content_addition_and_removal(self) -> None:
        mutations = {
            "same_size_content_change": lambda directory: (
                directory / "payload.bin"
            ).write_bytes(b"BBBB"),
            "file_addition": lambda directory: (
                directory / "unexpected.bin"
            ).write_bytes(b"extra"),
            "file_removal": lambda directory: (directory / "payload.bin").unlink(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cell, matrix, protocol, result, result_path, directory = self._context(root)
                cache = benchmark._new_validation_cache(root, matrix, protocol)
                cache.record(
                    root,
                    matrix,
                    protocol,
                    cell,
                    result,
                    result_path,
                    validated_fingerprint=benchmark._cell_directory_fingerprint(
                        benchmark.stage_directory(root, cell)
                    ),
                )
                cache.seal(root, matrix, protocol)
                mutate(directory)
                with self.assertRaisesRegex(
                    ValueError, "cell artifacts changed after semantic validation"
                ):
                    cache.lookup(root, matrix, protocol, cell)

    def test_cache_hit_detects_result_json_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell, matrix, protocol, result, result_path, _ = self._context(root)
            cache = benchmark._new_validation_cache(root, matrix, protocol)
            cache.record(
                root,
                matrix,
                protocol,
                cell,
                result,
                result_path,
                validated_fingerprint=benchmark._cell_directory_fingerprint(
                    benchmark.stage_directory(root, cell)
                ),
            )
            cache.seal(root, matrix, protocol)
            write_json_atomic(
                result_path,
                {"matrix_sha256": matrix["matrix_sha256"], "value": 2},
            )
            with self.assertRaisesRegex(
                ValueError, "result changed after validation"
            ):
                cache.lookup(root, matrix, protocol, cell)

    def test_semantic_validator_runs_once_while_each_hit_reauthenticates_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell, matrix, protocol, _, _, directory = self._context(root)
            cache = benchmark._new_validation_cache(root, matrix, protocol)
            with mock.patch.object(benchmark, "validate_dataset_result") as validator:
                first = benchmark._completed_result_for_cell(
                    root, matrix, protocol, cell, cache=cache
                )
                second = benchmark._completed_result_for_cell(
                    root, matrix, protocol, cell, cache=cache
                )
                self.assertEqual(first, second)
                self.assertEqual(validator.call_count, 1)
                (directory / "late-file.bin").write_bytes(b"late")
                with self.assertRaisesRegex(
                    ValueError, "cell artifacts changed after semantic validation"
                ):
                    benchmark._completed_result_for_cell(
                        root, matrix, protocol, cell, cache=cache
                    )
                self.assertEqual(validator.call_count, 1)

    def test_mutation_after_semantic_validation_cannot_become_cache_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cell, matrix, protocol, _, _, directory = self._context(root)
            cache = benchmark._new_validation_cache(root, matrix, protocol)

            def mutate_after_validation(*args, **kwargs):
                del args, kwargs
                (directory / "payload.bin").write_bytes(b"BBBB")

            with mock.patch.object(
                benchmark,
                "validate_dataset_result",
                side_effect=mutate_after_validation,
            ):
                with self.assertRaisesRegex(
                    ValueError, "changed during semantic validation"
                ):
                    benchmark._completed_result_for_cell(
                        root, matrix, protocol, cell, cache=cache
                    )


if __name__ == "__main__":
    unittest.main()
