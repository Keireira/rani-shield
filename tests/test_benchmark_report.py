"""Meaningful CPU-only reporting checks; no models, Torch, or network."""
import copy
import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SPEC = importlib.util.spec_from_file_location(
    "benchmark_report", Path(__file__).resolve().parents[1] / "benchmark_report.py"
)
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def pair(source, target, clean=.9, ps=.1, final=.2, condition="png"):
    return {"source_id": source, "target_id": target, "condition": condition,
            "rows": {name: {"source_face_detected": True, "swap_face_detected": True,
                            "facenet_cosine_to_clean_source": score}
                     for name, score in (("clean", clean), ("stage1", ps), ("final", final))}}


class ReportingTests(unittest.TestCase):
    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])

    def test_equal_source_weighting_differs_from_pair_weighting(self):
        reports = [pair("A", str(i)) for i in range(3)]
        reports.append(pair("B", "one", clean=.8, ps=.8, final=.2))
        summary = reporting.summarize(reports)
        self.assertEqual(summary["complete_pairs"], 4)
        self.assertAlmostEqual(summary["mean_scores"]["clean"], .875)
        self.assertAlmostEqual(summary["mean_scores"]["stage1"], .275)
        self.assertAlmostEqual(summary["mean_delta_final_minus_stage1"], -.075)
        self.assertAlmostEqual(summary["source_weighted_mean_scores"]["stage1"], .45)
        self.assertAlmostEqual(summary["source_weighted_delta_final_minus_stage1"], -.25)
        self.assertEqual(summary["pair_comparison_counts"], {"chain_better": 1, "ps_better": 3, "ties": 0})
        self.assertEqual(summary["source_comparison_counts"], {"chain_better": 1, "ps_better": 1, "ties": 0})
        self.assertEqual([entry["complete_pairs"] for entry in summary["per_source"]], [3, 1])

    def test_requires_clean_baseline_and_reports_detection_failures(self):
        clean_failed = pair("A", "clean_failed")
        clean_failed["rows"]["clean"].update(source_face_detected=False, swap_face_detected=False)
        output_failed = pair("A", "output_failed")
        output_failed["rows"]["stage1"].update(swap_face_detected=False, facenet_cosine_to_clean_source=None)
        reports = [pair("A", "ok"), clean_failed, output_failed]
        summary = reporting.summarize(reports)
        self.assertEqual(summary["complete_pairs"], 1)
        self.assertEqual(summary["excluded_pairs"], 2)
        self.assertAlmostEqual(summary["mean_scores"]["clean"], .9)
        clean = summary["detection_failures"]["clean"]
        self.assertEqual(clean["source_detection_failed"], 1)
        self.assertEqual(clean["swap_detection_failed"], 0)
        self.assertEqual(clean["swap_not_attempted_after_source_failure"], 1)
        self.assertEqual(summary["detection_failures"]["stage1"]["swap_detection_failed"], 1)
        self.assertEqual(summary["invalid_scores_by_method"]["stage1"], 1)
        self.assertEqual(len(summary["excluded_pair_details"]), 2)

    def test_null_nonfinite_and_invalid_scores_never_enter_means(self):
        reports = []
        for index, value in enumerate((None, float("nan"), float("inf"), -float("inf"), True, "0.7")):
            sample = pair("A", str(index))
            sample["rows"]["final"]["facenet_cosine_to_clean_source"] = value
            reports.append(sample)
        summary = reporting.summarize(reports)
        self.assertEqual(summary["complete_pairs"], 0)
        self.assertEqual(summary["invalid_scores_by_method"]["final"], 6)
        self.assertIsNone(summary["source_weighted_delta_final_minus_stage1"])
        self.assertTrue(all(value is None for value in summary["mean_scores"].values()))

    def test_conditions_are_unique_and_can_reverse_direction(self):
        png = pair("A", "target", ps=.5, final=.2)
        jpeg = pair("A", "target", ps=.2, final=.6, condition="jpeg95")
        summary = reporting.summarize([png, jpeg])
        self.assertAlmostEqual(summary["by_condition"]["png"]["source_weighted_delta_final_minus_stage1"], -.3)
        self.assertAlmostEqual(summary["by_condition"]["jpeg95"]["source_weighted_delta_final_minus_stage1"], .4)
        self.assertEqual(summary["complete_pairs"], 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            reporting.summarize([png, copy.deepcopy(png)])
        old = pair("B", "old")
        old.pop("condition")
        self.assertIn("png", reporting.summarize([old])["by_condition"])

    def test_baseline_sanity_is_separate_from_primary_quality_gate(self):
        failure = pair("A", "failed", clean=.2)
        failure["rows"]["clean"]["facenet_cosine_to_target"] = .6
        success = pair("A", "passed", clean=.9)
        success["rows"]["clean"]["facenet_cosine_to_target"] = .3
        summary = reporting.summarize([failure, success, pair("A", "unknown")])
        self.assertEqual(summary["complete_pairs"], 3)
        self.assertEqual(summary["qualified_pairs"], 2)
        self.assertEqual(summary["weak_baseline_pairs"], 1)
        self.assertEqual(summary["baseline_sanity_checks"], {"available_pairs": 2, "passed": 1, "failed": 1, "unavailable": 1})

    def test_weak_baseline_never_enters_primary_means_or_comparison_counts_and_raw_is_retained(self):
        weak = pair("weak", "target", clean=.68, ps=.3, final=.9)
        qualified = pair("qualified", "target", clean=.85, ps=.5, final=.2)
        before = copy.deepcopy([weak, qualified])
        summary = reporting.summarize([weak, qualified])
        self.assertEqual(summary["total_pairs"], 2)
        self.assertEqual(summary["complete_pairs"], 2)
        self.assertEqual(summary["qualified_pairs"], 1)
        self.assertEqual(summary["weak_baseline_pairs"], 1)
        self.assertEqual(summary["incomplete_pairs"], 0)
        self.assertAlmostEqual(summary["mean_scores"]["clean"], .85)
        self.assertAlmostEqual(summary["source_weighted_delta_final_minus_stage1"], -.3)
        self.assertEqual(summary["pair_comparison_counts"], {"chain_better": 1, "ps_better": 0, "ties": 0})
        self.assertEqual(summary["sources_with_qualified_pairs"], 1)
        self.assertEqual(summary["raw"]["complete_pairs"], 2)
        self.assertAlmostEqual(summary["raw"]["mean_scores"]["clean"], (.68 + .85) / 2)
        self.assertEqual(summary["raw"]["pair_comparison_counts"], {"chain_better": 1, "ps_better": 1, "ties": 0})
        self.assertTrue(summary["manual_baseline_inspection_required"])
        self.assertEqual(summary["excluded_pair_details"][0]["reasons"]["baseline"], ["weak_clean_baseline"])
        self.assertEqual([weak, qualified], before)

    def test_one_processing_condition_qualifies_and_the_other_is_inconclusive(self):
        png = pair("A", "target", clean=.8, ps=.5, final=.2)
        jpeg = pair("A", "target", clean=.68, ps=.2, final=.6, condition="jpeg95")
        summary = reporting.summarize([png, jpeg])
        self.assertEqual(summary["by_condition"]["png"]["qualified_pairs"], 1)
        self.assertEqual(summary["by_condition"]["jpeg95"]["qualified_pairs"], 0)
        self.assertEqual(summary["by_condition"]["jpeg95"]["weak_baseline_pairs"], 1)
        self.assertIsNone(summary["by_condition"]["jpeg95"]["source_weighted_delta_final_minus_stage1"])
        self.assertAlmostEqual(summary["by_condition"]["jpeg95"]["raw"]["source_weighted_delta_final_minus_stage1"], .4)
        self.assertAlmostEqual(summary["source_weighted_delta_final_minus_stage1"], -.3)
        self.assertAlmostEqual(summary["raw"]["source_weighted_delta_final_minus_stage1"], .05)
        self.assertEqual(summary["per_source"][0]["complete_pairs"], 2)
        self.assertEqual(summary["per_source"][0]["qualified_pairs"], 1)

    def test_configurable_cutoff_is_inclusive_finite_and_in_cosine_range(self):
        sample = pair("A", "target", clean=.8)
        self.assertEqual(reporting.summarize([sample], minimum_baseline=.8)["qualified_pairs"], 1)
        self.assertEqual(reporting.summarize([sample], minimum_baseline=.81)["qualified_pairs"], 0)
        for cutoff in (True, False, None, "0.8", float("nan"), float("inf"), -float("inf"), -.1, -1.1, 1.1):
            with self.subTest(cutoff=cutoff):
                with self.assertRaisesRegex(ValueError, "minimum_baseline"):
                    reporting.summarize([sample], minimum_baseline=cutoff)
        with self.temporary_directory() as directory:
            with self.assertRaisesRegex(ValueError, "minimum_baseline"):
                reporting.write_report(Path(directory) / "report", [sample], minimum_baseline=float("nan"))

    def test_weak_baseline_rendered_as_inconclusive_without_effect_direction_and_all_scores_preserved(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            image = root / "reference.png"
            Image.new("RGB", (40, 24), "#7799bb").save(image)
            weak = pair("weak", "target", clean=.68, ps=.3, final=.9)
            weak["target"] = {"path": str(image)}
            for row in weak["rows"].values():
                row.update(source_path=str(image), output_path=str(image))
            result = reporting.write_report(root / "report", [weak],
                {"correction_note": "The earlier unfiltered count does not establish defense efficacy."}, root=root)
            page = Path(result["html_path"]).read_text(encoding="utf-8")
            self.assertIn("Inconclusive — weak baseline", page)
            self.assertIn("manual baseline inspection required", page)
            self.assertIn("not a validated universal FaceNet threshold", page)
            self.assertIn("0.6800", page)
            self.assertIn("Raw unfiltered numerical statistics", page)
            self.assertIn("earlier unfiltered count", page)
            # The weak pair's table contains no directional label or primary delta.
            table = page.split("<h2>All source / target pairs</h2>", 1)[1].split("</table>", 1)[0]
            self.assertNotIn("ps_better", table)
            self.assertNotIn("chain_better", table)
            self.assertNotIn("0.6000", table)
            self.assertIn('src="assets/', table)
            with Path(result["csv_path"]).open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["direction"], "inconclusive_weak_baseline")
            self.assertEqual(row["final_minus_stage1"], "")
            self.assertAlmostEqual(float(row["raw_final_minus_stage1"]), .6)
            self.assertEqual(float(row["clean_cosine"]), .68)
            document = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(document["pair_reports"][0], weak)
            self.assertEqual(document["summary"]["raw"]["complete_pairs"], 1)
            self.assertIsNone(document["summary"]["source_weighted_delta_final_minus_stage1"])
            self.assertEqual(document["summary"]["pair_comparison_counts"]["ps_better"], 0)

    def test_ties_and_representative_selection_do_not_depend_on_winner(self):
        reports = [pair(str(i), "target", ps=.2, final=.2 + (5e-7 if i % 2 else -5e-7)) for i in range(12)]
        self.assertEqual(reporting.summarize(reports)["pair_comparison_counts"]["ties"], 12)
        analysed = reporting._analyses(reports)
        selected = reporting._representatives(analysed, 6)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 9)  # lexicographic source IDs: 0,1,10,11,2,...,9
        reversed_winners = copy.deepcopy(reports)
        for sample in reversed_winners:
            sample["rows"]["final"]["facenet_cosine_to_clean_source"] = .9
        self.assertEqual(selected, reporting._representatives(reporting._analyses(reversed_winners), 6))

    def test_writes_portable_outputs_and_json_without_nan(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            image = root / "reference.png"
            Image.new("RGB", (40, 24), "#7799bb").save(image)
            valid = pair('<script>alert(1)</script>', "target", condition="jpeg95")
            valid["target"] = {"path": "reference.png"}
            for row in valid["rows"].values():
                row.update(source_path="reference.png", output_path="reference.png")
            invalid = pair("B", "target")
            invalid["rows"]["final"]["facenet_cosine_to_clean_source"] = float("nan")
            metadata = {"nonfinite": float("inf"), "protected_role": "source_identity",
                        "reuses_exploratory_source": True, "exploratory_source_ids": ["public_portrait"],
                        "image_quality": [{"source_id": "public_portrait",
                                           "stage1": {"ssim": .98, "psnr_db": 39, "mae": 1.2, "max_pixel_change": 19},
                                           "final": {"ssim": .94, "psnr_db": 34, "mae": 3.1, "max_pixel_change": 31}}]}
            result = reporting.write_report(root / "report", [valid, invalid], metadata, root=root)
            document = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"),
                                  parse_constant=lambda value: self.fail(f"Nonstandard JSON {value}"))
            self.assertIsNone(document["metadata"]["nonfinite"])
            self.assertIsNone(document["pair_reports"][1]["rows"]["final"]["facenet_cosine_to_clean_source"])
            self.assertIsInstance(document["summary"]["complete_pairs"], int)
            page = Path(result["html_path"]).read_text(encoding="utf-8")
            self.assertIn("Separate processing conditions", page)
            self.assertIn("Clean swap", page)
            self.assertIn("jpeg95", page)
            self.assertIn("Protected role: source identity", page)
            self.assertIn("not a matched-budget ablation", page)
            self.assertIn("not a strictly held-out source evaluation", page)
            self.assertIn("Protected-image quality at current product settings", page)
            self.assertIn("31.0000", page)
            self.assertIn("&lt;script&gt;", page)
            self.assertNotIn("<script>alert(1)</script>", page)
            self.assertIn('src="assets/', page)
            self.assertNotIn('src="http', page)
            with Path(result["csv_path"]).open(encoding="utf-8-sig", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(csv_rows[0]["condition"], "jpeg95")
            with Image.open(result["contact_sheet_path"]) as sheet:
                self.assertGreater(sheet.width, 1000)
                self.assertGreater(sheet.height, 224)
            self.assertTrue(math.isnan(invalid["rows"]["final"]["facenet_cosine_to_clean_source"]))

    def test_empty_input_has_no_fabricated_means(self):
        summary = reporting.summarize([])
        self.assertEqual(summary["total_pairs"], 0)
        self.assertEqual(summary["by_condition"], {})
        self.assertIsNone(summary["source_weighted_delta_final_minus_stage1"])


if __name__ == "__main__":
    unittest.main()
