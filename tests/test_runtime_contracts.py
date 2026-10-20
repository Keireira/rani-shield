"""Model-free recipe validation covers external API inputs before any work starts."""
import copy
import unittest

from runtime_contracts import DEFAULTS, STAGE_DEFAULTS, pipeline_stages, protection_parameters


class PipelineRecipeContractTests(unittest.TestCase):
    def test_legacy_parameters_keep_their_flat_shape(self):
        self.assertEqual(protection_parameters({}), DEFAULTS)
        for mode in ("nullswap", "phantomseal", "faceshield", "phantomseal_faceshield"):
            result = protection_parameters({"mode": mode})
            self.assertEqual(set(result), set(DEFAULTS))
            for stages in (None, [], [{"type": "nullswap"}]):
                with self.subTest(mode=mode, stages=stages), self.assertRaisesRegex(ValueError, "only.*custom"):
                    protection_parameters({"mode": mode, "stages": stages})

    def test_stage_defaults_are_independent_from_legacy_flat_values_and_input(self):
        recipe = [{"type": "faceshield"}, {"type": "nullswap"}, {"type": "phantomseal"}, {"type": "nullswap"}]
        before = copy.deepcopy(recipe)
        result = protection_parameters({"mode": "custom", "stages": recipe, "iterations": 100,
                                        "nullswap_strength": 3, "phantom_iterations": 20})
        self.assertEqual(recipe, before)
        self.assertEqual([stage["params"] for stage in result["stages"]],
                         [STAGE_DEFAULTS[stage["type"]] for stage in recipe])
        result["stages"][1]["params"]["seed"] = 5
        self.assertEqual(result["stages"][3]["params"]["seed"], 0)
        self.assertEqual(STAGE_DEFAULTS["nullswap"]["seed"], 0)

    def test_all_stage_parameter_boundaries(self):
        boundaries = {
            "nullswap": {"strength": (0, 4), "seed": (0, 4294967295)},
            "phantomseal": {"iterations": (10, 2000)},
            "faceshield": {"iterations": (1, 200), "noise_clamp": (1, 32),
                           "step_size": (.25, 4), "analysis_size": (64, 1280)},
        }
        for kind, fields in boundaries.items():
            for field, (low, high) in fields.items():
                for value in (low, high):
                    with self.subTest(kind=kind, field=field, value=value):
                        result = pipeline_stages([{"type": kind, "params": {field: value}}])
                        self.assertEqual(result[0]["params"][field], value)
                for value in (low - 1, high + 1, True, str(low), None, float("inf"), float("nan"), 10**1000):
                    with self.subTest(kind=kind, field=field, value=value), self.assertRaises(ValueError):
                        pipeline_stages([{"type": kind, "params": {field: value}}])
        self.assertEqual(len(pipeline_stages([{"type": "nullswap"}] * 8)), 8)

    def test_integer_parameters_and_analysis_multiples(self):
        for kind, field, value in (("nullswap", "seed", .5), ("phantomseal", "iterations", 10.5),
                                   ("faceshield", "iterations", 1.5), ("faceshield", "noise_clamp", 2.5),
                                   ("faceshield", "analysis_size", 512.5), ("faceshield", "analysis_size", 65)):
            with self.subTest(kind=kind, field=field), self.assertRaises(ValueError):
                pipeline_stages([{"type": kind, "params": {field: value}}])

    def test_recipe_shape_and_stage_fields_are_strict(self):
        invalid = [None, {}, (), "nullswap", [], [{"type": "nullswap"}] * 9,
                   [None], [{}], [{"type": 1}], [{"type": []}], [{"type": "custom"}],
                   [{"type": "phantomseal_faceshield"}], [{"type": "nullswap", "params": []}],
                   [{"type": "nullswap", "parameters": {}}], [{"type": "nullswap", "enabled": True}],
                   [{"type": "nullswap", "params": {"nullswap_seed": 1}}],
                   [{"type": "phantomseal", "params": {"phantom_iterations": 10}}],
                   [{"type": "faceshield", "params": {"max_size": 1280}}]]
        for recipe in invalid:
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                protection_parameters({"mode": "custom", "stages": recipe})
        with self.assertRaises(ValueError):
            protection_parameters({"mode": "custom"})


if __name__ == "__main__":
    unittest.main()
