"""CPU-only checks of the upstream ensemble objective and feature preprocessing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import weakref

import torch
from torch import nn
import torch.nn.functional as F

from ensemble_encoders import (FaceShifterFeatures, HifiFace3DFeatures,
                               HifiFaceIDFeatures, SimSwapFeatures)
from phantomseal_assets import ensemble_readiness
from phantomseal_runtime import PhantomSealProtector, ensemble_loss, sequential_objective


# Deliberately independent of the production table: these are unify.yaml values.
TERMS = {"simswap": (10000., 10000., .003),
         "hififace_3d": (500., 5000., .03),
         "hififace_id": (150., 15000., .0003),
         "faceshifter": (35000., 1000000., .003)}


class ToyBranch(nn.Module):
    def __init__(self, scale, name=None, events=None):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(scale), requires_grad=False)
        self.name, self.events = name, events

    def forward(self, image):
        means = image.mean((-2, -1))
        result = torch.stack((means[:, 0] * self.weight + means[:, 1].square(),
                              means[:, 2] - means[:, 0] * .3), dim=1)
        if self.events is not None and result.requires_grad:
            self.events.append((self.name, "forward"))
            result.register_hook(lambda grad: self.events.append((self.name, "backward")))
        return result


class EnsembleObjectiveTests(unittest.TestCase):
    def test_failed_model_load_releases_partial_ensemble(self):
        runtime = PhantomSealProtector(Path(__file__).resolve().parents[1], device="cpu")
        references = []
        def fail_load(progress):
            runtime.facenet = nn.Linear(2, 2)
            runtime.branches = {"simswap": nn.Linear(2, 2)}
            references.extend(weakref.ref(model) for model in (runtime.facenet, runtime.branches["simswap"]))
            raise RuntimeError("Incomplete local checkpoint")
        with patch.object(runtime, "_load_models", side_effect=fail_load):
            with self.assertRaisesRegex(RuntimeError, "Incomplete local checkpoint"):
                runtime.load_models()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(runtime.branches, {})
        self.assertIsNone(runtime.facenet)
        self.assertFalse(runtime._loaded)

    def test_exact_weights_caps_and_gradient_above_below_cap(self):
        for name, (self_weight, cloak_weight, cap) in TERMS.items():
            for saturated in (False, True):
                with self.subTest(branch=name, saturated=saturated):
                    value = (cap * (4 if saturated else .25)) ** .5
                    feature = torch.full((2, 3, 2), value, requires_grad=True)
                    own = torch.zeros_like(feature)
                    cloak = torch.full_like(feature, -.1)
                    loss = ensemble_loss(name, feature, own, cloak)
                    expected = -self_weight * min(value * value, cap) + cloak_weight * (value + .1) ** 2
                    self.assertAlmostEqual(float(loss), expected, delta=max(1e-4, abs(expected) * 2e-6))
                    gradient, = torch.autograd.grad(loss, feature)
                    expected_grad = 2 * cloak_weight * (value + .1)
                    if not saturated:
                        expected_grad -= 2 * self_weight * value
                    torch.testing.assert_close(gradient, torch.full_like(feature, expected_grad / feature.numel()),
                                               rtol=2e-6, atol=2e-4)

    def fixture(self, events=None):
        original = torch.linspace(.15, .85, 3 * 8 * 9).reshape(1, 3, 8, 9)
        delta = (.012 * torch.sin(torch.arange(original.numel()))).reshape_as(original).requires_grad_()
        # Nonlinear view and resampling ensure each branch backpropagates through the actual view.
        def view(image):
            return F.interpolate(image, size=(5, 6), mode="bilinear", align_corners=False).square()
        branches = {name: ToyBranch(.4 + i * .3, name, events) for i, name in enumerate(TERMS)}
        references = {"_clean_face": view(original).detach()}
        for name, branch in branches.items():
            with torch.no_grad():
                references[name] = (branch(view(original)), branch(view(1 - original * .6)))
        return delta, original, view, branches, references

    def test_sequential_matches_one_joint_gradient_and_does_not_accumulate_grad(self):
        events = []
        delta, original, view, branches, references = self.fixture(events)
        total, gradient = sequential_objective(delta, original, view, branches, references)
        self.assertEqual(events, [(name, action) for name in TERMS for action in ("forward", "backward")])
        face = view((original + delta).clamp(0, 1))
        joint = 1000 * (face - references["_clean_face"]).square().mean()
        for name, (self_weight, cloak_weight, cap) in TERMS.items():
            feature = branches[name](face)
            self_ref, cloak_ref = references[name]
            own = (feature - self_ref).square().mean(1).clamp(max=cap)
            cloak = (feature - cloak_ref).square().mean(1)
            joint = joint + (-self_weight * own + cloak_weight * cloak).mean()
        expected_gradient, = torch.autograd.grad(joint, delta)
        self.assertAlmostEqual(total, float(joint.detach()), delta=max(.01, abs(total) * 2e-6))
        torch.testing.assert_close(gradient, expected_gradient, rtol=3e-6, atol=3e-4)
        self.assertFalse(gradient.requires_grad)
        self.assertTrue(torch.any(gradient.abs() > 1), "Accumulator must return raw gradients before one sign update.")
        self.assertIsNone(delta.grad)
        self.assertTrue(all(parameter.grad is None for branch in branches.values() for parameter in branch.parameters()))

    def test_partial_ensemble_rejected(self):
        delta, original, view, branches, references = self.fixture()
        del branches["hififace_3d"]
        with self.assertRaisesRegex(ValueError, "all four"):
            sequential_objective(delta, original, view, branches, references)

    def test_nonfinite_backward_rejected_even_with_finite_features(self):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, image):
                return image.mean((-2, -1))

            @staticmethod
            def backward(ctx, gradient):
                return torch.full((1, 3, 5, 6), float("nan"))

        delta, original, view, branches, references = self.fixture()
        branches["simswap"] = BadGradient.apply
        references["simswap"] = (torch.zeros(1, 3), torch.ones(1, 3))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            sequential_objective(delta, original, view, branches, references)


class RecordingBackbone(nn.Module):
    def forward(self, image):
        self.last_input = image.detach().clone()
        means = image.mean((-2, -1))
        self.features = torch.cat((means + 1, means.square() + .3), dim=1)
        return self.features


class EncoderPreprocessingTests(unittest.TestCase):
    def image(self, size=256):
        return torch.linspace(.03, .97, 3 * size * size).reshape(1, 3, size, size).requires_grad_()

    def check_wrapper(self, wrapper_type, expected_input, normalize, size=256):
        image = self.image(size)
        backbone = RecordingBackbone()
        output = wrapper_type(backbone)(image)
        torch.testing.assert_close(backbone.last_input, expected_input(image.detach()), rtol=0, atol=0)
        expected = F.normalize(backbone.features, dim=1) if normalize else backbone.features
        torch.testing.assert_close(output, expected)
        output[:, 0].sum().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0)

    def test_simswap_imagenet_nearest_112_then_l2(self):
        def expected(image):
            mean = torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1)
            std = torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1)
            return F.interpolate((image - mean) / std, (112, 112), mode="nearest")
        self.check_wrapper(SimSwapFeatures, expected, True)

    def test_faceshifter_raw_256_crop_then_aligned_112(self):
        def expected(image):
            image = F.interpolate(image, (256, 256), mode="bilinear", align_corners=False)
            return F.interpolate(image[:, :, 19:237, 19:237], (112, 112), mode="bilinear", align_corners=True)
        self.check_wrapper(FaceShifterFeatures, expected, False, size=192)

    def test_hififace_identity_minus_one_to_one_then_bilinear_112_l2(self):
        self.check_wrapper(HifiFaceIDFeatures,
                           lambda image: F.interpolate((image - .5) / .5, (112, 112),
                                                       mode="bilinear", align_corners=False), True)

    def test_hififace_3d_uses_raw_first_80_coefficients_without_normalizing(self):
        class CoefficientBackbone(nn.Module):
            def forward(self, image):
                self.last_input = image.detach().clone()
                return image.mean((1, 2, 3)).unsqueeze(1) + torch.arange(100, device=image.device)
        image = self.image()
        backbone = CoefficientBackbone()
        result = HifiFace3DFeatures(backbone)(image)
        torch.testing.assert_close(backbone.last_input, image.detach(), rtol=0, atol=0)
        torch.testing.assert_close(result, image.mean((1, 2, 3)).unsqueeze(1) + torch.arange(80))
        self.assertEqual(tuple(result.shape), (1, 80))
        result.sum().backward()
        self.assertGreater(float(image.grad.abs().sum()), 0)


class FullAssetReadinessTests(unittest.TestCase):
    def test_actual_vendor_paths_and_all_models_full_bank_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = ["weights/simswap/arcface_checkpoint.tar", "weights/facenet/facenet_vggface2.pt",
                     "weights/hififace/hififace_opensource_299999.ckpt",
                     "weights/faceshifter/model_ir_se50.pth", "weights/mtcnn/pnet.pt",
                     "weights/mtcnn/rnet.pt", "weights/mtcnn/onet.pt",
                     "ensemble_encoders.py", "third_party/phantomseal/full/faceshifter.py",
                     "third_party/phantomseal/full/iresnet.py", "third_party/phantomseal/full/deep3d.py"]
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            bank = root / "weights/phantomseal/cloaks/mix_full"
            bank.mkdir(parents=True)
            for i in range(214):
                (bank / f"{i:03}.jpg").touch()
            self.assertTrue(ensemble_readiness(root)["ready"], ensemble_readiness(root)["message"])
            for relative in paths:
                with self.subTest(missing=relative):
                    path = root / relative
                    path.unlink()
                    self.assertFalse(ensemble_readiness(root)["ready"])
                    path.touch()
            legacy = root / "models/phantomseal/arcface_checkpoint.tar"
            legacy.parent.mkdir(parents=True)
            legacy.touch()
            (root / "weights/simswap/arcface_checkpoint.tar").unlink()
            self.assertFalse(ensemble_readiness(root)["ready"], "Old model paths must not rescue missing central weights.")
            (root / "weights/simswap/arcface_checkpoint.tar").touch()
            (bank / "213.jpg").unlink()
            self.assertFalse(ensemble_readiness(root)["ready"])
            # Old starter assets must never rescue an incomplete original bank.
            starter = root / "data/phantomseal_cloaks"
            starter.mkdir(parents=True)
            for i in range(24):
                (starter / f"{i}.jpg").touch()
            self.assertFalse(ensemble_readiness(root)["ready"])


if __name__ == "__main__":
    unittest.main()
