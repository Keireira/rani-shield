"""Independent face detection/FaceNet measurement, without protection models."""
from pathlib import Path

from asset_paths import weight_path


class FaceEvaluator:
    def __init__(self, root_dir, device=None):
        self.root = Path(root_dir).resolve()
        self.requested_device = device
        self.facenet = self.detector = None

    def load(self):
        from scripts.runtime_environment import configure_environment
        configure_environment(self.root)
        import torch
        from facenet_pytorch import InceptionResnetV1
        from phantomseal_runtime import load_mtcnn
        self.device = torch.device(self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            state = torch.load(weight_path(self.root, "facenet", "facenet_vggface2.pt"),
                               map_location="cpu", weights_only=True)
            self.facenet = InceptionResnetV1(pretrained=None, classify=True,
                                           num_classes=state["logits.weight"].shape[0])
            self.facenet.load_state_dict(state, strict=True)
            self.facenet.classify = False
            self.facenet.eval().requires_grad_(False).to(self.device)
            self.detector = load_mtcnn(self.root, image_size=160, margin=0, keep_all=True, device="cpu")
            return self
        except BaseException:
            self.close()
            raise

    def detect(self, image):
        import numpy as np
        boxes, probabilities, landmarks = self.detector.detect(image, landmarks=True)
        if boxes is None:
            raise ValueError("No face detected.")
        valid = np.where(np.asarray(probabilities) >= .9)[0]
        if not len(valid):
            raise ValueError("Face detection confidence is below 0.9.")
        index = max(valid, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        return boxes[index], landmarks[index], len(boxes)

    def face_embedding(self, image, box):
        import numpy as np
        import torch
        face = self.detector.extract(image, np.asarray([box]), save_path=None)
        with torch.inference_mode():
            return self.facenet(face.to(self.device)).detach()

    def close(self):
        self.facenet = self.detector = None


class SimSwapEvaluator(FaceEvaluator):
    """Only the swapper's identity encoder, never the protection ensemble."""

    def __init__(self, root_dir, device=None):
        super().__init__(root_dir, device)
        self.branches = {}

    def load(self):
        from ensemble_encoders import SimSwapFeatures
        from protector import FaceShieldProtector
        try:
            super().load()
            path = weight_path(self.root, "simswap", "arcface_checkpoint.tar")
            branch = SimSwapFeatures(FaceShieldProtector._load_arcface(path))
            branch.eval().requires_grad_(False).to(self.device)
            branch.weight_path = str(path)
            self.branches["simswap"] = branch
            return self
        except BaseException:
            self.close()
            raise

    def identity(self, face):
        return self.branches["simswap"](face)

    def close(self):
        self.branches = {}
        super().close()
