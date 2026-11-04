from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import torch
from transformers import pipeline


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict):
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )


class GroundingDINO:

    def __init__(self, detector_id="IDEA-Research/grounding-dino-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector_id = detector_id
        self.detector = self.load_detector(detector_id)

    def load_detector(self, detector_id):
        detector_id = (
            detector_id
            if detector_id is not None
            else "IDEA-Research/grounding-dino-base"
        )
        object_detector = pipeline(
            model=detector_id, task="zero-shot-object-detection", device=self.device
        )
        return object_detector

    def detect(self, image, labels, threshold=0.3):
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        labels = [label if label.endswith(".") else label + "." for label in labels]
        results = self.detector(image, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]
        return results
