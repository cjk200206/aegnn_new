import aegnn.models.layer
import aegnn.models.networks
from aegnn.models.detection import DetectionModel
from aegnn.models.recognition import RecognitionModel
from aegnn.models.corner import CornerModel
from aegnn.models.corner_superpoint import CornerSuperpointModel
from aegnn.models.corner_heatmap import CornerHeatMapModel

################################################################################################
# Access functions #############################################################################
################################################################################################
import pytorch_lightning as pl


def by_task(task: str) -> pl.LightningModule.__class__:
    if task == "detection":
        return DetectionModel
    elif task == "recognition":
        return RecognitionModel
    elif task == "corner":
        return CornerModel
    elif task == "corner_superpoint":
        return CornerSuperpointModel
    elif task == "corner_heatmap":
        return CornerHeatMapModel
    else:
        raise NotImplementedError(f"Task {task} is not implemented!")
