import logging
from typing import Any, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleExporter

log = logging.getLogger(__name__)


class SparseMLCallback(Callback):
    """Enables SparseML aware training. Requires a recipe to run during training.

    Args:
        recipe_path: Path to a SparseML compatible yaml recipe.
            More information at https://docs.neuralmagic.com/sparseml/source/recipes.html
    """

    def __init__(self, recipe_path: str):
        self.manager = ScheduledModifierManager.from_yaml(recipe_path)

    def on_fit_start(self, trainer: Trainer,
                     pl_module: LightningModule) -> None:
        optimizer = trainer.optimizers

        if len(optimizer) > 1:
            raise MisconfigurationException(
                "SparseML only supports training with one optimizer.")
        optimizer = optimizer[0]
        optimizer = self.manager.modify(
            pl_module,
            optimizer,
            steps_per_epoch=trainer.estimated_stepping_batches,
            epoch=0)
        trainer.optimizers = [optimizer]

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.manager.finalize(pl_module)

    @staticmethod
    def export_to_sparse_onnx(model: LightningModule,
                              output_dir: str,
                              sample_batch: Optional[torch.Tensor] = None,
                              **export_kwargs: Any) -> None:
        """Exports the model to ONNX format."""
        with model._prevent_trainer_and_dataloaders_deepcopy():
            exporter = ModuleExporter(model, output_dir=output_dir)
            sample_batch = sample_batch if sample_batch is not None else model.example_input_array
            if sample_batch is None:
                raise MisconfigurationException(
                    "To export the model, a sample batch must be passed via "
                    "``SparseMLCallback.export_to_sparse_onnx(model, output_dir, sample_batch=sample_batch)`` "
                    "or an ``example_input_array`` property within the LightningModule"
                )
            exporter.export_onnx(sample_batch=sample_batch, **export_kwargs)
