import pytorch_lightning as pl

from src.callback import SparseMLCallback
from src.data import DummyDataModule
from src.model import LightningModel

if __name__ == "__main__":
    model = LightningModel(22, model_name="efficientnet_v2_s", pretrained=True)

    sparseml = SparseMLCallback("config/recipe.yaml")

    trainer = pl.Trainer(devices=1,
                         accelerator="auto",
                         strategy=None,
                         callbacks=[sparseml],
                         log_every_n_steps=1,
                         benchmark=True,
                         max_epochs=128,
                         precision=16)

    data_module = DummyDataModule(size=16, batch_size=2)
    trainer.fit(model, data_module)
