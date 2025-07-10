# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="default")      # loads default.yaml

trainer = Trainer(**cfg)
trainer.run()


