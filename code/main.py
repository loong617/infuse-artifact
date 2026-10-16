# Copyright (c) 2025-present, Royal Bank of Canada.
# Copyright (c) 2025-present, Kim et al.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

import os

from models.build import build_model, load_best_model, build_norm_module
from utils.parser import parse_args, load_config
from datasets.build import update_cfg_from_dataset
from trainer import build_trainer
from predictor import Predictor
from utils.misc import set_seeds, set_devices
from tta.tafas import build_adapter
import tta.petsa as petsa
from config import get_norm_module_cfg


def main():
    args = parse_args()
    cfg = load_config(args)
    update_cfg_from_dataset(cfg, cfg.DATA.NAME)
    
    cfg.RESULT_DIR = os.path.join(cfg.RESULT_DIR, cfg.TRAIN.CHECKPOINT_DIR.split('./checkpoints/')[-1])
    
    if not os.path.exists(cfg.RESULT_DIR):
        os.makedirs(cfg.RESULT_DIR)


    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)


    with open(os.path.join(cfg.RESULT_DIR, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    
    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)
    norm_module = build_norm_module(cfg) if cfg.NORM_MODULE.ENABLE else None

    if cfg.TRAIN.ENABLE:
        # build trainer
        trainer = build_trainer(cfg, model, norm_module=norm_module)
        trainer.train()
        
    if cfg.TTA.ENABLE or cfg.TEST.ENABLE:
        model = load_best_model(cfg, model)
        if cfg.NORM_MODULE.ENABLE:
            norm_module = load_best_model(get_norm_module_cfg(cfg), norm_module)
    if cfg.TTA.ENABLE:


        if 'TAFAS' in cfg.RESULT_DIR:
            print("TAFAS")
            adapter = build_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            adapter.count_parameters()
        elif 'PETSA' in cfg.RESULT_DIR:
            print("PETSA")
            adapter = petsa.build_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            adapter.count_parameters()
        elif 'DAdapter' in cfg.RESULT_DIR:
            print("delta-Adapter (ported)")
            from tta.dadapter import build_dadapter
            adapter = build_dadapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'RecycleHybrid' in cfg.RESULT_DIR:
            print("RecycleHybrid")
            os.environ.setdefault('RECYCLE_HYBRID', '1')
            from tta.recycle import build_recycle_adapter
            adapter = build_recycle_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'Online' in cfg.RESULT_DIR:
            print("Online (gradient)")
            from tta.baselines import build_online_adapter
            adapter = build_online_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'LoRA' in cfg.RESULT_DIR:
            print("LoRA-TTA")
            from tta.baselines import build_lora_adapter
            adapter = build_lora_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'RevIN' in cfg.RESULT_DIR:
            print("RevIN-TTA")
            from tta.more_baselines import build_revin_adapter
            adapter = build_revin_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'SOLID' in cfg.RESULT_DIR:
            print("SOLID")
            from tta.more_baselines import build_solid_adapter
            adapter = build_solid_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'ProceedTTA' in cfg.RESULT_DIR:
            print("Proceed")
            from tta.more_baselines import build_proceed_adapter
            adapter = build_proceed_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'SANTTA' in cfg.RESULT_DIR:
            print("SAN-TTA")
            from tta.norm_baselines import build_san_adapter
            adapter = build_san_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'FANTTA' in cfg.RESULT_DIR:
            print("FAN")
            from tta.norm_baselines import build_fan_adapter
            adapter = build_fan_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'COSAOfficial' in cfg.RESULT_DIR:
            print("COSA (ICLR 2026, official code)")
            from tta.cosa_official import build_adapter as build_cosa_official
            adapter = build_cosa_official(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'COSA' in cfg.RESULT_DIR:
            print("COSA (ICLR 2026)")
            from tta.iclr26 import build_cosa_adapter as build_cosa
            adapter = build_cosa(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'DAdapterOfficial' in cfg.RESULT_DIR:
            print("δ-Adapter (ICLR 2026, official)")
            from tta.dadapter_official import build_dadapter_official
            adapter = build_dadapter_official(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'ADAPTZ' in cfg.RESULT_DIR:
            print("ADAPT-Z (ICLR 2026)")
            from tta.iclr26 import build_adaptz_adapter as build_adaptz
            adapter = build_adaptz(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'ADAPTZOfficial' in cfg.RESULT_DIR:
            print("ADAPT-Z (ICLR 2026, official)")
            from tta.fixed_baselines import build_fixed_adaptz
            adapter = build_adaptz_official(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'FixedSAN' in cfg.RESULT_DIR:
            print("SAN-TTA (fixed)")
            from tta.fixed_baselines import build_fixed_san
            adapter = build_fixed_san(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'FixedFAN' in cfg.RESULT_DIR:
            print("FAN (fixed)")
            from tta.fixed_baselines import build_fixed_fan
            adapter = build_fixed_fan(cfg, model, norm_module=norm_module)
            adapter.adapt()
        elif 'Recycle' in cfg.RESULT_DIR:
            print("Recycle")
            from tta.recycle import build_recycle_adapter
            adapter = build_recycle_adapter(cfg, model, norm_module=norm_module)
            adapter.adapt()
            
            
    if cfg.TEST.ENABLE:
        predictor = Predictor(cfg, model, norm_module=norm_module)
        predictor.predict()


if __name__ == '__main__':
    main()
