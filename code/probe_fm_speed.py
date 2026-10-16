import time

import torch
from chronos import BaseChronosPipeline

for mid in ['amazon/chronos-bolt-small']:
    pipe = BaseChronosPipeline.from_pretrained(mid, device_map='cuda', torch_dtype=torch.bfloat16)
    print(type(pipe), pipe.model.device if hasattr(pipe, 'model') else '')
    x = torch.randn(4096, 512)
    for H in (64, 96):
        for bs in (256, 1024, 4096):
            torch.cuda.synchronize(); t0 = time.time()
            q, m = pipe.predict_quantiles(x[:bs], prediction_length=H, quantile_levels=[0.5])
            torch.cuda.synchronize()
            print(mid, 'H', H, 'bs', bs, '%.2fs' % (time.time() - t0), q.shape, q.device)
