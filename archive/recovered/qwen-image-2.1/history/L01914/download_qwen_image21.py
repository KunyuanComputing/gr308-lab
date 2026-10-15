from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
base=Path('/work')
info=HfApi().model_info('Qwen/Qwen-Image-2.1')
(base/'logs/model-revision.txt').write_text(info.sha+'\n')
print('revision',info.sha,flush=True)
p=snapshot_download('Qwen/Qwen-Image-2.1',revision=info.sha,local_dir=base/'model',max_workers=8)
print('downloaded',p,flush=True)
