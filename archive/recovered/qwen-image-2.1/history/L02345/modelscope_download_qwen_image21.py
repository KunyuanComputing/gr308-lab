from pathlib import Path
import time
from modelscope.hub.snapshot_download import snapshot_download
base=Path('/work')
t=time.time()
print('modelscope download starting',flush=True)
p=snapshot_download('Qwen/Qwen-Image-2.1', local_dir=str(base/'model'))
print('downloaded',p,'seconds',time.time()-t,flush=True)
