"""Fingerprint local dense checkpoints without copying model weights to Git."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json
HERE=Path(__file__).resolve().parent
paths=['/home/chokwans99/models_mlsys/layerskip-llama2-7b','/home/chokwans99/models_mlsys/AMD-Llama-135m','/data/chokwans99/models/layerskip-llama3-8B','/home/chokwans99/models_mlsys/Qwama-0.5B-Instruct']
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while data:=f.read(4*2**20):h.update(data)
 return dict(file=p.name,bytes=p.stat().st_size,sha256=h.hexdigest())
models=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for name in paths:
  path=Path(name);config=json.loads((path/'config.json').read_text())
  files=sorted([p for p in path.iterdir() if p.is_file() and (p.suffix in ['.safetensors','.json','.model'] or p.name=='merges.txt')])
  models.append(dict(path=name,resolved_path=str(path.resolve()),
      config={k:config.get(k) for k in ['architectures','num_hidden_layers','hidden_size','vocab_size','torch_dtype','quantization_config']},
      files=list(pool.map(digest,files))))
  print('Fingerprinted',path.name,flush=True)
(HERE/'MODEL_MANIFEST.json').write_text(json.dumps(models,indent=2))
