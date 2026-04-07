# Face Recognition Embedding Models for Identity Fingerprinting (2025–2026)

**Context:** Python 3.12, PyTorch 2.11, CUDA 13. Goal: produce stable 512-d identity embeddings from pre-cropped 112×112 faces (no real-time detection required).

---

## Model Comparison Table

| Model | pip install | Emb dim | LFW acc | IJB-C (TAR@FAR=1e-4) | C compilation? | HF hub path |
|---|---|---|---|---|---|---|
| **ArcFace R100** (insightface buffalo_l) | `pip install insightface onnxruntime-gpu` | 512 | 99.83% | 97.25% (E4) | No — pure ONNX | indirect (auto-download) |
| **ArcFace R100 ONNX** (standalone) | `pip install onnxruntime numpy opencv-python` | 512 | ~99.82% | ~96% | No | `onnx-community/arcface-onnx`, `garavv/arcface-onnx`, `onnxmodelzoo/arcfaceresnet100-8` |
| **ArcFace IR101** (CVLFace, transformers) | `pip install transformers huggingface_hub torch` | 512 | ~99.82% | ~97% | No | `minchul/cvlface_arcface_ir101_webface4m` |
| **AdaFace R100** | clone repo + `pip install -r requirements.txt` | 512 | 99.82% | 96.89% | No — pure PyTorch | no pip package; weights on Google Drive |
| **FaceNet InceptionResnetV1** | `pip install facenet-pytorch` | 512 | 99.65% (VGGFace2) / 99.05% (CASIA) | not reported | No | `py-feat/facenet` |
| **AuraFace R100** (commercial-friendly) | `pip install ...` (via HF) | 512 | not benchmarked vs LFW | not reported | No | `fal/AuraFace-v1` |

---

## 1. ArcFace via insightface

### Installation
```bash
pip install insightface
pip install onnxruntime-gpu  # GPU; or onnxruntime for CPU
```

**Current version:** 0.7.3 (April 2023 — last PyPI release; GitHub active).

**C compilation:** No. Since insightface>=0.2 the inference backend is pure ONNX Runtime. The only known pain point is the `mesh_core_cython` extension (used for 3D mesh, not face recognition). If you only use the recognition pipeline, you will not trigger this. Numpy 2.x can cause binary incompatibility — pin `numpy<2` if you see dtype size errors.

**CUDA 13 / onnxruntime-gpu note:** onnxruntime-gpu 1.24.x added CUDA 13 support. Check `onnxruntime-gpu` PyPI for the latest release that explicitly lists CUDA 13 wheels.

### API — embeddings without detection

```python
import insightface
import numpy as np

# Option A: full pipeline (detection + recognition)
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(img)  # img: BGR numpy array (HxWx3)
embedding = faces[0].embedding  # float32, shape (512,)

# Option B: recognition only (skip detection — pass pre-cropped 112x112)
handler = insightface.model_zoo.get_model('buffalo_l/w600k_r50.onnx')
handler.prepare(ctx_id=0)
# Or load the R100 recognizer directly from the downloaded model dir
```

**Models auto-download** to `~/.insightface/models/buffalo_l/` on first `prepare()`. The bundle contains: SCRFD detector + ArcFace W600K R50 recognizer.

**buffalo_l benchmarks:** LFW 99.83, CFP-FP 99.33, AgeDB-30 98.23, IJB-C(E4) 97.25.

---

## 2. ArcFace — ONNX-only path (no insightface, no compilation)

The cleanest path for embedding extraction without detection:

```bash
pip install onnxruntime-gpu numpy opencv-python
```

**HuggingFace ONNX models:**
- `onnx-community/arcface-onnx` — ResNet100, standard weights
- `garavv/arcface-onnx` — same, direct wget link
- `onnxmodelzoo/arcfaceresnet100-8` — ONNX Model Zoo R100
- `onnxmodelzoo/arcfaceresnet100-11-int8` — quantized int8 variant

```python
import onnxruntime as ort, numpy as np, cv2
from huggingface_hub import hf_hub_download

model_path = hf_hub_download("onnx-community/arcface-onnx", "arcface.onnx")
sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

def embed(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = (img.astype(np.float32) - 127.5) / 128.0
    inp = img[np.newaxis, ...]  # (1, 112, 112, 3) — channels last
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]
    return out / np.linalg.norm(out)  # L2-normalized, shape (512,)
```

**Input:** 112×112, RGB, channels last `(1, 112, 112, 3)`, normalized to `[-1, 1]` via `(x - 127.5) / 128`.
**Output:** 512-d float32.

---

## 3. ArcFace via transformers / HuggingFace hub

`minchul/cvlface_arcface_ir101_webface4m` — IR101 backbone, WebFace4M training.

```bash
pip install transformers huggingface_hub torch torchvision
```

```python
from transformers import AutoModel
from huggingface_hub import snapshot_download

path = snapshot_download("minchul/cvlface_arcface_ir101_webface4m")
model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().cuda()

from torchvision.transforms import Compose, ToTensor, Normalize
trans = Compose([ToTensor(), Normalize([0.5]*3, [0.5]*3)])
x = trans(pil_img).unsqueeze(0).cuda()  # 112x112 input
emb = model(x)  # (1, 512)
```

`trust_remote_code=True` is required — the repo ships a custom `wrapper.py`.
Also from `minchul`: `cvlface_adaface_ir101_webface4m` for AdaFace weights in the same format.

---

## 4. AdaFace

**No pip package.** No standalone HuggingFace model card with pretrained weights as of April 2026.

**Install path:**
```bash
git clone https://github.com/mk-minchul/AdaFace
cd AdaFace
pip install -r requirements.txt  # torch, torchvision, scikit-image, etc.
```

Pretrained weights are on Google Drive (linked from the README). R100/MS1MV2 is the canonical checkpoint.

**However:** `minchul/cvlface_adaface_ir101_webface4m` on HF hub loads AdaFace weights via the same transformers AutoModel pattern as above.

**Accuracy (R100/MS1MV2):**
- LFW: 99.82%
- IJB-C TAR@FAR=0.01%: 96.89%
- IJB-C TAR@FAR=1e-6: 89.74%

AdaFace outperforms ArcFace on mixed/low-quality benchmarks (IJB-S, IJB-C at strict FAR). On high-quality benchmarks (LFW, CFP-FP) they are essentially tied.

**Input:** BGR (not RGB — unlike insightface's ArcFace which uses RGB). 112×112.
**Embedding dim:** 512.

---

## 5. FaceNet (InceptionResnetV1)

```bash
pip install facenet-pytorch
```

```python
from facenet_pytorch import InceptionResnetV1
model = InceptionResnetV1(pretrained='vggface2').eval()
# Input: (B, 3, 160, 160), float, normalized to [-1, 1]
emb = model(x)  # (B, 512)
```

**Accuracy:**
- LFW: 99.65% (VGGFace2 weights), 99.05% (CASIA-Webface)
- IJB-C: not reported in standard benchmarks

**No C compilation.** Pure PyTorch. Weights download from HF hub automatically.
**Input size:** 160×160 (not 112×112 like ArcFace/AdaFace).
**HF hub:** `py-feat/facenet`

---

## 6. AuraFace (commercial-safe alternative)

HF hub: `fal/AuraFace-v1`. ResNet100, commercially licensed.
Performance is below ArcFace (trained on smaller dataset — no MS1MV2 due to license). CFP-FP 95.18% vs ArcFace 98.87%. Use only if commercial licensing of training data matters.

---

## Recommendation for vamp-interface

**Best choice for identity stability: ArcFace ONNX path (option 2)**

- `pip install onnxruntime-gpu huggingface_hub numpy opencv-python`
- Model: `onnx-community/arcface-onnx` or `onnxmodelzoo/arcfaceresnet100-8`
- Zero C compilation, no torch version sensitivity, CUDA 13 via onnxruntime-gpu 1.24+
- 512-d L2-normalized embeddings, same face → same embedding (deterministic ONNX)

**If PyTorch pipeline is already in use (SDXL/ComfyUI stack):** use insightface buffalo_l — it's the highest-accuracy off-the-shelf bundle and handles alignment automatically.

**AdaFace via minchul HF hub** (`cvlface_adaface_ir101_webface4m`) is worth trying if quality robustness matters more than installation simplicity.

---

## Sources

- [insightface PyPI](https://pypi.org/project/insightface/)
- [deepinsight/insightface GitHub](https://github.com/deepinsight/insightface)
- [mk-minchul/AdaFace GitHub](https://github.com/mk-minchul/AdaFace)
- [minchul/cvlface_arcface_ir101_webface4m — HF Hub](https://huggingface.co/minchul/cvlface_arcface_ir101_webface4m)
- [onnx-community/arcface-onnx — HF Hub](https://huggingface.co/onnx-community/arcface-onnx)
- [garavv/arcface-onnx — HF Hub](https://huggingface.co/garavv/arcface-onnx)
- [fal/AuraFace-v1 — HF Hub](https://huggingface.co/fal/AuraFace-v1)
- [AuraFace blog post](https://huggingface.co/blog/isidentical/auraface)
- [facenet-pytorch PyPI](https://pypi.org/project/facenet-pytorch/)
- [facenet-pytorch LFW evaluation](https://deepwiki.com/timesler/facenet-pytorch/4.4-evaluation-on-lfw-dataset)
- [AdaFace paper ar5iv](https://ar5iv.labs.arxiv.org/html/2204.00964)
- [insightface model zoo README](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
- [onnxruntime CUDA compatibility](https://onnxruntime.ai/docs/reference/compatibility.html)
- [onnxruntime CUDA 13 issue](https://github.com/microsoft/onnxruntime/issues/26238)
- [yakhyo/uniface GitHub](https://github.com/yakhyo/uniface)
