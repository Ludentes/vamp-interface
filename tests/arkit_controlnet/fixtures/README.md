# arkit_controlnet test fixtures

`face.png` — a single-face photo used by `test_eval_spike.py`. It is **not
committed** (this repo gitignores images). The eval tests `skipif` it is
absent. To populate it locally, copy any clear single-face photo, e.g. an
FFHQ sample:

```bash
python3 - <<'PY'
import cv2, glob, shutil
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
app.prepare(ctx_id=0)
for p in sorted(glob.glob("output/ffhq_images/*.png")):
    if len(app.get(cv2.imread(p))) == 1:
        shutil.copy(p, "tests/arkit_controlnet/fixtures/face.png")
        print("wrote face.png from", p)
        break
PY
```
