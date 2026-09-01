"""Validation suite for baidu/Unlimited-OCR on vLLM (dedicated unlimited-ocr image)."""
import base64
import io
import os
import sys

from openai import OpenAI
from PIL import Image, ImageDraw

BASE_URL = sys.argv[1].rstrip("/") + ("/v1" if not sys.argv[1].rstrip("/").endswith("/v1") else "")
TOKEN = os.environ.get("HF_TOKEN") or __import__("subprocess").run(["hf","auth","token"],capture_output=True,text=True).stdout.strip()
client = OpenAI(base_url=BASE_URL, api_key=TOKEN, timeout=600)
MODEL = sys.argv[2] if len(sys.argv) > 2 else client.models.list().data[0].id

# grounding markers, built programmatically to keep them literal
DET = "<" + "|det|" + ">"
REF = "<" + "|ref|" + ">"

results = []


def report(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def make_page(lines, size=(1024, 1024)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 40
    for line, font_size in lines:
        d.text((40, y), line, fill="black", font_size=font_size)
        y += font_size + 18
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


PAGE1_LINES = [
    ("Quarterly Report 2026", 36),
    ("Revenue increased by 24 percent year over year.", 24),
    ("The zebra crossed the quiet reservoir at dawn.", 24),
    ("Table 1: Unit sales by region", 24),
    ("North 1200  South 950  East 1430", 20),
]
PAGE2_LINES = [
    ("Appendix B: Methodology", 36),
    ("Samples were collected across fourteen sites.", 24),
    ("Calibration used the KX-900 reference standard.", 24),
]
PAGE1 = make_page(PAGE1_LINES)
PAGE2 = make_page(PAGE2_LINES)
GT1 = " ".join(t for t, _ in PAGE1_LINES)
GT2 = " ".join(t for t, _ in PAGE2_LINES)


def ocr(images, prompt="<image>document parsing.", window=128, skip_special=False):
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": u}} for u in images
    ]
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=8192,
        temperature=0.0,
        extra_body={
            "skip_special_tokens": skip_special,
            "vllm_xargs": {"ngram_size": 35, "window_size": window},
        },
    )


def word_overlap(text, gt):
    gt_words = {w.strip(".,:;").lower() for w in gt.split() if len(w) > 3}
    if not gt_words:
        return 0.0
    found = sum(1 for w in gt_words if w in text.lower())
    return found / len(gt_words)


# 1. models
models = client.models.list()
report("models.list", any(MODEL in m.id for m in models.data), f"{[m.id for m in models.data]}")

# 2. single image (gundam mode), grounding tokens kept
resp = ocr([PAGE1])
out = resp.choices[0].message.content or ""
ov = word_overlap(out, GT1)
report("ocr.single_image", len(out) > 50 and ov > 0.7, f"len={len(out)}, word_overlap={ov:.2f}")
report("ocr.grounding_tokens_kept", (DET in out) or (REF in out), f"det={DET in out}, ref={REF in out}")

# 3. skip_special_tokens default strips grounding tokens
resp = ocr([PAGE1], skip_special=True)
out_strip = resp.choices[0].message.content or ""
report(
    "ocr.skip_special_strips_grounding",
    DET not in out_strip and REF not in out_strip and len(out_strip) > 50,
    f"len={len(out_strip)}",
)

# 4. documented gotcha: missing literal <image> prefix degrades output
resp = ocr([PAGE1], prompt="document parsing.")
out_noprefix = resp.choices[0].message.content or ""
ov_noprefix = word_overlap(out_noprefix, GT1)
report(
    "ocr.no_prefix_degrades",
    ov_noprefix < ov,
    f"overlap_with_prefix={ov:.2f}, without={ov_noprefix:.2f}, len={len(out_noprefix)}",
)

# 5. multi-page (base mode), window_size=1024
resp = ocr([PAGE1, PAGE2], prompt="<image>Multi page parsing.", window=1024)
out_multi = resp.choices[0].message.content or ""
ov1, ov2 = word_overlap(out_multi, GT1), word_overlap(out_multi, GT2)
report("ocr.multi_page", ov1 > 0.7 and ov2 > 0.7, f"page1={ov1:.2f}, page2={ov2:.2f}")

# summary
fails = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(fails)}/{len(results)} passed" + (f" | FAILED: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
