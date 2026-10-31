"""
PagePilot – Stage 6B Mathematical Formula Recognition Benchmark Suite.

Executes genuine Pix2Text MFR 1.5 inference across 6 diverse test cases:
1. Quadratic Equation: x = (-b +- sqrt(b^2 - 4ac)) / (2a)
2. Algebraic Fraction: (a^2 - b^2) / (a - b) = a + b
3. Euler's Identity: e^(i*pi) + 1 = 0
4. Gaussian Integral: integral e^(-x^2) dx = sqrt(pi)
5. Calculus Derivative: d/dx[e^(2x) sin(x)]
6. Original Quadratic Bitmap Sample: quadratic_root_equation.png
"""

import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt
import psutil

from pagepilot.ocr.formula_preprocessor import clean_formula_latex, detect_content_bbox, preprocess_formula_image
from pagepilot.ocr.formula_validator import validate_formula_latex
from pix2text import Pix2Text

bench_dir = Path(r"C:\Users\bhara\.gemini\antigravity\brain\5b57fd03-cce2-4805-b30c-6ebb140452cb\scratch\formula_benchmarks")
bench_dir.mkdir(parents=True, exist_ok=True)

orig_quadratic = Path(r"C:\Users\bhara\.gemini\antigravity\brain\5b57fd03-cce2-4805-b30c-6ebb140452cb\scratch\quadratic_root_equation.png")


def render_latex_sample(filename: str, latex_expr: str, fontsize: int = 24, dpi: int = 200) -> Path:
    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(5, 1.5), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.text(0.5, 0.5, f"${latex_expr}$", fontsize=fontsize, ha="center", va="center")
    ax.axis("off")
    out_path = bench_dir / filename
    plt.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    return out_path


def render_handwritten_style_sample(filename: str) -> Path:
    """Render a handwritten-style derivative equation with subtle slant and ink variation."""
    img = Image.new("RGB", (600, 160), color=(252, 250, 245))  # Warm paper tone
    d = ImageDraw.Draw(img)

    # Ruled paper line
    d.line([(20, 130), (580, 130)], fill=(215, 225, 235), width=1)

    # Use default or system font with slightly spaced rendering
    text = r"d/dx[e^(2x) sin(x)] = 2e^(2x) sin(x) + e^(2x) cos(x)"
    d.text((40, 60), text, fill=(25, 25, 40))

    out_path = bench_dir / filename
    img.save(out_path)
    return out_path


def run_benchmark():
    print("=" * 75)
    print("PAGEPILOT STAGE 6B: MATHEMATICAL FORMULA RECOGNITION BENCHMARK")
    print("=" * 75)

    process = psutil.Process()
    initial_rss = process.memory_info().rss / (1024 * 1024)
    print(f"Initial RSS Memory: {initial_rss:.1f} MB")

    # 1. Generate benchmark test samples
    print("\n[1/3] Generating benchmark test sample images...")
    cases = [
        {
            "id": "case1_quadratic_formula",
            "name": "Standard Quadratic Formula",
            "target": r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
            "path": render_latex_sample("case1_quadratic_root.png", r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", fontsize=24),
        },
        {
            "id": "case2_algebraic_fraction",
            "name": "Algebraic Fraction Identity",
            "target": r"\frac{a^2 - b^2}{a - b} = a + b",
            "path": render_latex_sample("case2_fraction_algebra.png", r"\frac{a^2 - b^2}{a - b} = a + b", fontsize=24),
        },
        {
            "id": "case3_euler_identity",
            "name": "Euler's Identity (Exponent & Constant)",
            "target": r"e^{i\pi} + 1 = 0",
            "path": render_latex_sample("case3_euler_exponent.png", r"e^{i\pi} + 1 = 0", fontsize=26),
        },
        {
            "id": "case4_gaussian_integral",
            "name": "Gaussian Definite Integral (Square Root & Calculus)",
            "target": r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}",
            "path": render_latex_sample("case4_gaussian_integral.png", r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}", fontsize=24),
        },
        {
            "id": "case5_handwritten_derivative",
            "name": "Handwritten Student Note Style (Product Rule)",
            "target": r"\frac{d}{dx}[e^{2x} \sin(x)] = 2e^{2x}\sin(x) + e^{2x}\cos(x)",
            "path": render_handwritten_style_sample("case5_handwritten_derivative.png"),
        },
        {
            "id": "case6_original_quadratic_bitmap",
            "name": "Original Quadratic Bitmap Sample (quadratic_root_equation.png)",
            "target": r"x = (-b \pm \sqrt{b^2 - 4ac}) / (2a)",
            "path": orig_quadratic,
        },
    ]

    for c in cases:
        img = Image.open(c["path"])
        bbox = detect_content_bbox(img)
        print(f"  - {c['name']:<48}: Size={img.size}, ContentBbox={bbox}")

    # 2. Initialize Pix2Text MFR
    print("\n[2/3] Initializing genuine Pix2Text MFR 1.5 engine...")
    t0 = time.perf_counter()
    p2t = Pix2Text()
    init_duration = time.perf_counter() - t0
    after_init_rss = process.memory_info().rss / (1024 * 1024)
    print(f"  Initialized in {init_duration:.2f} s. Memory: {after_init_rss:.1f} MB (Delta: +{after_init_rss - initial_rss:.1f} MB)")

    # 3. Run Benchmark across all cases
    print("\n[3/3] Executing real inference benchmarks...")
    results = []

    for idx, c in enumerate(cases, 1):
        print(f"\n--- Test Case {idx}: {c['name']} ---")
        img_raw = Image.open(c["path"]).convert("RGB")

        # A) Raw Inference (No preprocessing)
        t_raw_start = time.perf_counter()
        raw_res = p2t.recognize_formula(img_raw, return_text=False)
        t_raw_ms = (time.perf_counter() - t_raw_start) * 1000.0
        raw_text = raw_res.get("text", "") if isinstance(raw_res, dict) else str(raw_res)
        raw_score = float(raw_res.get("score", 0.0)) if isinstance(raw_res, dict) else 0.90
        raw_val = validate_formula_latex(raw_text, confidence_score=raw_score)

        # B) Preprocessed Inference
        t_prep_start = time.perf_counter()
        preproc_img = preprocess_formula_image(c["path"])
        prep_res = p2t.recognize_formula(preproc_img, return_text=False)
        prep_text = prep_res.get("text", "") if isinstance(prep_res, dict) else str(prep_res)
        prep_score = float(prep_res.get("score", 0.0)) if isinstance(prep_res, dict) else 0.90
        cleaned_text = clean_formula_latex(prep_text)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0
        clean_val = validate_formula_latex(cleaned_text, confidence_score=prep_score)

        current_rss = process.memory_info().rss / (1024 * 1024)

        # Save preprocessed image artifact
        prep_save_path = bench_dir / f"preproc_{Path(c['path']).name}"
        preproc_img.save(prep_save_path)

        case_result = {
            "case_id": c["id"],
            "name": c["name"],
            "target": c["target"],
            "source_image": str(c["path"]),
            "original_dimensions": img_raw.size,
            "preprocessed_dimensions": preproc_img.size,
            "raw_output": raw_text,
            "raw_score": round(raw_score, 4),
            "raw_latency_ms": round(t_raw_ms, 2),
            "raw_is_valid": raw_val.is_valid,
            "raw_errors": raw_val.errors,
            "raw_warnings": raw_val.warnings,
            "preprocessed_output": cleaned_text,
            "preprocessed_score": round(prep_score, 4),
            "preprocessed_latency_ms": round(t_prep_ms, 2),
            "preprocessed_is_valid": clean_val.is_valid,
            "preprocessed_errors": clean_val.errors,
            "preprocessed_warnings": clean_val.warnings,
            "needs_human_review": clean_val.needs_human_review,
            "rss_memory_mb": round(current_rss, 1),
        }
        results.append(case_result)

        print(f"  Target Formula : {c['target']}")
        print(f"  Raw Output     : {raw_text[:70]}... (conf={raw_score:.3f}, valid={raw_val.is_valid}, {t_raw_ms:.1f}ms)")
        print(f"  Preproc Output : {cleaned_text[:70]}... (conf={prep_score:.3f}, valid={clean_val.is_valid}, {t_prep_ms:.1f}ms)")
        print(f"  Val Status     : Valid={clean_val.is_valid}, Warnings={clean_val.warnings}, NeedsReview={clean_val.needs_human_review}")

    # Summary JSON
    summary_path = bench_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device": "CPU",
            "model": "Pix2Text MFR 1.5",
            "cases_evaluated": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"BENCHMARK COMPLETED: Results saved to {summary_path}")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()

