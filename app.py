"""
Campus Vegetation Recognition — local web UI (Flask).

Loads the trained pipeline from outputs/ (same artefacts as evaluate.py).
Run: python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from evaluate import extract_features

ALLOWED_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp", "gif"})
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MiB


def load_image_bytes(raw: bytes, img_size: int) -> np.ndarray:
    img = Image.open(BytesIO(raw)).convert("RGB").resize((img_size, img_size))
    return np.array(img, dtype=np.uint8)


class PlantClassifier:
    def __init__(self, out_dir: Path, img_size: int = 128):
        out_dir = out_dir.resolve()
        self.img_size = img_size
        self.model = joblib.load(out_dir / "best_model.pkl")
        self.scaler = joblib.load(out_dir / "scaler.pkl")
        self.pca = joblib.load(out_dir / "pca.pkl")
        self.label_encoder = joblib.load(out_dir / "label_encoder.pkl")
        self.classes = list(self.label_encoder.classes_)

    def predict(self, img_rgb: np.ndarray) -> tuple[str, float, dict[str, float]]:
        x = extract_features(img_rgb).reshape(1, -1).astype(np.float32)
        x = self.scaler.transform(x)
        x = self.pca.transform(x)
        idx = int(self.model.predict(x)[0])
        label = str(self.label_encoder.inverse_transform([idx])[0])

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
        else:
            dec = self.model.decision_function(x)
            e = np.exp(dec - dec.max(axis=1, keepdims=True))
            proba = (e / e.sum(axis=1, keepdims=True))[0]

        conf = float(proba[idx])
        probs = {str(c): float(p) for c, p in zip(self.classes, proba)}
        return label, conf, probs


def create_app(out_dir: Path | None = None, img_size: int = 128) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    root = Path(__file__).resolve().parent
    app.config["OUT_DIR"] = Path(out_dir) if out_dir else root / "outputs"
    app.config["IMG_SIZE"] = img_size
    app.config["CLASSIFIER"] = None

    def get_classifier() -> PlantClassifier:
        clf = app.config.get("CLASSIFIER")
        if clf is None:
            clf = PlantClassifier(app.config["OUT_DIR"], app.config["IMG_SIZE"])
            app.config["CLASSIFIER"] = clf
        return clf

    @app.route("/", methods=["GET"])
    def index():
        err = request.args.get("error")
        return render_template("index.html", prediction=None, error=err, classes=None)

    @app.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return redirect(url_for("index", error="No file selected."))
        f = request.files["image"]
        if not f.filename:
            return redirect(url_for("index", error="No file selected."))
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return redirect(
                url_for(
                    "index",
                    error=f"Unsupported format (.{ext}). Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
                )
            )
        raw = f.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            return redirect(url_for("index", error="File too large (max 16 MB)."))

        try:
            clf = get_classifier()
            img = load_image_bytes(raw, clf.img_size)
            label, conf, probs = clf.predict(img)
        except FileNotFoundError as e:
            return redirect(url_for("index", error=str(e)))
        except Exception as e:
            return redirect(url_for("index", error=f"Could not read image: {e}"))

        top = sorted(probs.items(), key=lambda kv: -kv[1])[:8]
        return render_template(
            "index.html",
            prediction={"label": label, "confidence": conf},
            probs=top,
            filename=secure_filename(f.filename),
            error=None,
            classes=clf.classes,
        )

    @app.route("/health")
    def health():
        try:
            get_classifier()
            return {"status": "ok", "out_dir": str(app.config["OUT_DIR"])}
        except Exception as e:
            return {"status": "error", "detail": str(e)}, 500

    return app


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Local plant classifier web app (Flask)")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=root / "outputs",
        help="Folder with best_model.pkl, scaler.pkl, pca.pkl, label_encoder.pkl",
    )
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--debug", action="store_true", help="Flask debug mode (auto-reload)")
    args = p.parse_args()

    od = args.out_dir.resolve()
    for name in ("best_model.pkl", "scaler.pkl", "pca.pkl", "label_encoder.pkl"):
        if not (od / name).is_file():
            raise SystemExit(
                f"Missing {od / name}. Train first: python classifier.py --data_dir data --out_dir {od}"
            )

    app = create_app(out_dir=od, img_size=args.img_size)
    print(f"Open http://{args.host}:{args.port}/ in your browser.")
    print(f"Model directory: {od}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
