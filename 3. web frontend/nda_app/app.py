import os
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

from model_loader import ModelLoader
from parser import parse_nda
from flag_checker import check_flag


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

loader = ModelLoader()

AVAILABLE_MODELS = [
    "fine_tuned_bert-base-uncased_weighted",
    "fine_tuned_distilbert-base-uncased_weighted",
    "fine_tuned_roberta-base_weighted",
    "xgboost_model",
    "shallow_xgboost",
    "shallow_svm"
]

DISPLAY_NAMES = {
    "fine_tuned_bert-base-uncased_weighted": "BERT-base Weighted",
    "fine_tuned_distilbert-base-uncased_weighted": "DistilBERT Weighted",
    "fine_tuned_roberta-base_weighted": "RoBERTa Weighted",
    "xgboost_model": "XGBoost + MPNet",
    "shallow_xgboost": "Shallow XGBoost",
    "shallow_svm": "SVM"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_choice = request.form.get("model_name")
        file = request.files.get("file")

        if not file:
            return " Please upload a .docx file"

        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        df = parse_nda(file_path)
        if df is None or df.empty:
            return " File parsing failed. Ensure it's a valid NDA."

        df["predicted_category"] = df["normalized_sentence"].apply(
            lambda x: loader.predict(model_choice, x)
        )
        df["flagged"] = df.apply(
            lambda row: check_flag(row["predicted_category"], row["original_sentence"]),
            axis=1
        )
        df["model_used"] = DISPLAY_NAMES[model_choice]

        output_csv = os.path.join(OUTPUT_FOLDER, "nda_predictions.csv")
        df.to_csv(output_csv, index=False)
        results_df = df.drop(columns=["normalized_sentence"], errors='ignore')

        return render_template(
            "results.html",
            tables=results_df.to_dict(orient="records"),
            download_link="/download",
            model_used=DISPLAY_NAMES[model_choice]
        )


    return render_template("index.html", models=AVAILABLE_MODELS, names=DISPLAY_NAMES)


@app.route("/download")
def download_file():
    return send_file("outputs/nda_predictions.csv", as_attachment=True)


if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False,
        processes=1
    )
