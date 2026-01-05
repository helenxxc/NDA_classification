import os
import json
import torch
import joblib
import xgboost as xgb
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# HF_MODELS = {
#     "fine_tuned_bert-base-uncased_weighted": "helenxxc/nda-bert-weighted",
#     "fine_tuned_distilbert-base-uncased_weighted": "helenxxc/nda-distilbert-weighted",
#     "fine_tuned_roberta-base_weighted": "helenxxc/nda-roberta-weighted",
# }

HF_MODELS = {
    "fine_tuned_bert-base-uncased_weighted": "nda_flask/models/fine_tuned_bert-base-uncased_weighted",
    "fine_tuned_distilbert-base-uncased_weighted": "nda_flask/models/fine_tuned_distilbert-base-uncased_weighted",
    "fine_tuned_roberta-base_weighted": "nda_flask/models/fine_tuned_roberta-base_weighted",
}

class ModelLoader:
    def __init__(self):
        self.cache = {}  

    # def _load_hf_model(self, key: str):
    #     repo_id = HF_MODELS[key]
    #     tokenizer = AutoTokenizer.from_pretrained(repo_id)
    #     model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    #     model.eval()
    #     return {"model": model, "tokenizer": tokenizer, "id2label": model.config.id2label}
    def _load_hf_model(self, folder_name: str):
        model_path = MODELS_DIR / folder_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model folder not found: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_auth_token=None
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            use_auth_token=None
        )

        mapping_path = model_path / "label_mappings.json"
        if mapping_path.exists():
            with open(mapping_path, "r") as f:
                label_maps = json.load(f)
            label2id = {k: int(v) for k, v in label_maps["label2id"].items()}
            id2label = {int(k): v for k, v in label_maps["id2label"].items()}
        else:
            label2id = model.config.label2id
            id2label = {int(k): v for k, v in model.config.id2label.items()}

        model.eval()

        return {
            "type": "huggingface",
            "model": model,
            "tokenizer": tokenizer,
            "label2id": label2id,
            "id2label": id2label,
        }

    def _load_xgb_model(self, folder_name: str):
        model_path = MODELS_DIR / folder_name
        model = xgb.XGBClassifier()
        model.load_model(str(model_path / "xgb_nda_model_weighted.json"))

        embedder_name = "sentence-transformers/all-mpnet-base-v2"
        if (model_path / "embedder_name.txt").exists():
            embedder_name = (model_path / "embedder_name.txt").read_text().strip()

        embedder = SentenceTransformer(embedder_name)
        mapping = json.load(open(model_path / "label_mappings.json"))
        label2id = {k: int(v) for k, v in mapping["label2id"].items()}
        id2label = {int(k): v for k, v in mapping["id2label"].items()}

        return {
            "type": "xgboost_st",
            "model": model,
            "embedder": embedder,
            "label2id": label2id,
            "id2label": id2label,
        }

    def _load_svm_model(self, folder_name: str):
        model_path = MODELS_DIR / folder_name
        svm_model = joblib.load(model_path / "svm_model.pkl")
        tfidf_vectorizer = joblib.load(model_path / "tfidf_vectorizer.pkl")
        label_encoder = joblib.load(model_path / "label_encoder.pkl")
        classes = json.load(open(model_path / "classes.json"))

        return {
            "type": "shallow_svm",
            "model": svm_model,
            "vectorizer": tfidf_vectorizer,
            "label_encoder": label_encoder,
            "classes": classes,
        }

    def _load_shallow_xgboost(self, folder_name: str):
        model_path = MODELS_DIR / folder_name
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(model_path / "xgboost_model.json"))
        tfidf_vectorizer = joblib.load(model_path / "tfidf_vectorizer.pkl")
        label_encoder = joblib.load(model_path / "label_encoder.pkl")
        classes = json.load(open(model_path / "classes.json"))

        return {
            "type": "shallow_xgboost",
            "model": xgb_model,
            "vectorizer": tfidf_vectorizer,
            "label_encoder": label_encoder,
            "classes": classes,
        }

    def get_model(self, key: str):
        if key in self.cache:
            return self.cache[key]

        if key == "xgboost_model":
            info = self._load_xgb_model("xgboost_model")
        elif key == "shallow_svm":
            info = self._load_svm_model("shallow_svm")
        elif key == "shallow_xgboost":
            info = self._load_shallow_xgboost("shallow_xgboost")
        else:
            info = self._load_hf_model(key)

        self.cache[key] = info
        return info

    def predict(self, key: str, text: str) -> str:
        info = self.get_model(key)

        if info["type"] == "huggingface":
            tokenizer = info["tokenizer"]
            model = info["model"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_id = int(torch.argmax(logits, dim=1).item())

        elif info["type"] == "xgboost_st":
            embedder = info["embedder"]
            model = info["model"]
            embedding = embedder.encode([text], convert_to_numpy=True)
            pred_id = int(model.predict(embedding)[0])

        elif info["type"] == "shallow_svm":
            vec = info["vectorizer"].transform([text])
            pred_label = info["model"].predict(vec)[0]
            return pred_label

        elif info["type"] == "shallow_xgboost":
            vec = info["vectorizer"].transform([text])
            pred_enc = info["model"].predict(vec)
            return info["label_encoder"].inverse_transform(pred_enc)[0]

        return info["id2label"][pred_id]
