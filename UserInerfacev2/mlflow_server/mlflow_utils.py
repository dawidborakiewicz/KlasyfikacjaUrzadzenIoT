import pandas as pd
import json
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:8080")
client = MlflowClient()

def _j(tags: dict) -> str:
    return json.dumps(tags or {}, ensure_ascii=False)

def fetch_models_registry():
    rows = []
    names = []

    for rm in client.search_registered_models():
        name = rm.name
        names.append(name)
        model_tags = dict(rm.tags or {})

        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            rows.append({
                "model": name, "stage": "", "version": "",
                "model_tags": _j(model_tags), "version_tags": _j({}),
            })
            continue

        for mv in versions:
            rows.append({
                "model": name,
                "stage": mv.current_stage or "None",
                "version": mv.version,
                "model_tags": _j(model_tags),
                "version_tags": _j(dict(mv.tags or {})),
            })

    df = pd.DataFrame(rows, columns=["model", "stage", "version", "model_tags", "version_tags"])
    return df, sorted(set(names))

def choice_to_model_uri(choice: str, prefer_stage: bool = True) -> str:
    parts = [p.strip() for p in choice.split("|")]
    name = parts[0]
    stage = parts[1].replace("stage=", "").strip()
    v = parts[2].replace("v=", "").strip()

    if prefer_stage and stage and stage.lower() != "none":
        return f"models:/{name}/{stage}"
    if v:
        return f"models:/{name}/{v}"
    raise ValueError(f"Wybrany model '{name}' nie ma wersji/stage.")


def pick_model_uri_for_name(name: str) -> str:
    try:
        mlflow.pyfunc.load_model(f"models:/{name}/Production")
        return f"models:/{name}/Production"
    except Exception:
        pass

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise ValueError(f"Model '{name}' nie ma wersji w registry.")
    latest = max(versions, key=lambda mv: int(mv.version))
    return f"models:/{name}/{latest.version}"

def load_selected_model(name: str):
    if not name:
        return None, "", "⚠️ Wybierz model."
    uri = pick_model_uri_for_name(name)

    model = mlflow.sklearn.load_model(uri)

    return model, name, f"✅ Załadowano `{name}` ({uri}) [sklearn]"