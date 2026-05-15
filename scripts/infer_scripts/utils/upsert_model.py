#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LiteLLM model upsert script

Usage:
  export LITELLM_BASE_URL="http://127.0.0.1:20497"
  export LITELLM_API_KEY="sk-your-master-key"

  python upsert_model.py

Notes:
- Reads existing models from GET /model/info
- Creates model via POST /model/new
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

BASE_URL = os.getenv("LITELLM_BASE_URL", "http://127.0.0.1:4000").rstrip("/")
API_KEY = os.getenv("LITELLM_API_KEY", "sk-1234")

TIMEOUT = 30


class LiteLLMError(RuntimeError):
    pass


def _headers() -> Dict[str, str]:
    if not API_KEY:
        raise LiteLLMError("LITELLM_API_KEY is empty.")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text


def get_model_info() -> Any:
    url = f"{BASE_URL}/model/info"
    resp = requests.get(url, headers=_headers(), timeout=TIMEOUT)
    if resp.status_code != 200:
        raise LiteLLMError(
            f"GET {url} failed: {resp.status_code} {_safe_json(resp)}"
        )
    return resp.json()


def create_model(payload: Dict[str, Any]) -> Any:
    url = f"{BASE_URL}/model/new"
    resp = requests.post(url, headers=_headers(), json=payload, timeout=TIMEOUT)
    if resp.status_code not in (200, 201):
        raise LiteLLMError(
            f"POST {url} failed: {resp.status_code} {_safe_json(resp)}"
        )
    return resp.json()


def delete_model(model_id: str) -> Any:
    url = f"{BASE_URL}/model/delete"
    payload = {"id": model_id}
    resp = requests.post(url, headers=_headers(), json=payload, timeout=TIMEOUT)
    if resp.status_code not in (200, 201):
        raise LiteLLMError(
            f"POST {url} failed: {resp.status_code} {_safe_json(resp)}"
        )
    return resp.json()


def find_existing_model(model_info_resp: Any, model_name: str) -> Optional[Dict[str, Any]]:
    """
    LiteLLM /model/info response shape can vary by version/config.
    This function tries a few common shapes.
    """
    candidates: List[Dict[str, Any]] = []

    if isinstance(model_info_resp, list):
        candidates = [x for x in model_info_resp if isinstance(x, dict)]
    elif isinstance(model_info_resp, dict):
        # common shapes
        for key in ("data", "models", "model_info", "result"):
            value = model_info_resp.get(key)
            if isinstance(value, list):
                candidates = [x for x in value if isinstance(x, dict)]
                break
        else:
            candidates = [model_info_resp]

    for item in candidates:
        if item.get("model_name") == model_name:
            return item

    return None


def upsert_single_model(desired: Dict[str, Any], all_existing_models: Any) -> bool:
    """
    Upsert a single model. Returns True if successful.
    """
    model_name = desired.get("model_name")
    if not model_name:
        print("[WARN] Model skipped: 'model_name' missing in definition.")
        return False

    print(f"\n[INFO] Processing model: {model_name}")
    existing = find_existing_model(all_existing_models, model_name)

    if existing:
        print(f"[INFO] '{model_name}' already exists. Deleting and recreating...")
        model_id = (
            existing.get("id") or 
            existing.get("model_id") or 
            existing.get("model_info", {}).get("id") or 
            existing.get("model_info", {}).get("model_id")
        )
        if not model_id:
            raise LiteLLMError(f"Found '{model_name}' but could not determine its ID for deletion.")

        delete_model(model_id)
        print(f"[INFO] '{model_name}' (id: {model_id}) deleted.")

    print(f"[INFO] Creating '{model_name}'...")
    create_model(desired)
    print(f"[OK] '{model_name}' processed.")
    return True


def load_config(config_path: str) -> List[Dict[str, Any]]:
    """
    Load model_list from LiteLLM config YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise LiteLLMError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config or "model_list" not in config:
        print(f"[WARN] No 'model_list' found in {config_path}")
        return []

    return config["model_list"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Upsert models into LiteLLM from config.")
    parser.add_argument(
        "-c", "--config",
        default="litellm_config.yaml",
        help="Path to LiteLLM config YAML (default: litellm_config.yaml)"
    )
    args = parser.parse_args()

    try:
        models_to_upsert = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return 1

    if not models_to_upsert:
        print("[INFO] No models to process.")
        return 0

    print(f"[INFO] Fetching current model info from {BASE_URL}...")
    try:
        all_existing_models = get_model_info()
    except Exception as e:
        print(f"[ERROR] Failed to fetch model info: {e}")
        return 1

    success_count = 0
    for model_def in models_to_upsert:
        try:
            if upsert_single_model(model_def, all_existing_models):
                success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to upsert model '{model_def.get('model_name')}': {e}")

    print(f"\n[DONE] Successfully processed {success_count}/{len(models_to_upsert)} models.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)