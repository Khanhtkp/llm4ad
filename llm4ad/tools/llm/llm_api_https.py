from __future__ import annotations

import http.client
import json
import time
from typing import Any
import traceback
from urllib.parse import urlencode
from ...base import LLM

import google.generativeai as genai

class HttpsApi(LLM):
    def __init__(self, key, model, timeout=60, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model)
        self._timeout = timeout
        self._kwargs = kwargs

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        try:
            print(f"[INFO] Calling Gemini model: {self._model.model_name}")
            response = self._model.generate_content(
                prompt
            )
            return response.text
        except Exception as e:
            print(f"[ERROR] Gemini API call failed:\n{traceback.format_exc()}")
            return "API_FAILED"
