from __future__ import annotations

from abc import ABC, abstractmethod

import requests
from llama_cpp import Llama
from transformers import AutoTokenizer

from src.config.settings import Settings


class LLMBackend(ABC):
    name: str

    @abstractmethod
    def ready(self) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def max_context_tokens(self) -> int:
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    name = "llama_cpp"

    def __init__(self, settings: Settings):
        self.settings = settings
        cfg = settings.llama_cpp
        self.llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            n_threads=cfg.n_threads,
            verbose=False,
        )

    def ready(self) -> tuple[bool, str]:
        return True, "ok"

    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> str:
        response = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
        return response["choices"][0]["text"].strip()

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8")))

    def max_context_tokens(self) -> int:
        return self.settings.llama_cpp.n_ctx


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    def ready(self) -> tuple[bool, str]:
        cfg = self.settings.ollama
        try:
            r = requests.get(f"{cfg.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = {m["name"] for m in r.json().get("models", [])}
            if cfg.model not in models and f"{cfg.model}:latest" not in models:
                return False, f"model {cfg.model} not found"
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    def generate(self, *, prompt: str, max_tokens: int, temperature: float) -> str:
        cfg = self.settings.ollama
        r = requests.post(
            f"{cfg.base_url}/api/generate",
            json={
                "model": cfg.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def max_context_tokens(self) -> int:
        return self.settings.ollama.context_window


def build_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "llama_cpp":
        return LlamaCppBackend(settings)
    return OllamaBackend(settings)
