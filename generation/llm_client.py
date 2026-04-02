# generation/llm_client.py
import logging
from typing import List, Dict, Any
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(
        self,
        model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int = 8,
        temperature: float = 0.1,
        max_tokens: int = 512
    ):
        logger.info(f"Загрузка LLM: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_answer(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Генерирует ответ на основе запроса и контекста."""
        context = "\n\n".join([
            f"Документ {i+1} (источник: {doc['metadata']['source']}):\n{doc['text']}"
            for i, doc in enumerate(context_documents)
        ])
        
        prompt = f"""Ты — эксперт службы поддержки компании Sommers. Отвечай ТОЛЬКО на основе предоставленных документов ниже.

ИНСТРУКЦИИ:
1. Если ответ нельзя вывести из документов — скажи: «В предоставленных материалах нет информации по этому вопросу».
2. НИКОГДА не придумывай детали.
3. Ссылайся на источник: укажи название документа (например, «согласно инструкции по виртуальной кассе»).
4. Отвечай кратко, по делу, на русском языке.

ДОКУМЕНТЫ:
{context}

ВОПРОС: {query}

ОТВЕТ:"""
        
        # Логируем длину промпта в токенах
        prompt_tokens = len(self.llm.tokenize(prompt.encode("utf-8")))
        logger.info(f"Длина промпта: {prompt_tokens} токенов (лимит: 4096)")
        if prompt_tokens > 3800:
            logger.warning("⚠️ Промпт близок к лимиту контекста! Ответ может быть обрезан.")
        
        response = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n\n", "ВОПРОС:", "ИНСТРУКЦИИ:"],
            echo=False
        )
        
        answer = response["choices"][0]["text"].strip()
        return answer if answer else "Не удалось сгенерировать ответ."