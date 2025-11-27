"""
LLM Validator Module
Uses LLM (local ollama or API) to validate semantic coherence of sentences.
"""
import requests
import json
import os
from typing import Optional


class LLMValidator:
    """Validates sentence semantic coherence using an LLM."""

    def __init__(self, model: str = "gemma2:9b", provider: str = "ollama", timeout: int = 10):
        """
        Initialize the LLM validator.

        Args:
            model: Model name to use (default: gemma2:9b - best balance of speed and quality)
            provider: 'ollama' (local, free), 'openai' (API, costs $), or 'anthropic' (API, costs $)
            timeout: Request timeout in seconds
        """
        self.model = model
        self.provider = provider
        self.timeout = timeout
        self.checked_count = 0
        self.accepted_count = 0

        # Setup provider-specific configuration
        if provider == 'ollama':
            self.url = "http://localhost:11434/api/generate"
            # Test ollama connection early
            self._test_ollama_connection()
        elif provider == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.url = "https://api.openai.com/v1/chat/completions"
        elif provider == 'anthropic':
            self.api_key = os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.url = "https://api.anthropic.com/v1/messages"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _test_ollama_connection(self):
        """Test if ollama server is running and accessible."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to ollama server at http://localhost:11434\n"
                "Please start ollama:\n"
                "  1. Run: ollama serve\n"
                "  2. Or start it in the background: ollama serve &\n"
                "  3. Or use the start.sh script which starts it automatically"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                "Ollama server is not responding (timeout after 2s)\n"
                "Please check if ollama is running: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Error connecting to ollama: {e}")

    def is_coherent(self, sentence: str) -> bool:
        """
        Lightweight local validation (no external LLM).
        Accept unless obvious structural issues are found.
        """
        self.checked_count += 1
        s = sentence.strip()
        if not s:
            return False
        # Must have at least 3 words
        if len(s.split()) < 3:
            return False
        # Reject trivial copula fragments
        lower = s.lower()
        if any(
            frag in lower
            for frag in [
                "existe",
                "está presente",
                "se encuentran aquí",
                "es importante",
                "están presentes",
                "se encuentra aquí",
            ]
        ):
            return False
        # Basic check for verb presence
        verb_markers = ["ar ", "er ", "ir ", "a ", "e ", "o ", "an ", "en ", "on ", "ió", "ió ", "ando", "iendo"]
        if not any(marker in lower for marker in verb_markers):
            return False
        self.accepted_count += 1
        return True

    def _check_ollama(self, sentence: str) -> bool:
        """Check coherence using local ollama."""
        prompt = f"""You are a Spanish language validator. Answer YES unless you see a clear error.

Say NO only if you find clear issues such as:
- Missing subject or verb (e.g., "Acondicionado informe", "Análisis verlo")
- Obvious gender/number mismatch ("el casa", "la trabajo", "hogar es amplia")
- Obvious verb conjugation errors
- Nonsensical combinations ("hueso del motor", "zombies pagan", "Almirante Consiste", "Agua cerrará")
- Fragmented or incoherent text that would not be said by a native speaker
- Contains non-Spanish text or non-Latin characters (e.g., Chinese, Korean, emoji)
- Boilerplate or meta text that is not a real sentence

Otherwise, if the sentence is complete Spanish and plausible, answer YES.

Sentence: {sentence}

Answer YES or NO:"""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 10
            }
        }

        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        answer = result['response'].strip().lower()

        is_good = any(word in answer for word in ['yes', 'sí', 'si', 'coherent', 'meaningful'])
        if is_good:
            self.accepted_count += 1
        return is_good

    def generate_openai(self, prompt: str, temperature: float = 0.6, max_tokens: int = 50, model_override: Optional[str] = None) -> str:
        """Generate text using OpenAI (for generation when provider differs)."""
        model = model_override or self.model
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You generate Spanish sentences following instructions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens
        }
        resp = requests.post(
            self.url if self.provider == 'openai' else "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            # Surface the error, but let caller handle fallback
            raise RuntimeError(f"OpenAI generation failed: {resp.text}") from exc
        result = resp.json()
        return result['choices'][0]['message']['content'].strip()

    def generate_anthropic(self, prompt: str, temperature: float = 0.6, max_tokens: int = 100, model_override: Optional[str] = None) -> str:
        """Generate text using Anthropic (for generation when provider differs)."""
        model = model_override or self.model
        headers = {
            "x-api-key": os.getenv('ANTHROPIC_API_KEY'),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        resp = requests.post(
            self.url if self.provider == 'anthropic' else "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Anthropic generation failed: {resp.text}") from exc
        result = resp.json()
        return result['content'][0]['text'].strip()

    def _check_openai(self, sentence: str) -> bool:
        """Check coherence using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        prompt = f"""You are a Spanish language validator. Answer YES unless you see a clear error.

Say NO only if you find clear issues such as:
- Missing subject or verb
- Obvious gender/number mismatch
- Obvious verb conjugation errors
- Nonsensical combinations
- Fragmented or incoherent text that would not be said by a native speaker

If it is borderline or simply uncommon but plausible, say YES.

Sentence: {sentence}

Answer:"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a Spanish language expert. Answer only 'yes' or 'no'."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_completion_tokens": 10
        }

        response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        answer = result['choices'][0]['message']['content'].strip().lower()

        is_good = any(word in answer for word in ['yes', 'sí', 'si', 'coherent', 'meaningful'])
        if is_good:
            self.accepted_count += 1
        return is_good

    def _check_anthropic(self, sentence: str) -> bool:
        """Check coherence using Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

        prompt = f"""You are a Spanish language validator. Answer YES unless you see a clear error.

Say NO only if you find clear issues such as:
- Missing subject or verb
- Obvious gender/number mismatch
- Obvious verb conjugation errors
- Nonsensical combinations
- Fragmented or incoherent text that would not be said by a native speaker

If it is borderline or simply uncommon but plausible, say YES.

Sentence: {sentence}

Answer only 'yes' or 'no'."""

        payload = {
            "model": self.model,
            "max_tokens": 10,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        answer = result['content'][0]['text'].strip().lower()

        is_good = any(word in answer for word in ['yes', 'sí', 'si', 'coherent', 'meaningful'])
        if is_good:
            self.accepted_count += 1
        return is_good

    def get_stats(self) -> dict:
        """Get validation statistics."""
        return {
            'checked': self.checked_count,
            'accepted': self.accepted_count,
            'rejected': self.checked_count - self.accepted_count,
            'acceptance_rate': self.accepted_count / self.checked_count if self.checked_count > 0 else 0
        }
