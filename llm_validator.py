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
        Check if a Spanish sentence is semantically coherent.
        
        Args:
            sentence: Spanish sentence to validate
            
        Returns:
            True if sentence is coherent, False otherwise
        """
        self.checked_count += 1
        
        try:
            if self.provider == 'ollama':
                return self._check_ollama(sentence)
            elif self.provider == 'openai':
                return self._check_openai(sentence)
            elif self.provider == 'anthropic':
                return self._check_anthropic(sentence)
        except requests.exceptions.Timeout:
            # If timeout, accept sentence (don't block on slow LLM)
            self.accepted_count += 1
            return True
        except Exception as e:
            # If any error, accept sentence (fail open)
            print(f"\n  Warning: LLM validation error: {e}")
            self.accepted_count += 1
            return True
    
    def _check_ollama(self, sentence: str) -> bool:
        """Check coherence using local ollama."""
        prompt = f"""Is this Spanish sentence grammatically correct?

Check for these CRITICAL errors (reject if found):
- Wrong gender agreement (e.g., "el casa", "la trabajo", "hogar es amplia")
- Wrong adjective endings (masculine/feminine mismatch)
- Verb conjugation errors
- Nonsensical word combinations (e.g., "hueso del motor", "zombies pagan")

Answer 'yes' ONLY if the sentence has:
- Correct gender/number agreement throughout
- Proper verb conjugation
- Logical meaning (not random words forced together)

Sentence: {sentence}

Answer:"""
        
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
    
    def _check_openai(self, sentence: str) -> bool:
        """Check coherence using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a Spanish language expert. Answer only 'yes' or 'no'."},
                {"role": "user", "content": f"Is this Spanish sentence semantically coherent and meaningful?\n\nSentence: {sentence}\n\nAnswer:"}
            ],
            "temperature": 0.1,
            "max_tokens": 10
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
        
        payload = {
            "model": self.model,
            "max_tokens": 10,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": f"Is this Spanish sentence semantically coherent and meaningful? Answer only 'yes' or 'no'.\n\nSentence: {sentence}\n\nAnswer:"}
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
