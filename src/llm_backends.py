"""
LLM Backends Module
===================
Provides unified interface for different LLM backends:
- Mistral-7B-Instruct (with optional 4-bit quantization)
- T5-base / Flan-T5-base
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass
from functools import lru_cache

# PyTorch import with graceful fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    logging.warning("PyTorch not available. LLM backends will not work.")

# Transformers imports
try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForSeq2SeqLM = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
    logging.warning("Transformers not available. LLM backends will not work.")

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    model_name: str


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""
    model_name: str
    device: str = "auto"  # "auto", "cuda", "cpu"
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    use_quantization: bool = False  # For Mistral 4-bit quantization
    load_in_8bit: bool = False  # Alternative quantization


class BaseLLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, context: Optional[str] = None) -> GenerationResult:
        """Generate text given a prompt and optional context."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available."""
        pass


class Mistral7BBackend(BaseLLMBackend):
    """
    Mistral-7B-Instruct backend with optional quantization.

    Falls back to CPU if GPU is not available.
    Supports 4-bit quantization via bitsandbytes if requested.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize Mistral-7B backend.

        Args:
            config: LLM configuration. Uses defaults if not provided.
        """
        resolved_model_name = "mistralai/Mistral-7B-Instruct-v0.1"

        self.config = config or LLMConfig(
            model_name=resolved_model_name,
            device="auto",
            max_new_tokens=256,
            temperature=0.1
        )

        # Override model_name from resolved name, not from passed config alias
        self.config.model_name = resolved_model_name

        self.model = None
        self.tokenizer = None
        self._loaded = False

        logger.info(f"Initializing Mistral-7B backend: {self.config.model_name}")

    def _setup_quantization(self):
        """Set up model quantization if requested."""
        if not TRANSFORMERS_AVAILABLE:
            return None

        try:
            from transformers import BitsAndBytesConfig

            if self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("4-bit quantization enabled")
                return quantization_config
        except ImportError:
            logger.warning("bitsandbytes not available, using full precision")
        return None

    def _load_model(self):
        """Load the model and tokenizer."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch or Transformers not available. Please install: pip install torch transformers")

        if self._loaded:
            return

        try:
            import accelerate
            ACCELERATE_AVAILABLE = True
        except ImportError:
            ACCELERATE_AVAILABLE = False
            logger.warning("accelerate not available, using manual device placement")

        try:
            # Determine device
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with optional quantization
            quantization_config = self._setup_quantization() if self.config.use_quantization else None

            if quantization_config:
                if not ACCELERATE_AVAILABLE:
                    raise ImportError("accelerate is required for quantization. Install with: pip install accelerate")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif ACCELERATE_AVAILABLE:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                # Fallback: load without accelerate, manually move to device
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                if device == "cuda":
                    self.model = self.model.cuda()
                elif device == "cpu":
                    self.model = self.model.to(device)

            self._loaded = True
            logger.info(f"Mistral-7B model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Mistral-7B: {e}")
            self._loaded = False
            raise

    def generate(self, prompt: str, context: Optional[str] = None) -> GenerationResult:
        """
        Generate text with Mistral-7B.

        Args:
            prompt: The user's question/query.
            context: Optional retrieved context to ground the answer.

        Returns:
            GenerationResult with generated text and metadata.
        """
        import time

        self._load_model()

        # Build full prompt
        if context:
            full_prompt = self._build_prompt_with_context(prompt, context)
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"

        start_time = time.time()

        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - self.config.max_new_tokens
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        latency = time.time() - start_time

        return GenerationResult(
            text=generated_text.strip(),
            prompt_tokens=inputs['input_ids'].shape[1],
            completion_tokens=len(outputs[0]) - inputs['input_ids'].shape[1],
            total_tokens=len(outputs[0]),
            latency_seconds=latency,
            model_name=self.get_model_name()
        )

    def _build_prompt_with_context(self, question: str, context: str) -> str:
        """Build a prompt with context for grounding."""
        prompt = f"""<s>[INST] You are a document-grounded assistant. Use ONLY the provided context chunks from the retrieved documents to answer. If the answer is not found in the context, say "I cannot find sufficient evidence in the provided documents." Cite the chunk IDs you used.

Question: {question}

Context Chunks:
{context}

Provide a concise, well-structured answer. Add citations like [C1], [C2] that map to chunk IDs. [/INST]"""
        return prompt

    def get_model_name(self) -> str:
        return self.config.model_name

    def is_available(self) -> bool:
        """Check if the model can be loaded."""
        try:
            self._load_model()
            return self._loaded
        except Exception:
            return False


class T5BaseBackend(BaseLLMBackend):
    """
    T5-base or Flan-T5-base backend.

    More efficient than Mistral-7B, suitable for baseline testing.
    Limited to 512 token input.
    """

    def __init__(self, config: Optional[LLMConfig] = None, use_flant5: bool = True):
        """
        Initialize T5 backend.

        Args:
            config: LLM configuration.
            use_flant5: Whether to use Flan-T5 (instruction-tuned) version.
        """
        resolved_model_name = "google/flan-t5-base" if use_flant5 else "t5-base"

        self.config = config or LLMConfig(
            model_name=resolved_model_name,
            device="auto",
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False  # T5 often works better without sampling
        )

        # Override model_name from resolved name, not from passed config alias
        self.config.model_name = resolved_model_name

        self.model = None
        self.tokenizer = None
        self._loaded = False

        logger.info(f"Initializing T5 backend: {self.config.model_name}")

    def _load_model(self):
        """Load the model and tokenizer."""
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            raise ImportError("PyTorch or Transformers not available. Please install: pip install torch transformers")

        if self._loaded:
            return

        try:
            device = self.config.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

            if device == "cuda":
                self.model = self.model.cuda()
            else:
                self.model = self.model.to(device)

            self._loaded = True
            logger.info(f"T5 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load T5: {e}")
            self._loaded = False
            raise

    def generate(self, prompt: str, context: Optional[str] = None) -> GenerationResult:
        """
        Generate text with T5.

        Args:
            prompt: The question or task description.
            context: Optional context to include.

        Returns:
            GenerationResult with generated text and metadata.
        """
        import time

        self._load_model()

        # Build input for T5 (prefixed task)
        if context:
            # Truncate context to fit within 512 token limit
            full_input = self._build_t5_input(prompt, context)
        else:
            full_input = f"Answer the following question: {prompt}"

        start_time = time.time()

        # Tokenize with truncation
        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=512 - self.config.max_new_tokens,
            padding=True
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        latency = time.time() - start_time

        return GenerationResult(
            text=generated_text.strip(),
            prompt_tokens=inputs['input_ids'].shape[1],
            completion_tokens=len(outputs[0]),
            total_tokens=inputs['input_ids'].shape[1] + len(outputs[0]),
            latency_seconds=latency,
            model_name=self.get_model_name()
        )

    def _build_t5_input(self, question: str, context: str) -> str:
        """Build T5-formatted input with context."""
        # T5 works well with explicit task prefixes
        max_context_len = 350  # Leave room for question and task prefix

        truncated_context = context[:max_context_len] if len(context) > max_context_len else context

        return f"""Given the following context, answer the question. If the answer is not in the context, say "I don't know."

Context: {truncated_context}

Question: {question}

Answer:"""

    def get_model_name(self) -> str:
        return self.config.model_name

    def is_available(self) -> bool:
        """Check if the model can be loaded."""
        try:
            self._load_model()
            return self._loaded
        except Exception:
            return False


class MockLLMBackend(BaseLLMBackend):
    """
    Mock LLM backend for testing without model loading.

    Returns predefined responses for testing purposes.
    """

    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name
        self.responses = {
            "test": "This is a mock response for testing purposes.",
            "default": "Mock response generated."
        }

    def generate(self, prompt: str, context: Optional[str] = None) -> GenerationResult:
        """Generate mock response."""
        import time

        start_time = time.time()

        # Simple mock logic
        if "?" in prompt:
            response_text = f"Based on the context provided, here is my answer to: {prompt[:50]}... [C1]"
        else:
            response_text = "Mock response text."

        return GenerationResult(
            text=response_text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(prompt.split()) + len(response_text.split()),
            latency_seconds=time.time() - start_time,
            model_name=self.model_name
        )

    def get_model_name(self) -> str:
        return self.model_name

    def is_available(self) -> bool:
        return True


def create_llm_backend(
    backend_type: str,
    config: Optional[LLMConfig] = None
) -> BaseLLMBackend:
    """
    Factory function to create LLM backends.

    Args:
        backend_type: One of "mistral7b", "t5base", "flant5", "mock"
        config: Optional LLM configuration.

    Returns:
        Configured LLM backend instance.

    Raises:
        ValueError: If backend_type is not recognized.
    """
    backend_type = backend_type.lower()

    if backend_type == "mistral7b":
        return Mistral7BBackend(config)
    elif backend_type in ("t5base", "t5-base"):
        return T5BaseBackend(config, use_flant5=False)
    elif backend_type in ("flant5", "flan-t5", "flant5base"):
        return T5BaseBackend(config, use_flant5=True)
    elif backend_type == "mock":
        return MockLLMBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. "
                        f"Valid options: mistral7b, t5base, flant5, mock")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test T5 backend (lighter, faster to load)
    print("Testing T5 backend...")
    t5_backend = create_llm_backend("flant5")

    if t5_backend.is_available():
        result = t5_backend.generate(
            "What is RAG?",
            context="RAG stands for Retrieval-Augmented Generation."
        )
        print(f"Model: {result.model_name}")
        print(f"Response: {result.text}")
        print(f"Latency: {result.latency_seconds:.2f}s")
    else:
        print("T5 backend not available")
