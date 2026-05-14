"""
Unified LLM Factory — Heterogeneous Compute
==============================================
Supports multiple inference backends for the three-tier architecture:

  GPU:  llama-cpp-python (CUDA)   — Lead Orchestrator
  NPU:  OpenVINO                  — Utility Parser / Recon Agent
  CPU:  Embeddings via sentence-transformers (for RAG)

Also retains legacy support for:
  - Ollama (via LiteLLM or direct ChatOpenAI)
  - OpenAI, Anthropic, Gemini (cloud APIs)
"""

import os
import time
import sys
import uuid
from typing import Optional, List, Any, Dict, Sequence

# ── CUDA Path Injection (PRD 4.1 Stability Fix) ──────────────────────────────
# Injects NVIDIA driver and CUDA paths to resolve DLL loading issues on Windows
if os.name == 'nt':
    print("--- CUDA Path Injection ---")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nvidia_base = os.path.join(project_root, "venv", "Lib", "site-packages", "nvidia")
    
    if os.path.exists(nvidia_base):
        for root, dirs, files in os.walk(nvidia_base):
            if "bin" in dirs:
                bin_path = os.path.abspath(os.path.join(root, "bin"))
                print(f"  Linking: {bin_path}")
                os.add_dll_directory(bin_path)
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

    # Global toolkit paths (if installed)
    paths_to_add = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Windows\System32",
    ]
    for path in paths_to_add:
        if os.path.exists(path):
            print(f"  Linking Toolkit: {path}")
            os.add_dll_directory(path)
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
    print("---------------------------\n")

from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

try:
    from agents.json_grammar import get_grammar_string, create_llama_grammar
    HAS_GRAMMAR = True
except (ImportError, ModuleNotFoundError):
    HAS_GRAMMAR = False


def get_llm(agent_spec=None):
    """
    Factory to return the appropriate LLM based on agent spec or environment.

    Provider dispatch order:
    1. AgentSpec.provider (if provided)
    2. LLM_PROVIDER env var
    3. Default: llamacpp

    Supported providers:
      llamacpp  — llama.cpp with CUDA (GPU)
      openvino  — Intel OpenVINO (NPU/CPU)
      ollama    — Ollama via LiteLLM or direct
      openai    — OpenAI API
      anthropic — Anthropic API
      gemini    — Google Gemini API
    """
    # Extract config from agent_spec or environment
    if agent_spec:
        provider = agent_spec.provider.lower()
        model_name = agent_spec.model
        temperature = agent_spec.temperature
        model_path = getattr(agent_spec, 'model_path', '') or ''
        context_size = getattr(agent_spec, 'context_size', 4096)
        json_grammar = getattr(agent_spec, 'json_grammar', False)
    else:
        provider = os.environ.get("LLM_PROVIDER", "llamacpp").lower()
        model_name = os.environ.get("LOCAL_MODEL_NAME", "")
        temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
        model_path = os.environ.get("GPU_MODEL_PATH", "")
        context_size_str = os.environ.get("GPU_CONTEXT_SIZE", "4096").split('#')[0].split()[0]
        context_size = int(context_size_str)
        json_grammar = False

        # Legacy env var support
        if os.environ.get("USE_LOCAL_LLM", "").lower() == "true" and not provider:
            provider = "ollama"
        elif os.environ.get("USE_GEMINI", "").lower() == "true":
            provider = "gemini"
            model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
        elif not provider:
            provider = "llamacpp"

    # Resolve model_path env vars (${VAR:-default} syntax)
    if model_path.startswith("${"):
        import re
        match = re.match(r'\$\{([^}:]+)(?::-([^}]*))?\}', model_path)
        if match:
            env_val = os.environ.get(match.group(1), match.group(2) or '')
            model_path = env_val

    # Dispatch to the correct backend
    if provider == "llamacpp":
        return _create_llamacpp_model(model_path, context_size, temperature, json_grammar)
    elif provider == "openvino":
        return _create_openvino_model(model_path, temperature)

    elif provider == "google_genai":
        return _create_direct_model("gemini", model_name, temperature)

    # Cloud/Ollama backends — try LiteLLM first, then direct
    try:
        return _create_litellm_model(provider, model_name, temperature)
    except ImportError:
        print("LiteLLM not available, falling back to direct provider clients...")
        return _create_direct_model(provider, model_name, temperature)


# =============================================================================
# llama.cpp (GPU / CUDA)
# =============================================================================

def _create_llamacpp_model(model_path: str, n_ctx: int, temperature: float,
                           json_grammar: bool = False):
    """
    Create an LLM using llama-cpp-python with CUDA backend.

    This is the Lead Orchestrator's primary inference path. Loads a GGUF
    model file directly onto the GPU via llama.cpp's CUDA kernels.
    """
    from llama_cpp import Llama

    # Resolve relative path against project root
    if model_path and not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, model_path)

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"GGUF model not found at: {model_path}\n"
            f"Download a model into models/gpu/ or run setup_environment.ps1"
        )

    n_gpu_layers_str = os.environ.get("GPU_N_GPU_LAYERS", "-1").split('#')[0].split()[0]
    n_gpu_layers = int(n_gpu_layers_str)
    
    max_tokens_str = os.environ.get("GPU_MAX_TOKENS", "2048").split('#')[0].split()[0]
    max_tokens = int(max_tokens_str)

    print(f"Initializing llama.cpp (CUDA): {os.path.basename(model_path)}")
    print(f"  Context: {n_ctx} | GPU layers: {n_gpu_layers} | Max tokens: {max_tokens}")

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        verbose=False,
    )

    # Wrap in a LangChain-compatible interface
    return LlamaCppChatWrapper(
        llm=llm,
        temperature=temperature,
        max_tokens=max_tokens,
        json_grammar=json_grammar,
    )


class LlamaCppChatWrapper(BaseChatModel):
    """
    LangChain-compatible wrapper around llama-cpp-python's Llama class.
    
    Inherits from BaseChatModel to provide full integration with LangGraph
    and LangChain's tool-calling ecosystems.
    """
    llm: Any
    temperature: float = 0.0
    max_tokens: int = 2048
    json_grammar: bool = False
    _grammar: Any = None
    _tools: Sequence[Any] = []

    def __init__(self, llm, temperature: float = 0.0, max_tokens: int = 2048,
                 json_grammar: bool = False, **kwargs):
        super().__init__(llm=llm, temperature=temperature, max_tokens=max_tokens,
                         json_grammar=json_grammar, **kwargs)
        if json_grammar and HAS_GRAMMAR:
            self._grammar = create_llama_grammar(strict=True) # Use strict for stability

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        """Internal generator method for BaseChatModel."""
        import json as _json
        import re as _re

        # Convert LangChain messages to llama.cpp chat format
        chat_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": msg.content or ""})
            elif hasattr(msg, 'type') and msg.type == 'tool':
                chat_messages.append({
                    "role": "user",
                    "content": "[Tool Result: " + str(getattr(msg, 'name', 'tool')) + "]\n" + str(msg.content)
                })
            else:
                chat_messages.append({"role": "user", "content": str(msg.content)})

        completion_kwargs = {
            "messages": chat_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "repeat_penalty": 1.1,
        }

        if self._grammar:
            completion_kwargs["grammar"] = self._grammar

        token_estimate = sum(len(m.get("content", "")) for m in chat_messages) // 4
        print("\n[Orchestrator] Thinking... (~" + str(token_estimate) + " tokens)")
        gen_start = time.time()
        response = self.llm.create_chat_completion(**completion_kwargs)
        duration = time.time() - gen_start
        content = response["choices"][0]["message"].get("content") or ""
        print("[Orchestrator] Inference complete in " + str(round(duration, 1)) + "s")
        if content:
            preview = content[:200].replace("\n", " ")
            print("[Orchestrator] Response: " + preview + "...")

        tool_calls = []
        if self._tools:
            try:
                # Pattern 1: Llama 3.1 native JSON tool calls
                for pattern in [r'(\{"name":\s*"[^"]+",\s*"parameters":\s*\{[^}]*\}\})', r'(\{[^}]*"tool"[^}]*\})']:
                    json_match = _re.search(pattern, content, _re.DOTALL)
                    if json_match:
                        try:
                            data = _json.loads(json_match.group(1))
                            tool_name = data.get("name") or data.get("tool")
                            tool_args = data.get("parameters") or data.get("arguments") or {}
                            if tool_name and any(t.name == tool_name for t in self._tools):
                                print(f"[Tool Call] {tool_name}({tool_args})")
                                tool_id = f"call_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                                tool_calls.append({"name": tool_name, "args": tool_args, "id": tool_id, "type": "tool_call"})
                                break
                        except _json.JSONDecodeError:
                            pass

                # Pattern 3: Action / Action Input (ReAct standard) - Find LAST occurrence to handle repetition
                if not tool_calls:
                    all_actions = list(_re.finditer(r'Action:\s*(\w+)', content))
                    all_inputs = list(_re.finditer(r'Action Input:\s*(\{.*?\})', content, _re.DOTALL))
                    
                    if not all_inputs:
                        all_inputs = list(_re.finditer(r'Action Input:\s*(.+)', content))
                        
                    if all_actions:
                        # Use the last one found if model is repeating itself
                        action_match = all_actions[-1]
                        input_match = all_inputs[-1] if all_inputs else None
                        
                        tool_name = action_match.group(1).strip()
                        raw_input = input_match.group(1).strip() if input_match else "{}"
                        try:
                            tool_args = _json.loads(raw_input)
                        except:
                            tool_args = {"command": raw_input} if tool_name == "run_in_sandbox" else {"query": raw_input}
                        
                        if any(t.name == tool_name for t in self._tools):
                            print(f"[Tool Call] {tool_name}({tool_args})")
                            tool_id = f"call_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                            tool_calls.append({"name": tool_name, "args": tool_args, "id": tool_id, "type": "tool_call"})

                # Pattern 2: Python-style calls
                if not tool_calls:
                    python_match = _re.search(r'(\w+)\((.+?)\)', content)
                    if python_match:
                        tool_name = python_match.group(1).strip()
                        raw_args = python_match.group(2).strip()
                        if any(t.name == tool_name for t in self._tools):
                            args = {}
                            kv_matches = _re.findall(r"(\w+)\s*=\s*['\"](.+?)['\"]", raw_args)
                            if kv_matches:
                                args = dict(kv_matches)
                            else:
                                str_match = _re.search(r"['\"](.+?)['\"]", raw_args)
                                if str_match:
                                    val = str_match.group(1)
                                    if tool_name == "run_in_sandbox": args = {"command": val}
                                    elif tool_name == "search_knowledge_base": args = {"query": val}
                                    elif tool_name == "mark_phase_complete": args = {"reasoning": val}
                                    elif tool_name == "read_file_from_sandbox": args = {"path": val}
                                    else: args = {"input": val}
                            print("[Tool Call] " + tool_name + "(" + str(args) + ")")
                            tool_id = f"call_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                            tool_calls.append({"name": tool_name, "args": args, "id": tool_id, "type": "tool_call"})
            except Exception as e:
                print("[Tool Extract Error] " + str(e))

        if tool_calls:
            message = AIMessage(content=content, tool_calls=tool_calls)
        else:
            if not content:
                content = "(No response generated)"
            message = AIMessage(content=content)

        return ChatResult(generations=[ChatGeneration(message=message)])
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-python"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        """Store tool definitions to enable tool-call parsing."""
        self._tools = tools
        return self

    def with_config(self, **kwargs):
        """Compatibility method for LangGraph."""
        return self


def _load_json_grammar():
    """Load the GBNF grammar for constrained JSON tool-call output."""
    try:
        from llama_cpp import LlamaGrammar
        grammar_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'agents', 'tool_call.gbnf'
        )
        if os.path.exists(grammar_path):
            with open(grammar_path, 'r') as f:
                return LlamaGrammar.from_string(f.read())
    except Exception as e:
        print(f"Warning: Could not load JSON grammar: {e}")
    return None


# =============================================================================
# OpenVINO (NPU / CPU)
# =============================================================================

def _create_openvino_model(model_path: str, temperature: float):
    """
    Create an LLM using Intel OpenVINO for NPU inference.

    This is the Utility/Recon Agent's inference path. Runs a heavily
    quantized INT4 model on the Intel NPU for fast text parsing.
    Falls back to CPU if NPU is unavailable.
    """
    # Resolve relative path
    if model_path and not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, model_path)

    device = os.environ.get("NPU_DEVICE", "NPU")
    max_tokens_str = os.environ.get("NPU_MAX_TOKENS", "1024").split('#')[0].split()[0]
    max_tokens = int(max_tokens_str)

    print(f"Initializing OpenVINO model: {model_path}")
    print(f"  Device: {device} | Max tokens: {max_tokens}")

    try:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer

        model = OVModelForCausalLM.from_pretrained(
            model_path,
            device=device,
            library_name="transformers",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return OpenVINOChatWrapper(
            model=model,
            tokenizer=tokenizer,
            temperature=temperature,
            max_tokens=max_tokens,
            device=device,
        )
    except Exception as e:
        if device == "NPU":
            print(f"NPU initialization failed: {e}")
            print("Falling back to CPU...")
            return _create_openvino_model_cpu(model_path, temperature, max_tokens)
        raise


def _create_openvino_model_cpu(model_path: str, temperature: float, max_tokens: int):
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline
    
    print(f"Falling back to CPU...")
    model = OVModelForCausalLM.from_pretrained(model_path, device="CPU", library_name="transformers")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return OpenVINOChatWrapper(
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_tokens=max_tokens,
        device="CPU",
    )


class OpenVINOChatWrapper(BaseChatModel):
    """
    LangChain-compatible wrapper around an OpenVINO causal LM.
    """
    model: Any
    tokenizer: Any
    temperature: float = 0.0
    max_tokens: int = 1024
    device: str = "NPU"

    def __init__(self, model, tokenizer, temperature: float = 0.0,
                 max_tokens: int = 1024, device: str = "NPU", **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, temperature=temperature,
                         max_tokens=max_tokens, device=device, **kwargs)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        # Build a simple prompt from messages
        prompt_parts = []
        for msg in messages:
            role = getattr(msg, 'type', 'user')
            if role == 'system':
                prompt_parts.append(f"<|system|>\n{msg.content}")
            elif role == 'human':
                prompt_parts.append(f"<|user|>\n{msg.content}")
            elif role == 'ai':
                prompt_parts.append(f"<|assistant|>\n{msg.content}")

        prompt_parts.append("<|assistant|>\n")
        full_prompt = "\n".join(prompt_parts)

        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        content = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "openvino"

    def bind_tools(self, tools, **kwargs):
        """No-op for NPU agent (doesn't use tools directly)."""
        return self

    def with_config(self, **kwargs):
        return self


# =============================================================================
# NPU Summarizer — convenience factory
# =============================================================================

def get_npu_summarizer():
    """
    Get the NPU-backed summarizer for parsing large terminal output.

    Returns an OpenVINO model configured for fast, short-form summarization.
    Falls back to CPU if NPU is unavailable.
    """
    model_path = os.environ.get("NPU_MODEL_PATH", "models/npu/")
    temperature = 0.0

    try:
        return _create_openvino_model(model_path, temperature)
    except Exception as e:
        print(f"Warning: NPU summarizer unavailable: {e}")
        print("Large output parsing will fall back to the GPU model.")
        return None


# =============================================================================
# LiteLLM (Ollama, OpenAI, Anthropic, Gemini)
# =============================================================================

def _create_litellm_model(provider: str, model_name: str, temperature: float):
    """Create an LLM instance using LiteLLM as the universal backend."""
    from langchain_community.chat_models import ChatLiteLLM

    litellm_model = _build_model_string(provider, model_name)

    print(f"Initializing LLM via LiteLLM: {litellm_model}")

    kwargs = {
        "model": litellm_model,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    if provider == "ollama":
        api_base = os.environ.get("LOCAL_API_BASE", "http://localhost:11434")
        if api_base.endswith("/v1"):
            api_base = api_base[:-3]
        kwargs["api_base"] = api_base

    if provider == "gemini":
        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key and not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = google_key

    return ChatLiteLLM(**kwargs)


def _create_direct_model(provider: str, model_name: str, temperature: float):
    """Fallback: create LLM using direct provider clients (no LiteLLM)."""
    if provider == "ollama":
        from langchain_openai import ChatOpenAI
        api_base = os.environ.get("LOCAL_API_BASE", "http://localhost:11434/v1")
        print(f"Initializing local LLM: {model_name} at {api_base}")
        return ChatOpenAI(
            model=model_name,
            base_url=api_base,
            api_key="not-needed",
            temperature=temperature,
            max_tokens=4096
        )

    elif provider == "gemini" or provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        print(f"Initializing Gemini LLM (Direct): {model_name}")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            temperature=temperature,
            max_tokens=4096
        )

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            print(f"Initializing Anthropic LLM: {model_name}")
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=4096
            )
        except ImportError:
            raise ImportError(
                "langchain-anthropic is required for Anthropic models. "
                "Install with: pip install langchain-anthropic"
            )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        print(f"Initializing OpenAI LLM: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=4096
        )

    else:
        from langchain_openai import ChatOpenAI
        print(f"Initializing LLM (OpenAI-compatible): {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature
        )


def _build_model_string(provider: str, model_name: str) -> str:
    """
    Build a LiteLLM-compatible model string.

    LiteLLM expects format like:
      - ollama/gemma4:e4b
      - gpt-4o (OpenAI is default, no prefix needed)
      - anthropic/claude-4-sonnet
    """
    if "/" in model_name:
        return model_name

    prefix_map = {
        "ollama": "ollama",
        "openai": "",
        "anthropic": "anthropic",
        "gemini": "gemini",
    }

    prefix = prefix_map.get(provider, "")
    if prefix:
        return f"{prefix}/{model_name}"
    return model_name


# =============================================================================
# Vision model (unchanged)
# =============================================================================

def get_llm_for_vision():
    """
    Get a vision-capable LLM for multimodal analysis.
    Always uses Gemini Flash for speed/cost on vision tasks.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0)
    except Exception as e:
        print(f"Warning: Could not initialize vision model: {e}")
        return None
