import json
import io
import os
import sys
from typing import List, Dict, Any
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file in project root
import pathlib
project_root = pathlib.Path(__file__).parent.parent  # Go up from app/ to project root
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)  # Load from project root
load_dotenv()  # Fallback to default behavior

# Try to import Anthropic for Claude support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from app.shared.utils.logging_config import get_logger
from app.shared.prompts import (
    # Finance Prompts - Variance Analysis
    VARIANCE_ANALYSIS_22_RULES_SYSTEM_PROMPT,
    VARIANCE_ANALYSIS_SYSTEM_PROMPT,
    # Finance Prompts - Sheet Detection
    SHEET_DETECTION_SYSTEM_PROMPT,
    create_sheet_detection_prompt,
    # Finance Prompts - Account Extraction
    ACCOUNT_EXTRACTION_SYSTEM_PROMPT,
    create_account_extraction_prompt,
    # Finance Prompts - Consolidation
    CONSOLIDATION_SYSTEM_PROMPT,
    # Finance Prompts - Revenue Analysis
    REVENUE_ANALYSIS_SYSTEM_PROMPT,
    create_revenue_analysis_prompt,
)

logger = get_logger(__name__)


class LLMFinancialAnalyzer:
    # Supported models with max token limits
    # To switch providers: set AI_PROVIDER=anthropic or AI_PROVIDER=openai in .env
    # To change model: set ANTHROPIC_MODEL or OPENAI_MODEL in .env
    CLAUDE_MODELS = {
        "claude-opus-4-5-20251101": {"max_tokens": 200000},
        "claude-opus-4-20250514": {"max_tokens": 200000},
        "claude-sonnet-4-20250514": {"max_tokens": 200000},
        "claude-3-5-sonnet-20241022": {"max_tokens": 200000},
        "claude-3-5-haiku-20241022": {"max_tokens": 200000},
    }

    # GPT-5 model configurations
    GPT5_MODELS = {
        "gpt-5": {"max_tokens": 400000},
        "gpt-5-mini": {"max_tokens": 400000},
        "gpt-5-nano": {"max_tokens": 400000},
        "gpt-5.1": {"max_tokens": 400000},
    }

    # Legacy OpenAI model configurations
    LEGACY_MODELS = {
        "gpt-4o": {"max_tokens": 128000},
        "gpt-4o-mini": {"max_tokens": 128000},
        "gpt-4-turbo": {"max_tokens": 128000},
    }

    def __init__(self, model_name: str = "gpt-5", progress_callback=None, initial_progress=0):
        """Initialize LLM analyzer with OpenAI GPT or Claude model."""
        self.progress_callback = progress_callback
        self.current_progress = initial_progress  # Start from the current progress
        self.progress_increment = 1  # Default 1% increment per API call
        # Debug information for cloud deployments
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Environment: {'RENDER' if os.getenv('RENDER') else 'LOCAL'}")

        # Determine AI provider from environment (openai or anthropic)
        self.ai_provider = os.getenv("AI_PROVIDER", "openai").lower()

        # Get model configuration from environment
        if self.ai_provider == "anthropic":
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-20250514")
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            self.model = os.getenv("OPENAI_MODEL", "gpt-5")
            self.api_key = os.getenv("OPENAI_API_KEY")

        # Keep backward compatibility
        self.openai_model = self.model
        self.openai_api_key = self.api_key

        # GPT-5 specific parameters
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "medium")  # low, medium, high, minimal, none
        self.verbosity = os.getenv("OPENAI_VERBOSITY", "medium")  # low, medium, high

        # Service tier for priority processing (auto, default, priority, flex)
        self.service_tier = os.getenv("OPENAI_SERVICE_TIER", "auto")

        # Determine model types
        self.is_claude = self.model.startswith("claude")
        self.is_gpt5 = self.model.startswith("gpt-5")

        # Get max context tokens based on model
        if self.model in self.CLAUDE_MODELS:
            self.max_context_tokens = self.CLAUDE_MODELS[self.model]["max_tokens"]
        elif self.model in self.GPT5_MODELS:
            self.max_context_tokens = self.GPT5_MODELS[self.model]["max_tokens"]
        elif self.model in self.LEGACY_MODELS:
            self.max_context_tokens = self.LEGACY_MODELS[self.model]["max_tokens"]
        else:
            # Fallback default
            self.max_context_tokens = 128000

        # Debug: Show if API key was loaded
        provider_name = "Anthropic" if self.is_claude else "OpenAI"
        if self.api_key:
            logger.info(f"{provider_name} API key loaded: {self.api_key[:10]}...{self.api_key[-4:]}")
        else:
            logger.error(f"{provider_name} API key not found in environment variables")
            logger.info(f".env file path: {env_path}")
            logger.info(f".env file exists: {env_path.exists()}")

        key_env_var = "ANTHROPIC_API_KEY" if self.is_claude else "OPENAI_API_KEY"
        if not self.api_key or self.api_key.startswith("your_"):
            raise ValueError(
                f"{provider_name} API key not found! Please set {key_env_var} in your .env file.\n"
                f"Get your API key from: {'https://console.anthropic.com/settings/keys' if self.is_claude else 'https://platform.openai.com/api-keys'}"
            )

        # Initialize AI client based on provider
        self.openai_client = None
        self.anthropic_client = None

        if self.is_claude:
            # Initialize Anthropic client for Claude
            if not ANTHROPIC_AVAILABLE:
                raise RuntimeError("Anthropic library not installed. Run: pip install anthropic")
            try:
                print(f"Attempting Anthropic client initialization...")
                self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
                print(f"Anthropic client initialized successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
            logger.info(f"Using Claude model: {self.model}")
            logger.info(f"API key configured: {self.api_key[:8]}...{self.api_key[-4:]}")
            logger.info(f"Claude mode enabled")
            logger.info(f"   • Max context tokens: {self.max_context_tokens:,}")
        else:
            # Initialize OpenAI client with comprehensive error handling for deployment environments
            # Increase timeout to 120 seconds for large requests
            client_kwargs = {"api_key": self.api_key, "timeout": 120.0}

            # Try multiple initialization approaches for different environments
            initialization_attempts = [
                lambda: OpenAI(**client_kwargs),
                lambda: OpenAI(api_key=self.api_key, timeout=120.0),  # Explicit API key with timeout
                lambda: self._init_openai_minimal(),  # Minimal initialization for cloud environments
                lambda: self._init_openai_aggressive(),  # Most aggressive approach for stubborn cases
            ]

            last_error = None

            for attempt_num, init_func in enumerate(initialization_attempts, 1):
                try:
                    print(f"Attempting OpenAI client initialization (attempt {attempt_num})...")
                    self.openai_client = init_func()
                    print(f"OpenAI client initialized successfully on attempt {attempt_num}")
                    break
                except TypeError as e:
                    last_error = e
                    error_msg = str(e).lower()
                    print(f"Attempt {attempt_num} failed: {e}")

                    if "proxies" in error_msg:
                        print("   → Issue related to proxy parameter - trying next approach")
                        continue
                    elif "unexpected keyword argument" in error_msg:
                        print("   → Unexpected parameter issue - trying simpler initialization")
                        continue
                    else:
                        print(f"   → Unknown TypeError: {e}")
                        continue
                except Exception as e:
                    last_error = e
                    print(f"Attempt {attempt_num} failed with unexpected error: {e}")
                    continue

            if self.openai_client is None:
                raise RuntimeError(f"Failed to initialize OpenAI client after {len(initialization_attempts)} attempts. Last error: {last_error}")
            logger.info(f"Using OpenAI model: {self.model}")
            logger.info(f"API key configured: {self.api_key[:8]}...{self.api_key[-4:]}")

            # Log GPT-5 specific configuration
            if self.is_gpt5:
                logger.info(f"GPT-5 mode enabled")
                logger.info(f"   • reasoning_effort: {self.reasoning_effort}")
                logger.info(f"   • service_tier: {self.service_tier}")
                logger.info(f"   • Max context tokens: {self.max_context_tokens:,}")

    def _init_openai_minimal(self):
        """Minimal OpenAI initialization for cloud environments that may have issues with advanced parameters."""
        # Clear any proxy-related environment variables that might interfere
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        original_values = {}
        for var in proxy_vars:
            if var in os.environ:
                original_values[var] = os.environ[var]
                del os.environ[var]
                print(f"   → Temporarily cleared {var} environment variable")

        try:
            # Try monkey-patching the OpenAI Client to ignore proxies parameter
            import openai
            original_init = openai.OpenAI.__init__

            def patched_init(self, **kwargs):
                # Remove any proxy-related parameters that might cause issues
                clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['proxies', 'proxy', 'http_client']}
                return original_init(self, **clean_kwargs)

            # Temporarily patch the __init__ method
            openai.OpenAI.__init__ = patched_init

            try:
                client = openai.OpenAI(api_key=self.openai_api_key)
                print("   → Successfully initialized with monkey patch")
            finally:
                # Restore original __init__ method
                openai.OpenAI.__init__ = original_init

            # Restore original environment variables
            for var, value in original_values.items():
                os.environ[var] = value

            return client
        except Exception as e:
            print(f"   → Minimal initialization also failed: {e}")
            # Restore environment variables even if failed
            for var, value in original_values.items():
                os.environ[var] = value
            raise e

    def _init_openai_aggressive(self):
        """Direct API approach bypassing OpenAI client initialization entirely."""
        print("   → Attempting direct API approach bypassing OpenAI client")

        try:
            # Create a minimal client-like object that directly handles API calls
            class DirectOpenAIClient:
                def __init__(self, api_key):
                    self.api_key = api_key
                    self.base_url = "https://api.openai.com/v1"

                def chat_completions_create(self, **kwargs):
                    import httpx

                    # Extract parameters
                    model = kwargs.get('model', 'gpt-4o')
                    messages = kwargs.get('messages', [])
                    temperature = kwargs.get('temperature', 0.1)
                    max_tokens = kwargs.get('max_tokens', 4000)

                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }

                    data = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }

                    # Make direct HTTP request to OpenAI API
                    with httpx.Client(timeout=60.0) as http_client:
                        response = http_client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=data
                        )
                        response.raise_for_status()
                        return response.json()

                @property
                def chat(self):
                    return self

                @property
                def completions(self):
                    return self

                def create(self, **kwargs):
                    return self.chat_completions_create(**kwargs)

            client = DirectOpenAIClient(self.openai_api_key)
            print("   → Successfully created direct API client bypassing OpenAI library")
            return client

        except Exception as e:
            print(f"   → Direct API approach failed: {e}")
            raise e

    # ===========================
    # OpenAI API Methods
    # ===========================
    def _call_openai(self, system_prompt: str, user_prompt: str, retry_count: int = 0, max_retries: int = 3) -> dict:
        """Call OpenAI API with retry logic for rate limits. Supports GPT-5 series with reasoning_effort and verbosity."""
        import time

        try:
            total_chars = len(system_prompt) + len(user_prompt)
            estimated_tokens = total_chars // 4

            print(f" Making OpenAI API call...")
            print(f"      • Model: {self.openai_model}")
            print(f"      • GPT-5 mode: {self.is_gpt5}")
            print(f"      • System prompt: {len(system_prompt):,} chars")
            print(f"      • User prompt: {len(user_prompt):,} chars")
            print(f"      • Estimated tokens: ~{estimated_tokens:,}")

            # Build API call parameters
            api_params = {
                "model": self.openai_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_completion_tokens": 32000
            }

            # GPT-5 doesn't support custom temperature - only default (1) is allowed
            # Only add temperature for non-GPT-5 models
            if not self.is_gpt5:
                api_params["temperature"] = 0.1

            # GPT-5 specific parameters - stored separately to handle API compatibility
            gpt5_params = {}

            # Add service_tier for priority/flex processing
            # - "priority": 40% faster processing (Enterprise, premium pricing)
            # - "flex": 50% cheaper, higher latency (available for gpt-5, o3, o4-mini)
            # - "auto": automatically use best available tier
            # - "default": standard processing
            if self.service_tier and self.service_tier != "default":
                gpt5_params["service_tier"] = self.service_tier
                print(f"      • Service tier: {self.service_tier}")

            # Add GPT-5 specific parameters
            if self.is_gpt5:
                # reasoning_effort: top-level parameter that controls how much reasoning the model does
                # Values: "none", "minimal", "low", "medium", "high" (xhigh only for gpt-5.1-codex-max)
                # Note: gpt-5.1 defaults to "none", all other gpt-5 models default to "medium"
                if self.reasoning_effort:
                    gpt5_params["reasoning_effort"] = self.reasoning_effort
                    print(f"      • Reasoning effort: {self.reasoning_effort}")

            # Try with GPT-5 params first, fallback to basic params if API doesn't support them
            try:
                response = self.openai_client.chat.completions.create(**api_params, **gpt5_params)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # OpenAI library doesn't support GPT-5 params yet, use basic params only
                    print(f"      ⚠️ GPT-5 params not supported by OpenAI library, using basic params")
                    response = self.openai_client.chat.completions.create(**api_params)
                else:
                    raise

            print(f" OpenAI API call completed successfully")
            print(f"      • max_completion_tokens requested: 32,000")

            # Update progress by increment after successful API call (200 response)
            if self.progress_callback:
                self.current_progress = min(self.current_progress + self.progress_increment, 100)
                self.progress_callback(self.current_progress, f"AI analysis in progress (~{estimated_tokens:,} tokens processed)")

            print(f"      • Response type: {type(response)}")

            # Handle both standard OpenAI response and direct API response
            if isinstance(response, dict):
                # Direct API response (JSON dict)
                print(f"      • Processing direct API response")
                if 'choices' not in response or not response['choices']:
                    raise RuntimeError("Direct API returned no choices")

                content = response['choices'][0]['message']['content']
                usage = response.get('usage', {})

                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)

                return {
                    "message": {
                        "content": content
                    },
                    "prompt_eval_count": prompt_tokens,
                    "eval_count": completion_tokens,
                    "total_tokens": usage.get('total_tokens', 0)
                }
            else:
                # Standard OpenAI client response
                print(f"      • Processing standard OpenAI response")
                if not response:
                    raise RuntimeError("OpenAI API returned empty response")

                print(f"      • Has choices: {hasattr(response, 'choices')}")
                if not response.choices or len(response.choices) == 0:
                    raise RuntimeError("OpenAI API returned no choices")

                print(f"      • Choices count: {len(response.choices)}")
                if not response.choices[0].message:
                    raise RuntimeError("OpenAI API returned no message")

                print(f"      • Has message: {hasattr(response.choices[0], 'message')}")
                content = response.choices[0].message.content
                print(f"      • Content type: {type(content)}")
                print(f"      • Content length: {len(content) if content else 0}")

                if content is None:
                    raise RuntimeError("OpenAI API returned None content")

                # Get token counts
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0

                # DEBUG: Check if response was truncated
                finish_reason = response.choices[0].finish_reason if response.choices else None
                print(f"      • Completion tokens: {completion_tokens:,}")
                print(f"      • Finish reason: {finish_reason}")
                if finish_reason == "length":
                    print(f"      ⚠️  WARNING: Response truncated due to max_completion_tokens limit!")
                    print(f"      💡 Need to increase max_completion_tokens beyond 32,000")

                # Return in consistent format
                return {
                    "message": {
                        "content": content
                    },
                    "prompt_eval_count": prompt_tokens,
                    "eval_count": completion_tokens,
                    "total_tokens": total_tokens
                }
        except Exception as e:
            import time
            from openai import RateLimitError, APITimeoutError

            # Handle rate limit errors with retry logic
            if isinstance(e, RateLimitError) and retry_count < max_retries:
                # Extract wait time from error message
                error_msg = str(e)
                wait_time = 15  # Default wait time

                # Try to parse suggested wait time from error message
                if "Please try again in" in error_msg:
                    try:
                        # Extract seconds from "Please try again in 11.65s"
                        import re
                        match = re.search(r'try again in ([\d.]+)s', error_msg)
                        if match:
                            wait_time = float(match.group(1)) + 2  # Add 2 second buffer
                    except:
                        pass

                print(f"  Rate limit hit. Waiting {wait_time:.1f}s before retry {retry_count + 1}/{max_retries}...")
                time.sleep(wait_time)

                # Recursive retry
                return self._call_ai(system_prompt, user_prompt, retry_count + 1, max_retries)

            # Handle timeout errors with retry logic
            if isinstance(e, APITimeoutError) and retry_count < max_retries:
                wait_time = 5  # Short wait for timeouts
                print(f"  Request timed out. Retrying in {wait_time}s (attempt {retry_count + 1}/{max_retries})...")
                time.sleep(wait_time)

                # Recursive retry
                return self._call_ai(system_prompt, user_prompt, retry_count + 1, max_retries)

            # For non-retryable errors or exhausted retries, raise the error
            print(f"   ❌ OpenAI API call failed: {str(e)}")
            print(f"      • Error type: {type(e)}")
            import traceback
            print(f"      • Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def _call_anthropic(self, system_prompt: str, user_prompt: str, retry_count: int = 0, max_retries: int = 3) -> dict:
        """Call Anthropic Claude API with streaming (required for Claude Opus 4.5 long requests)."""
        import time

        try:
            total_chars = len(system_prompt) + len(user_prompt)
            estimated_tokens = total_chars // 4

            print(f" Making Anthropic API call (streaming)...")
            print(f"      • Model: {self.model}")
            print(f"      • System prompt: {len(system_prompt):,} chars")
            print(f"      • User prompt: {len(user_prompt):,} chars")
            print(f"      • Estimated tokens: ~{estimated_tokens:,}")

            # Use streaming for Claude Opus 4.5 (required for long requests)
            content_parts = []
            prompt_tokens = 0
            completion_tokens = 0
            stop_reason = None

            with self.anthropic_client.messages.stream(
                model=self.model,
                max_tokens=32000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            ) as stream:
                for text in stream.text_stream:
                    content_parts.append(text)

                # Get final message for usage stats
                final_message = stream.get_final_message()
                if final_message:
                    prompt_tokens = final_message.usage.input_tokens if final_message.usage else 0
                    completion_tokens = final_message.usage.output_tokens if final_message.usage else 0
                    stop_reason = final_message.stop_reason

            content = "".join(content_parts)
            total_tokens = prompt_tokens + completion_tokens

            print(f" Anthropic API call completed successfully")

            # Update progress by increment after successful API call
            if self.progress_callback:
                self.current_progress = min(self.current_progress + self.progress_increment, 100)
                self.progress_callback(self.current_progress, f"AI analysis in progress (~{estimated_tokens:,} tokens processed)")

            print(f"      • Input tokens: {prompt_tokens:,}")
            print(f"      • Output tokens: {completion_tokens:,}")
            print(f"      • Stop reason: {stop_reason}")

            # Return in consistent format
            return {
                "message": {
                    "content": content
                },
                "prompt_eval_count": prompt_tokens,
                "eval_count": completion_tokens,
                "total_tokens": total_tokens
            }

        except Exception as e:
            error_msg = str(e)

            # Handle rate limit errors with retry logic
            if "rate_limit" in error_msg.lower() and retry_count < max_retries:
                wait_time = 15
                print(f"  Rate limit hit. Retrying in {wait_time}s (attempt {retry_count + 1}/{max_retries})...")
                time.sleep(wait_time)
                return self._call_anthropic(system_prompt, user_prompt, retry_count + 1, max_retries)

            # Handle timeout errors
            if "timeout" in error_msg.lower() and retry_count < max_retries:
                wait_time = 5
                print(f"  Request timed out. Retrying in {wait_time}s (attempt {retry_count + 1}/{max_retries})...")
                time.sleep(wait_time)
                return self._call_anthropic(system_prompt, user_prompt, retry_count + 1, max_retries)

            # For non-retryable errors or exhausted retries, raise the error
            print(f"   ❌ Anthropic API call failed: {str(e)}")
            print(f"      • Error type: {type(e)}")
            import traceback
            print(f"      • Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")

    def _call_ai(self, system_prompt: str, user_prompt: str, retry_count: int = 0, max_retries: int = 3) -> dict:
        """Route AI call to appropriate provider (OpenAI or Anthropic)."""
        if self.is_claude:
            return self._call_anthropic(system_prompt, user_prompt, retry_count, max_retries)
        else:
            return self._call_ai(system_prompt, user_prompt, retry_count, max_retries)

    # ===========================
    # Main entrypoints
    # ===========================
    def analyze_raw_excel_file(
        self,
        excel_bytes: bytes,
        filename: str,
        subsidiary: str,
        config: dict
    ) -> List[Dict[str, Any]]:
        """Analyze raw Excel file focusing on BS Breakdown and PL Breakdown sheets."""
        logger.info(f"===== STARTING RAW EXCEL ANALYSIS FOR {subsidiary} =====")
        logger.info(f"File: {filename}")
        logger.info(f"File Size: {len(excel_bytes):,} bytes ({len(excel_bytes)/1024:.1f} KB)")
        logger.info(f"Model: {self.openai_model}")

        try:
            print(f"\nSTEP 1: Loading Raw Excel Sheets")
            # Get all sheet names and use fuzzy matching
            print(f" Reading Excel file structure...")
            excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
            sheet_names = excel_file.sheet_names
            print(f" Found {len(sheet_names)} sheets: {sheet_names}")

            # Use fuzzy matching to identify BS and PL sheets (no AI needed!)
            print(f" Using fuzzy matching to identify BS and PL sheets...")
            from app.shared.utils.sheet_detection import find_sheet_fuzzy
            bs_sheet = find_sheet_fuzzy(sheet_names, is_balance_sheet=True)
            pl_sheet = find_sheet_fuzzy(sheet_names, is_balance_sheet=False)

            print(f" Fuzzy matching identified sheets:")
            print(f"      • BS Sheet: '{bs_sheet}'")
            print(f"      • PL Sheet: '{pl_sheet}'")

            if not bs_sheet or not pl_sheet:
                raise ValueError(f"Could not identify BS and PL sheets from: {sheet_names}. Please ensure sheets contain 'BS' or 'Balance' and 'PL' or 'Profit' in their names.")

            # Now read the identified sheets
            print(f" Reading BS sheet: '{bs_sheet}'...")
            bs_raw = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=bs_sheet, header=None, dtype=str)
            print(f" BS sheet loaded: {len(bs_raw)} rows, {len(bs_raw.columns)} columns")

            print(f" Reading PL sheet: '{pl_sheet}'...")
            pl_raw = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=pl_sheet, header=None, dtype=str)
            print(f" PL sheet loaded: {len(pl_raw)} rows, {len(pl_raw.columns)} columns")

            print(f"\nSTEP 2: Converting to CSV for AI Analysis")
            print(f" Converting raw Excel data to CSV format...")

            # Convert raw DataFrames to CSV - keep all rows but optimize format
            # Remove completely empty rows and columns to reduce token usage
            bs_clean = bs_raw.dropna(how='all').dropna(axis=1, how='all')
            pl_clean = pl_raw.dropna(how='all').dropna(axis=1, how='all')

            # OPTIMIZE: Filter only relevant account rows to reduce tokens
            print(f" Filtering relevant accounts to reduce token usage...")
            bs_filtered = self._filter_relevant_accounts(bs_clean, is_balance_sheet=True)
            pl_filtered = self._filter_relevant_accounts(pl_clean, is_balance_sheet=False)

            print(f"      • BS rows: {len(bs_clean)} → {len(bs_filtered)} (filtered)")
            print(f"      • PL rows: {len(pl_clean)} → {len(pl_filtered)} (filtered)")

            # Use more compact CSV format but INCLUDE headers so AI can see period names
            bs_csv = bs_filtered.to_csv(index=False, header=True, quoting=1, float_format='%.0f')
            pl_csv = pl_filtered.to_csv(index=False, header=True, quoting=1, float_format='%.0f')

            print(f" CSV conversion complete (optimized format):")
            print(f"      • BS CSV: {len(bs_csv):,} characters (from {len(bs_raw)} rows to {len(bs_filtered)} filtered rows)")
            print(f"      • PL CSV: {len(pl_csv):,} characters (from {len(pl_raw)} rows to {len(pl_filtered)} filtered rows)")
            print(f" Sample of filtered BS accounts (first 5 non-header rows):")
            for idx, row in bs_filtered.iloc[10:15].iterrows():
                print(f"      Row {idx}: {list(row[:3])}...")

            # Debug: Show sample of CSV data
            print(f" Debug: BS CSV sample (first 500 chars):")
            print(f"      {bs_csv[:500]}...")
            print(f" Debug: PL CSV sample (first 500 chars):")
            print(f"      {pl_csv[:500]}...")

            print(f"\nSTEP 3: Creating AI Analysis Prompt")

            # Check if data will exceed token limits and chunk if necessary
            estimated_prompt_length = len(bs_csv) + len(pl_csv) + 10000  # Add system prompt overhead
            estimated_tokens = estimated_prompt_length // 4

            print(f" Token estimation:")
            print(f"      • Estimated prompt length: {estimated_prompt_length:,} characters")
            print(f"      • Estimated input tokens: {estimated_tokens:,}")

            # GPT-4o has 128k context window, but we have 30K TPM limit
            # For 22-rule analysis, we use three-step AI process for large data
            if estimated_tokens > 25000:  # Stay under 30K TPM limit
                print(f"  Data large ({estimated_tokens:,} tokens), using three-step AI process...")
                print(f" Step 1: AI will extract accounts from raw CSV in chunks (< 30K tokens each)")
                print(f" Step 2: AI will consolidate and validate all extracted accounts")
                print(f" Step 3: AI will apply 22 rules to validated account data")
                return self._analyze_with_three_step_process(bs_csv, pl_csv, subsidiary, filename, config)

            prompt = self._create_raw_excel_prompt(bs_csv, pl_csv, subsidiary, filename, config)
            prompt_length = len(prompt)
            print(f" Prompt generation complete:")
            print(f"      • Total prompt length: {prompt_length:,} characters")

            print(f"\nSTEP 4: AI Model Processing")
            response = None
            options = None
            attempt = 1

            try:
                print(f"   🚀 Attempt {attempt}: OpenAI GPT-4o processing")
                print(f" Sending complete raw Excel data to AI...")

                response = self._call_ai(
                    system_prompt=self._get_raw_excel_system_prompt(),
                    user_prompt=prompt
                )

                # Extract token usage information if available
                if response and 'total_tokens' in response:
                    input_tokens = response.get('total_tokens', 0)
                    output_tokens = response.get('eval_count', 0)
                    total_tokens = response.get('total_tokens', 0)
                    print(f" Token Usage:")
                    print(f"      • Input tokens: {input_tokens:,}")
                    print(f"      • Output tokens: {output_tokens:,}")
                    print(f"      • Total tokens: {total_tokens:,}")

                print(f" AI analysis successful on attempt {attempt}")

            except Exception as e:
                print(f"   ❌ AI analysis failed: {str(e)}")
                return [{
                    "subsidiary": subsidiary,
                    "account_code": "SYSTEM_ERROR",
                    "rule_name": "AI Analysis Error",
                    "description": f"Raw Excel AI analysis failed: {str(e)[:100]}...",
                    "details": f"Error processing raw Excel file: {str(e)}",
                    "current_value": 0,
                    "previous_value": 0,
                    "change_amount": 0,
                    "change_percent": 0,
                    "severity": "High",
                    "sheet_type": "Error"
                }]

            print(f"\nSTEP 5: Processing AI Response")
            print(f" Debug: Response type: {type(response)}")
            print(f" Debug: Response keys: {list(response.keys()) if response else 'None'}")

            if not response:
                print(f"   ❌ Response is None or empty")
                raise RuntimeError("OpenAI API returned None response")

            if 'message' not in response:
                print(f"   ❌ No 'message' key in response")
                raise RuntimeError("OpenAI API response missing 'message' key")

            if not response['message']:
                print(f"   ❌ Response message is None")
                raise RuntimeError("OpenAI API response message is None")

            if 'content' not in response['message']:
                print(f"   ❌ No 'content' key in message")
                raise RuntimeError("OpenAI API response missing 'content' key")

            if response['message']['content'] is None:
                print(f"   ❌ Response content is None")
                raise RuntimeError("OpenAI API returned None content")

            result = response['message']['content'] or ""
            response_length = len(result)

            # Extract final token usage from successful response
            total_input_tokens = response.get('prompt_eval_count', 0)
            total_output_tokens = response.get('eval_count', 0)
            total_tokens_used = response.get('total_tokens', 0)

            print(f" Response received successfully:")
            print(f"      • Response length: {response_length:,} characters")
            if total_tokens_used > 0:
                print(f" TOKEN USAGE:")
                print(f"      • Input tokens:  {total_input_tokens:,}")
                print(f"      • Output tokens: {total_output_tokens:,}")
                print(f"      • TOTAL TOKENS:  {total_tokens_used:,}")
                print(f"      • Model: {self.openai_model}")

            print(f"   📝 Response preview: {result[:200]}...")

            # Debug: Print the full AI response
            print(f"\n📄 ===== FULL AI RESPONSE =====")
            print(result)
            print(f"===== END AI RESPONSE =====\n")

            print(f"\n🔍 STEP 6: JSON Parsing & Validation")
            anomalies = self._parse_llm_response(result, subsidiary)

            print(f" Parsing completed successfully:")
            print(f"      • Anomalies detected: {len(anomalies)}")

            print(f"\n🎉 ===== RAW EXCEL AI ANALYSIS COMPLETE FOR {subsidiary} =====")
            print(f"📊 Final Results: {len(anomalies)} anomalies identified")

            # Print comprehensive summary banner
            if total_tokens_used > 0:
                print("\n" + "="*80)
                print("✅ AI ANALYSIS COMPLETE - SUMMARY")
                print("="*80)
                print(f"📄 Subsidiary: {subsidiary}")
                print(f"📊 Anomalies detected: {len(anomalies)}")
                print("")
                print("📊 TOKEN USAGE:")
                print(f"   • Input tokens:  {total_input_tokens:,}")
                print(f"   • Output tokens: {total_output_tokens:,}")
                print(f"   • TOTAL TOKENS:  {total_tokens_used:,}")
                print(f"   • Model:         {self.openai_model}")
                print("="*80 + "\n")

            return anomalies

        except Exception as e:
            print(f"\n❌ Raw Excel analysis failed for '{subsidiary}': {e}")
            return [{
                "subsidiary": subsidiary,
                "account_code": "SYSTEM_ERROR",
                "rule_name": "Raw Excel Analysis Failed",
                "description": f"Failed to analyze raw Excel file: {str(e)[:100]}...",
                "details": f"Raw Excel analysis error: {str(e)}",
                "current_value": 0,
                "previous_value": 0,
                "change_amount": 0,
                "change_percent": 0,
                "severity": "High",
                "sheet_type": "Error"
            }]

    def analyze_financial_data(
        self,
        bs_df: pd.DataFrame,
        pl_df: pd.DataFrame,
        subsidiary: str,
        config: dict
    ) -> List[Dict[str, Any]]:
        logger.info(f"===== STARTING AI ANALYSIS FOR {subsidiary} =====")
        logger.info("Input Data Validation:")
        logger.info(f"   • Balance Sheet: {len(bs_df)} rows, {len(bs_df.columns)} columns")
        logger.info(f"   • Profit & Loss: {len(pl_df)} rows, {len(pl_df.columns)} columns")
        logger.info(f"   • Model: {self.openai_model}")

        # Quick sanity checks (both sheets should be non-empty by the time we get here)
        if pl_df is None or pl_df.empty:
            print("❌ ERROR: Profit & Loss data is empty or None")
            raise ValueError("Profit & Loss data is empty or None")
        if bs_df is None or bs_df.empty:
            print("❌ ERROR: Balance Sheet data is empty or None")
            raise ValueError("Balance Sheet data is empty or None")

        """
        Analyze financial data using OpenAI ChatGPT API to detect anomalies and provide explanations.
        Returns a list of anomaly dictionaries.
        """
        # Step 1: Convert DataFrames to simple CSV format for AI
        print(f"\nSTEP 1: Raw Data Preparation")
        print(f" Converting Excel data to CSV format for AI analysis...")

        # Convert to simple CSV strings that AI can easily read
        bs_csv = bs_df.to_csv(index=False)
        pl_csv = pl_df.to_csv(index=False)

        print(f" Data conversion complete:")
        print(f"      • Balance Sheet: {len(bs_df)} rows, {len(bs_df.columns)} columns")
        print(f"      • P&L: {len(pl_df)} rows, {len(pl_df.columns)} columns")
        print(f"      • Full raw data passed to AI for comprehensive analysis")

        # Step 2: Create analysis prompt with raw data
        print(f"\nSTEP 2: Prompt Generation")
        print(f" Building AI analysis prompt with full Excel data...")
        prompt = self._create_raw_data_prompt(bs_csv, pl_csv, subsidiary, config)
        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4  # Rough estimate: 4 chars per token
        print(f" Prompt generation complete:")
        print(f"      • Prompt length: {prompt_length:,} characters")
        print(f"      • Estimated input tokens: {estimated_tokens:,}")

        # Step 3: AI Model Processing with Fallback Strategy
        print(f"\n🤖 STEP 3: AI Model Processing")
        response = None
        options = None
        attempt = 1

        try:
            print(f"   🚀 Attempt {attempt}: OpenAI GPT processing")
            print(f" Sending request to OpenAI...")

            response = self._call_ai(
                system_prompt=self._get_system_prompt(),
                user_prompt=prompt
            )

            # Extract token usage information if available
            if response and 'total_tokens' in response:
                input_tokens = response.get('total_tokens', 0)
                output_tokens = response.get('eval_count', 0)
                total_tokens = response.get('total_tokens', 0)
                print(f" Token Usage:")
                print(f"      • Input tokens: {input_tokens:,}")
                print(f"      • Output tokens: {output_tokens:,}")
                print(f"      • Total tokens: {total_tokens:,}")

            print(f" AI analysis successful on attempt {attempt}")

        except Exception as e1:
            attempt = 2
            print(f" Attempt 1 failed: {str(e1)[:100]}...")
            print(f"   🚀 Attempt {attempt}: Retry with OpenAI GPT-4o")
            try:
                print(f" Retrying with OpenAI API...")

                response = self._call_ai(
                    system_prompt=self._get_raw_excel_system_prompt(),
                    user_prompt=prompt
                )

                # Extract token usage information if available
                if response and 'total_tokens' in response:
                    input_tokens = response.get('total_tokens', 0)
                    output_tokens = response.get('eval_count', 0)
                    total_tokens = response.get('total_tokens', 0)
                    print(f" Token Usage:")
                    print(f"      • Input tokens: {input_tokens:,}")
                    print(f"      • Output tokens: {output_tokens:,}")
                    print(f"      • Total tokens: {total_tokens:,}")

                print(f" AI analysis successful on attempt {attempt}")

            except Exception as e2:
                attempt = 3
                print(f" Attempt 2 failed: {str(e2)[:100]}...")
                print(f"   🚀 Attempt {attempt}: Final retry with OpenAI GPT-4o")
                try:
                    print(f" Final retry with OpenAI API...")

                    response = self._call_ai(
                        system_prompt=self._get_raw_excel_system_prompt(),
                        user_prompt=prompt
                    )

                    # Extract token usage information if available
                    if response and 'total_tokens' in response:
                        input_tokens = response.get('total_tokens', 0)
                        output_tokens = response.get('eval_count', 0)
                        total_tokens = input_tokens + output_tokens
                        print(f" Token Usage:")
                        print(f"      • Input tokens: {input_tokens:,}")
                        print(f"      • Output tokens: {output_tokens:,}")
                        print(f"      • Total tokens: {total_tokens:,}")

                    print(f" AI analysis successful on attempt {attempt}")

                except Exception as e3:
                    print(f"   ❌ All attempts failed!")
                    print(f"      • Final error: {str(e3)}")
                    print(f"      • Check OpenAI server status and model availability")
                    return [{
                        "subsidiary": subsidiary,
                        "account_code": "SYSTEM_ERROR",
                        "rule_name": "AI Analysis Error",
                        "description": f"AI analysis failed after 3 attempts: {str(e3)[:100]}...",
                        "details": f"All retry strategies exhausted. Last error: {str(e3)}. Check if OpenAI is running and {self.openai_model} model is available.",
                        "current_value": 0,
                        "previous_value": 0,
                        "change_amount": 0,
                        "change_percent": 0,
                        "severity": "High",
                        "sheet_type": "Error"
                    }]

        # Step 4: Response Validation and Parsing
        print(f"\n📄 STEP 4: Response Processing")
        try:
            if not response or 'message' not in response or not response['message'] or 'content' not in response['message']:
                print(f"   ❌ Invalid response structure from OpenAI")
                raise RuntimeError("Empty response payload from OpenAI (no message.content)")

            result = response['message']['content'] or ""
            response_length = len(result)
            estimated_output_tokens = response_length // 4

            # Extract final token usage from successful response
            total_input_tokens = response.get('total_tokens', 0)
            total_output_tokens = response.get('eval_count', 0)
            total_tokens_used = total_input_tokens + total_output_tokens

            print(f" Response received successfully:")
            print(f"      • Response length: {response_length:,} characters")
            print(f"      • Estimated output tokens: {estimated_output_tokens:,}")
            print(f"      • Configuration used: ctx={options.get('num_ctx') if options else 'n/a'}, predict={options.get('num_predict') if options else 'n/a'}")

            if total_tokens_used > 0:
                print(f" FINAL TOKEN SUMMARY:")
                print(f"      • Total Input Tokens: {total_input_tokens:,}")
                print(f"      • Total Output Tokens: {total_output_tokens:,}")
                print(f"      • TOTAL TOKENS USED: {total_tokens_used:,}")
                print(f"      • Model: {self.openai_model}")
                print(f"   📝 Response preview: {result[:200]}...")

            # Debug: Check if response looks like JSON
            stripped = result.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                print(f" Response appears to be JSON array format")
            elif '{' in stripped and '}' in stripped:
                print(f"  Response contains JSON objects but may need format correction")
            else:
                print(f"   🚨 Response does not appear to be JSON format - parsing may fail")

            print(f"\n🔍 STEP 5: JSON Parsing & Validation")
            print(f" Parsing AI response into structured anomaly data...")

            # Debug: Print the full AI response
            print(f"\n📄 ===== FULL AI RESPONSE =====")
            print(result)
            print(f"===== END AI RESPONSE =====\n")

            anomalies = self._parse_llm_response(result, subsidiary)

            print(f" Parsing completed successfully:")
            print(f"      • Anomalies detected: {len(anomalies)}")
            if anomalies:
                print(f"      • Anomaly types: {', '.join(set(a.get('severity', 'Unknown') for a in anomalies))}")

            print(f"\n🎉 ===== AI ANALYSIS COMPLETE FOR {subsidiary} =====")
            print(f"📊 Final Results: {len(anomalies)} anomalies identified")
            if total_tokens_used > 0:
                print(f"🔢 Processing Summary: {total_tokens_used:,} tokens used (FREE with OpenAI)")
            print()
            return anomalies
        except Exception as e:
            # Return a fallback error anomaly instead of empty list
            return [{
                "subsidiary": subsidiary,
                "account_code": "SYSTEM_ERROR",
                "rule_name": "AI Analysis Error",
                "description": f"AI analysis failed: {str(e)[:100]}...",
                "details": "The AI model returned an invalid/empty payload.",
                "current_value": 0,
                "previous_value": 0,
                "change_amount": 0,
                "change_percent": 0,
                "severity": "Low",
                "sheet_type": "Error"
            }]

    def _analyze_with_three_step_process(self, bs_csv, pl_csv, subsidiary, filename, config):
        """
        Three-step AI process for large datasets with chunking to handle 30K TPM limit.

        STEP 1: AI extracts accounts from raw CSV in CHUNKS (each < 30K tokens)
        STEP 2: AI consolidates and validates all extracted account groups
        STEP 3: AI applies 22 rules to the validated grouped data

        This works with ANY Excel format and stays under API rate limits!
        """
        print(f"\n   🎯 THREE-STEP AI ANALYSIS PROCESS (with chunking)")
        print(f"   " + "="*70)

        # ============================================================
        # STEP 1: CHUNKED ACCOUNT EXTRACTION
        # ============================================================
        print(f"\n   📊 STEP 1: Chunked Account Extraction (< 30K tokens per chunk)")
        print(f"   ─────────────────────────────────────────")

        # Split CSV into chunks (targeting 20K tokens per chunk, well under 30K limit)
        bs_chunks = self._split_csv_into_chunks(bs_csv, max_tokens=20000)
        pl_chunks = self._split_csv_into_chunks(pl_csv, max_tokens=20000)

        print(f"   📦 Data split into {len(bs_chunks)} BS chunks + {len(pl_chunks)} PL chunks")

        all_extracted_accounts = []
        step1_total_tokens = 0

        # Process BS chunks
        for i, bs_chunk in enumerate(bs_chunks, 1):
            print(f" Processing BS chunk {i}/{len(bs_chunks)}...")
            chunk_prompt = self._create_chunk_extraction_prompt(bs_chunk, "", subsidiary, filename, f"BS Chunk {i}")

            chunk_response = self._call_ai(
                system_prompt=self._get_account_extraction_system_prompt(),
                user_prompt=chunk_prompt
            )

            if chunk_response and 'message' in chunk_response:
                extracted = self._parse_extraction_response(chunk_response['message']['content'])
                all_extracted_accounts.extend(extracted.get('accounts', []))
                step1_total_tokens += chunk_response.get('total_tokens', 0)
                print(f"      ✅ Extracted {len(extracted.get('accounts', []))} accounts")

        # Process PL chunks
        for i, pl_chunk in enumerate(pl_chunks, 1):
            print(f" Processing PL chunk {i}/{len(pl_chunks)}...")
            chunk_prompt = self._create_chunk_extraction_prompt("", pl_chunk, subsidiary, filename, f"PL Chunk {i}")

            chunk_response = self._call_ai(
                system_prompt=self._get_account_extraction_system_prompt(),
                user_prompt=chunk_prompt
            )

            if chunk_response and 'message' in chunk_response:
                extracted = self._parse_extraction_response(chunk_response['message']['content'])
                all_extracted_accounts.extend(extracted.get('accounts', []))
                step1_total_tokens += chunk_response.get('total_tokens', 0)
                print(f"      ✅ Extracted {len(extracted.get('accounts', []))} accounts")

        print(f" Step 1 complete: {len(all_extracted_accounts)} account entries from {len(bs_chunks) + len(pl_chunks)} chunks")
        print(f"      • Total Step 1 tokens: {step1_total_tokens:,}")

        # DEBUG: Show sample of extracted accounts
        if all_extracted_accounts:
            print(f" Debug: Sample extracted accounts (first 3):")
            for acc in all_extracted_accounts[:3]:
                print(f"      {acc}")
        else:
            print(f"  Warning: No accounts were extracted from any chunks!")

        # ============================================================
        # STEP 2: CONSOLIDATION & VALIDATION
        # ============================================================
        print(f"\n   🔍 STEP 2: Consolidation & Validation")
        print(f"   ─────────────────────────────────────────")
        print(f" Consolidating {len(all_extracted_accounts)} extracted account entries...")

        consolidation_prompt = self._create_consolidation_prompt(all_extracted_accounts, subsidiary, filename)

        try:
            step2_response = self._call_ai(
                system_prompt=self._get_consolidation_system_prompt(),
                user_prompt=consolidation_prompt
            )

            if not step2_response or 'message' not in step2_response:
                raise RuntimeError("Step 2 failed: No response from AI")

            consolidated_text = step2_response['message']['content']

            # DEBUG: Show what AI returned
            print(f" Debug: Step 2 response length: {len(consolidated_text)} chars")
            print(f" Debug: Step 2 response preview (first 1000 chars):")
            print(f"      {consolidated_text[:1000]}")

            grouped_accounts = self._parse_extraction_response(consolidated_text)

            step2_tokens = step2_response.get('total_tokens', 0)

            print(f" Step 2 complete: Validated account groups")
            bs_accounts = grouped_accounts.get('bs_accounts', {})
            pl_accounts = grouped_accounts.get('pl_accounts', {})
            print(f"      • BS accounts: {len(bs_accounts)}")
            print(f"      • PL accounts: {len(pl_accounts)}")
            print(f"      • Months: {grouped_accounts.get('months', [])}")
            print(f"      • Step 2 tokens: {step2_tokens:,}")

            # Print detailed account breakdown
            print(f"\n   📊 Consolidated BS Accounts:")
            for acc_code, acc_data in list(bs_accounts.items())[:10]:  # Show first 10
                acc_name = acc_data.get('name', 'Unknown')
                values = acc_data.get('values', [])
                if values:
                    print(f"      • {acc_code} - {acc_name}: {len(values)} months, Latest: {values[-1] if values else 0:,.0f} VND")
            if len(bs_accounts) > 10:
                print(f"      ... and {len(bs_accounts) - 10} more BS accounts")

            print(f"\n   📊 Consolidated PL Accounts:")
            for acc_code, acc_data in list(pl_accounts.items())[:10]:  # Show first 10
                acc_name = acc_data.get('name', 'Unknown')
                values = acc_data.get('values', [])
                if values:
                    print(f"      • {acc_code} - {acc_name}: {len(values)} months, Latest: {values[-1] if values else 0:,.0f} VND")
            if len(pl_accounts) > 10:
                print(f"      ... and {len(pl_accounts) - 10} more PL accounts")

        except Exception as e:
            print(f"   ❌ Step 2 failed: {str(e)}")
            return [{
                "File": subsidiary,
                "Rule_ID": "STEP2_ERROR",
                "Priority": "🔴 Critical",
                "Issue": "Account Consolidation Failed",
                "Accounts": "N/A",
                "Period": "Current",
                "Reason": f"AI consolidation failed: {str(e)[:100]}",
                "Flag_Trigger": "STEP2_ERROR"
            }]

        # ============================================================
        # STEP 3: AI RULE APPLICATION
        # ============================================================
        print(f"\n   🎯 STEP 3: Applying 22 Variance Analysis Rules")
        print(f"   ─────────────────────────────────────────")
        print(f" Sending consolidated accounts to AI for rule analysis...")

        step3_prompt = self._create_rule_application_prompt(grouped_accounts, subsidiary, filename, config)

        try:
            step3_response = self._call_ai(
                system_prompt=self._get_raw_excel_system_prompt(),  # Use the 22-rule prompt
                user_prompt=step3_prompt
            )

            if not step3_response or 'message' not in step3_response:
                raise RuntimeError("Step 3 failed: No response from AI")

            # Parse variance flags
            variances = self._parse_llm_response(step3_response['message']['content'], subsidiary)

            step3_tokens = step3_response.get('total_tokens', 0)

            print(f" Step 3 complete: {len(variances)} variance flags detected")
            print(f"      • Step 3 tokens: {step3_tokens:,}")

            # Print detailed variance breakdown
            if variances:
                print(f"\n   🚨 Variance Flags Found:")
                for i, var in enumerate(variances[:10], 1):  # Show first 10
                    rule_id = var.get('Rule_ID', var.get('analysis_type', 'Unknown'))
                    priority = var.get('Priority', var.get('severity', 'Unknown'))
                    issue = var.get('Issue', var.get('description', 'No description'))
                    print(f"      {i}. [{priority}] {rule_id}: {issue[:80]}")
                if len(variances) > 10:
                    print(f"      ... and {len(variances) - 10} more variances")
            else:
                print(f"\n   ℹ️  No variance flags detected")
                print(f"      This could mean:")
                print(f"      • All 22 rules passed (good financial hygiene)")
                print(f"      • AI needs more explicit data (check filtered accounts above)")
                print(f"      • Account filtering removed critical data (verify sample above)")

        except Exception as e:
            print(f"   ❌ Step 3 failed: {str(e)}")
            return [{
                "File": subsidiary,
                "Rule_ID": "STEP3_ERROR",
                "Priority": "🔴 Critical",
                "Issue": "Rule Application Failed",
                "Accounts": "N/A",
                "Period": "Current",
                "Reason": f"AI rule application failed: {str(e)[:100]}",
                "Flag_Trigger": "STEP3_ERROR"
            }]

        # ============================================================
        # SUMMARY
        # ============================================================
        total_tokens = step1_total_tokens + step2_tokens + step3_tokens

        print(f"\n   ✅ THREE-STEP ANALYSIS COMPLETE")
        print(f"   " + "="*70)
        print(f"      • Total variances: {len(variances)}")
        print(f"      • Total API calls: {len(bs_chunks) + len(pl_chunks) + 2}")
        print(f"      • Total tokens: {total_tokens:,}")
        print(f"      • Step 1 (Chunked Extraction): {step1_total_tokens:,} tokens")
        print(f"      • Step 2 (Consolidation): {step2_tokens:,} tokens")
        print(f"      • Step 3 (Rule Application): {step3_tokens:,} tokens")

        return variances

    def _get_sheet_detection_system_prompt(self) -> str:
        """System prompt for detecting which sheets are BS and PL based on sheet names only."""
        return SHEET_DETECTION_SYSTEM_PROMPT

    def _create_sheet_detection_prompt(self, all_sheets_csv, subsidiary, filename):
        """Create prompt for AI to detect which sheets are BS and PL."""
        return create_sheet_detection_prompt(all_sheets_csv, subsidiary, filename)

    def _filter_relevant_accounts(self, df, is_balance_sheet=True):
        """
        Filter DataFrame to keep only relevant account rows for analysis.
        This reduces token usage by 60-80% while keeping all critical accounts.

        Keeps:
        - Header rows (first 10 rows)
        - Rows containing account codes we care about (111, 112, 133, 217, 241, 242, 341, 511, 515, 632, 635, 641, 642)
        - Rows containing totals (Total Assets, Total Liabilities, Total Equity, etc.)
        - Rows with significant values (> 1M VND in any column)
        """
        import re

        if df is None or df.empty:
            return df

        # Always keep header rows (usually in first 10 rows)
        header_rows = df.head(10).copy()

        # Define account patterns we care about for the 22 rules
        if is_balance_sheet:
            account_patterns = [
                r'111', r'112',  # Cash and cash equivalents (multiple rules)
                r'133',          # VAT Input (A3, E3)
                r'217',          # Investment Property (A1, A3)
                r'241',          # CIP (A3)
                r'242',          # Broker Assets (E1, E2)
                r'341',          # Loans (A2, A4)
                r'131',          # AR (E5, E6)
                r'138',          # Other receivables
                r'421',          # Retained earnings (D2)
                r'total', r'tổng',  # Total rows
                r'asset', r'tài sản',  # Asset totals
                r'liability', r'nợ phải trả',  # Liability totals
                r'equity', r'vốn',  # Equity totals
            ]
        else:  # P&L
            account_patterns = [
                r'511',          # Revenue (multiple rules)
                r'515',          # Interest Income (A5, F1)
                r'632',          # COGS/D&A (A1, F3)
                r'635',          # Interest Expense (A2, A4, F2)
                r'641',          # Selling Expense (E4)
                r'642',          # G&A Expense (C1, C2, E4)
                r'total', r'tổng',  # Total rows
                r'revenue', r'doanh thu',  # Revenue totals
                r'expense', r'chi phí',  # Expense totals
                r'profit', r'lợi nhuận',  # Profit totals
            ]

        # Filter rows that match patterns
        filtered_rows = []
        for idx, row in df.iterrows():
            # Convert row to string for pattern matching
            row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))

            # Check if row matches any account pattern
            if any(re.search(pattern, row_str, re.I) for pattern in account_patterns):
                filtered_rows.append(idx)
                continue

            # Check if row has significant numeric values (> 1M VND)
            numeric_values = []
            for val in row:
                if pd.notna(val):
                    try:
                        num_val = float(str(val).replace(',', '').replace(' ', ''))
                        if abs(num_val) > 1_000_000:  # > 1M VND
                            filtered_rows.append(idx)
                            break
                    except:
                        pass

        # Combine header rows and filtered rows
        if filtered_rows:
            body_df = df.loc[filtered_rows]
            result = pd.concat([header_rows, body_df]).drop_duplicates()
            return result.reset_index(drop=True)
        else:
            # If no rows match, return header + first 50 data rows as fallback
            return df.head(60)

    def _get_account_extraction_system_prompt(self) -> str:
        """System prompt for Step 1: Account extraction from raw CSV chunks."""
        return ACCOUNT_EXTRACTION_SYSTEM_PROMPT

    def _create_account_extraction_prompt(self, bs_csv, pl_csv, subsidiary, filename):
        """Create prompt for Step 1: Account extraction."""
        return create_account_extraction_prompt(bs_csv, pl_csv, subsidiary, filename)

    def _create_rule_application_prompt(self, grouped_accounts, subsidiary, filename, config):
        """Create prompt for Step 2: Apply 22 rules to grouped data."""
        import json

        accounts_json = json.dumps(grouped_accounts, indent=2)

        # Use string concatenation instead of f-string to avoid format specifier issues with JSON braces
        prompt = """
22-RULE VARIANCE ANALYSIS REQUEST

Company: """ + subsidiary + """
File: """ + filename + """

=== EXTRACTED & GROUPED ACCOUNT DATA ===
""" + accounts_json + """

=== INSTRUCTIONS ===

You have been provided with pre-extracted and grouped account data.

Apply ALL 22 variance analysis rules (from your system prompt) to this data.

The data is already organized by account families, so you can easily check cross-account relationships:
- A1: Check if "217" increased but "632100001"/"632100002" did not
- A2: Check if "341" increased but "635" did not (with day adjustment)
- A3: Check if "217"/"241" increased but "133" did not
- D1: Check balance sheet equation using total rows
- And so on for all 22 rules...

Return JSON array with variance flags for any rules that triggered.

Use the ACTUAL month names from the "months" array in the period field.

📋 REQUIRED OUTPUT FORMAT:

Return a JSON array where each variance flag has these fields:

{
  "analysis_type": "A1 - Asset capitalized but depreciation not started",
  "account": "217xxx ↔ 632100001/632100002",
  "description": "Investment Property increased but depreciation did not increase",
  "explanation": "Detailed explanation of the variance and why it matters",
  "period": "Jan 2025 → Feb 2025",
  "severity": "Critical",  // Critical, Review, or Info
  "details": "IP increased by 2,000,000 VND but D&A unchanged"
}

The "analysis_type" field should match the rule ID and name from the 22 rules above."""

        return prompt

    def analyze_comprehensive_revenue_impact(
        self,
        excel_bytes: bytes,
        filename: str,
        subsidiary: str,
        config: dict
    ) -> List[Dict[str, Any]]:
        """
        Dedicated comprehensive revenue impact analysis focusing on 511/641/642 accounts.
        Mirrors the functionality of analyze_comprehensive_revenue_impact_from_bytes in core.py.
        """
        print(f"\n🎯 ===== COMPREHENSIVE REVENUE IMPACT ANALYSIS =====")
        print(f"📁 File: {filename}")
        print(f"🏢 Subsidiary: {subsidiary}")
        print(f"🤖 AI Model: {self.openai_model}")

        try:
            # Step 1: Load and prepare Excel data
            print(f"\n📊 STEP 1: Excel Data Loading & Preparation")
            print(f" Loading Excel file from bytes...")

            bs_raw, pl_raw = self._load_excel_sheets(excel_bytes)
            bs_clean, pl_clean = self._clean_data_for_ai(bs_raw, pl_raw, subsidiary)

            # Step 2: Convert to CSV for AI analysis
            print(f"\nSTEP 2: CSV Conversion for AI Processing")
            bs_csv = bs_clean.to_csv(index=False, header=True, quoting=1, float_format='%.0f')
            pl_csv = pl_clean.to_csv(index=False, header=True, quoting=1, float_format='%.0f')

            print(f" CSV conversion complete:")
            print(f"      • BS CSV: {len(bs_csv):,} characters")
            print(f"      • PL CSV: {len(pl_csv):,} characters")

            # Step 3: Create specialized revenue analysis prompt
            print(f"\nSTEP 3: Creating Comprehensive Revenue Analysis Prompt")
            prompt = self._create_revenue_analysis_prompt(bs_csv, pl_csv, subsidiary, filename, config)
            prompt_length = len(prompt)
            print(f" Prompt generation complete:")
            print(f"      • Total prompt length: {prompt_length:,} characters")

            # Step 4: AI Model Processing
            print(f"\nSTEP 4: AI Revenue Analysis Processing")
            try:
                print(f"   🚀 Sending comprehensive revenue analysis request to AI...")

                response = self._call_ai(
                    system_prompt=self._get_revenue_analysis_system_prompt(),
                    user_prompt=prompt
                )

                # Extract token usage information
                if response and 'total_tokens' in response:
                    input_tokens = response.get('total_tokens', 0)
                    output_tokens = response.get('eval_count', 0)
                    total_tokens = response.get('total_tokens', 0)
                    print(f" Token Usage:")
                    print(f"      • Input tokens: {input_tokens:,}")
                    print(f"      • Output tokens: {output_tokens:,}")
                    print(f"      • Total tokens: {total_tokens:,}")

                print(f" AI revenue analysis successful")

            except Exception as e:
                print(f"   ❌ AI revenue analysis failed: {str(e)}")
                return [{
                    "subsidiary": subsidiary,
                    "analysis_type": "system_error",
                    "account": "SYSTEM_ERROR",
                    "description": f"Comprehensive revenue analysis failed: {str(e)[:100]}...",
                    "explanation": f"Error processing comprehensive revenue analysis: {str(e)}",
                    "current_value": 0,
                    "previous_value": 0,
                    "change_amount": 0,
                    "change_percent": 0,
                    "severity": "High"
                }]

            # Step 5: Process AI Response
            print(f"\nSTEP 5: Processing AI Revenue Analysis Response")

            if not response or 'message' not in response or not response['message'] or 'content' not in response['message']:
                print(f"   ❌ Invalid response structure from OpenAI")
                raise RuntimeError("Empty response payload from OpenAI")

            result = response['message']['content'] or ""
            response_length = len(result)

            print(f" Response received successfully:")
            print(f"      • Response length: {response_length:,} characters")

            # Debug: Print the full AI response
            print(f"\n📄 ===== FULL AI REVENUE ANALYSIS RESPONSE =====")
            print(result)
            print(f"===== END AI RESPONSE =====\n")

            # Parse the comprehensive revenue analysis response
            revenue_analysis = self._parse_revenue_analysis_response(result, subsidiary)

            print(f" Parsing completed successfully:")
            print(f"      • Analysis items generated: {len(revenue_analysis)}")

            print(f"\n🎉 ===== COMPREHENSIVE REVENUE ANALYSIS COMPLETE =====")
            print(f"📊 Final Results: {len(revenue_analysis)} analysis items")

            return revenue_analysis

        except Exception as e:
            print(f"\n❌ Comprehensive revenue analysis failed: {str(e)}")
            return [{
                "subsidiary": subsidiary,
                "analysis_type": "system_error",
                "account": "SYSTEM_ERROR",
                "description": f"Comprehensive revenue analysis failed: {str(e)[:100]}...",
                "explanation": f"System error during comprehensive revenue analysis: {str(e)}",
                "current_value": 0,
                "previous_value": 0,
                "change_amount": 0,
                "change_percent": 0,
                "severity": "High"
            }]

    # ===========================
    # Data preparation (more permissive)
    # ===========================
    def _prepare_data_summary(
        self,
        bs_df: pd.DataFrame,
        pl_df: pd.DataFrame,
        subsidiary: str,
        config: dict
    ) -> Dict[str, Any]:
        """Prepare financial data summary for LLM analysis - passes all account data to AI."""
        _ = config  # AI-only mode doesn't use manual configuration
        print(f"      🔄 Starting data preparation for {subsidiary}")
        print(f"      📊 AI-only mode: All accounts with data will be passed to AI for analysis")

        summary = {
            "subsidiary": subsidiary,
            "balance_sheet": {},
            "profit_loss": {}
        }

        def include_line(prev_val, curr_val):
            """Include any account that has data - AI will determine significance."""
            prev_val = 0 if pd.isna(prev_val) else float(prev_val)
            curr_val = 0 if pd.isna(curr_val) else float(curr_val)
            change = curr_val - prev_val
            if prev_val == 0:
                pct = 100.0 if curr_val != 0 else 0.0
            else:
                pct = (change / abs(prev_val)) * 100

            # Include any account that has any data (current or previous values)
            has_data = (prev_val != 0 or curr_val != 0)
            return has_data, change, pct

        # ---------- Balance Sheet ----------
        print(f"      🏦 Processing Balance Sheet data...")
        if bs_df is not None and not bs_df.empty:
            periods = bs_df.columns[1:] if len(bs_df.columns) > 1 else []
            print(f"         • Available periods: {len(periods)} ({', '.join(periods[:3])}{'...' if len(periods) > 3 else ''})")
            if len(periods) >= 2:
                current = periods[-1]; previous = periods[-2]
                print(f"         • Comparing: {previous} → {current}")
                summary["balance_sheet"] = {"periods": [str(previous), str(current)], "accounts": {}}
                total_bs_accounts = 0
                included_bs_accounts = 0
                for _, row in bs_df.iterrows():
                    account = row.iloc[0]
                    if pd.notna(account):
                        total_bs_accounts += 1
                        prev_val = row[previous] if previous in row.index else 0
                        curr_val = row[current]  if current  in row.index else 0
                        ok, change, pct = include_line(prev_val, curr_val)
                        if ok:
                            included_bs_accounts += 1
                            summary["balance_sheet"]["accounts"][str(account)] = {
                                "previous": float(prev_val) if pd.notna(prev_val) else 0.0,
                                "current":  float(curr_val) if pd.notna(curr_val) else 0.0,
                                "change":   float(change),
                                "change_percent": float(pct) if abs(pct) < 10000 else 0.0
                            }
                print(f"         ✅ BS processing: {included_bs_accounts}/{total_bs_accounts} accounts with data included")
            else:
                print(f"         ⚠️ Insufficient periods for comparison")

        # ---------- Profit & Loss ----------
        print(f"      💰 Processing Profit & Loss data...")
        if pl_df is not None and not pl_df.empty:
            periods = pl_df.columns[1:] if len(pl_df.columns) > 1 else []
            print(f"         • Available periods: {len(periods)} ({', '.join(periods[:3])}{'...' if len(periods) > 3 else ''})")
            if len(periods) >= 2:
                current = periods[-1]; previous = periods[-2]
                print(f"         • Comparing: {previous} → {current}")
                summary["profit_loss"] = {"periods": [str(previous), str(current)], "accounts": {}}
                total_pl_accounts = 0
                included_pl_accounts = 0
                revenue_accounts = 0
                utilities_accounts = 0
                interest_accounts = 0

                for _, row in pl_df.iterrows():
                    account = row.iloc[0]
                    if pd.notna(account):
                        total_pl_accounts += 1
                        account_str = str(account)

                        # Track key account types
                        if account_str.startswith('511'):
                            revenue_accounts += 1
                        elif account_str.startswith(('627', '641')):
                            utilities_accounts += 1
                        elif account_str.startswith(('515', '635')):
                            interest_accounts += 1

                        prev_val = row[previous] if previous in row.index else 0
                        curr_val = row[current]  if current  in row.index else 0
                        ok, change, pct = include_line(prev_val, curr_val)
                        if ok:
                            included_pl_accounts += 1
                            summary["profit_loss"]["accounts"][str(account)] = {
                                "previous": float(prev_val) if pd.notna(prev_val) else 0.0,
                                "current":  float(curr_val) if pd.notna(curr_val) else 0.0,
                                "change":   float(change),
                                "change_percent": float(pct) if abs(pct) < 10000 else 0.0
                            }

                print(f"         ✅ P&L processing: {included_pl_accounts}/{total_pl_accounts} accounts with data included")
                print(f"         📊 Key account types found:")
                print(f"            • Revenue (511*): {revenue_accounts} accounts")
                print(f"            • Utilities (627*/641*): {utilities_accounts} accounts")
                print(f"            • Interest (515*/635*): {interest_accounts} accounts")
            else:
                print(f"         ⚠️ Insufficient periods for comparison")

        print(f"      ✅ Data preparation complete for {subsidiary}")
        return summary

    # ===========================
    # Prompts for Raw Excel Analysis
    # ===========================
    def _get_raw_excel_system_prompt(self) -> str:
        """System prompt for 22-rule variance analysis using AI."""
        return VARIANCE_ANALYSIS_22_RULES_SYSTEM_PROMPT

    def _create_raw_excel_prompt(self, bs_csv: str, pl_csv: str, subsidiary: str, filename: str, config: dict) -> str:
        """Create analysis prompt with complete raw Excel data."""
        _ = config  # AI determines all parameters autonomously

        return f"""
COMPLETE RAW EXCEL FINANCIAL ANALYSIS

Company: {subsidiary}
File: {filename}
Analysis Type: Comprehensive anomaly detection on raw Excel data

=== RAW BALANCE SHEET DATA (BS Breakdown Sheet) ===
{bs_csv}

=== RAW PROFIT & LOSS DATA (PL Breakdown Sheet) ===
{pl_csv}

=== ANALYSIS INSTRUCTIONS ===

You are analyzing the COMPLETE raw Excel data above. This includes all formatting, headers, account codes, and financial values exactly as they appear in the original Excel file.

🎯 ANALYSIS FOCUS:
1. AUTOMATIC ACCOUNT DETECTION:
   - Scan raw CSV for account patterns: "(111)", "(112)", "1. Tien (111)", "511000000", etc.
   - Extract account codes and match with descriptive names
   - Build account-to-value mappings from the raw Excel structure

2. PERIOD COLUMN IDENTIFICATION:
   - Find period headers like "As of Jan 2025", "As of Feb 2025", etc.
   - Identify which columns contain actual financial values (not zeros)
   - Focus on the rightmost 2-3 columns with meaningful data

3. VALUE EXTRACTION AND ANALYSIS:
   - Extract actual VND amounts for each detected account
   - Calculate month-over-month changes automatically
   - Focus on Vietnamese Chart of Accounts: 511* (Revenue), 627*/641* (Utilities), 515*/635* (Interest)
   - Identify material anomalies based on account patterns and changes

📊 DETAILED ANALYSIS STEPS:
1. PARSE RAW STRUCTURE: Understand the Excel layout from CSV data
2. EXTRACT ACCOUNTS: Find all account codes and names automatically
   - Balance Sheet: Look for "(111)", "(112)", "(120)" pattern accounts
   - P&L: Look for "511000000", "627000000", "641000000" pattern accounts
3. IDENTIFY PERIODS: Find the latest financial periods with data
   - Look for column headers like "As of Jan 2025", "As of Feb 2025", "Jan 2025", "Feb 2025"
   - Extract the EXACT period name from the header (including "As of" if present)
   - The "period" field in JSON output MUST use this exact period name from the current month column
4. CALCULATE CHANGES: Compute absolute and percentage changes between periods
5. APPLY MATERIALITY: Focus on accounts >100M VND or changes >15%
6. DETECT ANOMALIES: Identify unusual patterns requiring audit attention

🎯 SPECIFIC ACCOUNT PATTERNS TO DETECT:
- Cash accounts: "Tien (111)", "Cac khoan tuong duong tien (112)"
- Revenue accounts: "511000000 - Revenue from sale and service provision"
- Expense accounts: "627000000 - Cost of goods sold", "641000000 - Sales expenses"
- Interest accounts: "515000000 - Financial income", "635000000 - Financial expenses"

💡 CONTEXT AWARENESS:
- Vietnamese business environment (Tet holidays, regulatory changes)
- Seasonal patterns in revenue and expenses
- Industry-specific considerations
- Related account relationships (e.g., revenue vs utilities scaling)

🚨 CRITICAL INSTRUCTION: You are analyzing real financial data. There WILL be variance patterns to detect. Do NOT return an empty array unless there is literally no numerical data in the sheets. Analyze every account with values and identify at least 3-5 significant patterns, changes, or anomalies.

Return detailed JSON analysis with specific findings from the raw Excel data."""

    # ===========================
    # Prompts (wider hunting)
    # ===========================
    def _get_system_prompt(self) -> str:
        """Enhanced system prompt for specific, actionable financial analysis."""
        return VARIANCE_ANALYSIS_SYSTEM_PROMPT

    def _create_raw_data_prompt(self, bs_csv: str, pl_csv: str, subsidiary: str, config: dict) -> str:
        """Create analysis prompt with full raw Excel data matching Python mode logic."""
        # Extract actual config values
        materiality_vnd = config.get("materiality_vnd", 1000000000)
        bs_pct_threshold = config.get("bs_pct_threshold", 0.05)
        recurring_pct_threshold = config.get("recurring_pct_threshold", 0.05)
        revenue_opex_pct_threshold = config.get("revenue_opex_pct_threshold", 0.10)

        return f"""
FINANCIAL VARIANCE ANALYSIS REQUEST

Company: {subsidiary}
Analysis Type: Rule-based anomaly detection (AI-powered)

=== BALANCE SHEET DATA ===
{bs_csv}

=== PROFIT & LOSS DATA ===
{pl_csv}

=== ANALYSIS INSTRUCTIONS ===

You are a senior Vietnamese financial auditor. Analyze the Excel data following the EXACT same rules as Python mode:

🎯 STEP 1: IDENTIFY ALL MONTH COLUMNS
- Look for columns with month names (Jan 2025, Feb 2025, etc.)
- Identify the LAST TWO months with data
- Calculate month-over-month changes between these two periods

📊 STEP 2: BALANCE SHEET ANALYSIS
Apply these SPECIFIC rules for Balance Sheet accounts:
- **Materiality Threshold**: {materiality_vnd:,} VND
- **Percentage Threshold**: {bs_pct_threshold * 100}%
- **Rule**: Flag if absolute change >= {materiality_vnd:,} VND AND percentage change > {bs_pct_threshold * 100}%
- **Status**: "Needs Review"
- **Trigger**: "BS >{bs_pct_threshold * 100}% & ≥{materiality_vnd/1e9}B"

💰 STEP 3: PROFIT & LOSS ANALYSIS
Apply different rules based on account classification:

**3A. RECURRING ACCOUNTS** (621*, 622*, 623*, 627*, 641*, 642*):
- **Rule**: Flag if absolute change >= {materiality_vnd:,} VND AND percentage change > {recurring_pct_threshold * 100}%
- **Trigger**: "Recurring >{recurring_pct_threshold * 100}% & ≥{materiality_vnd/1e9}B"

**3B. REVENUE/OPEX ACCOUNTS** (511*, 515*, 632*, 635*, and others):
- **Rule**: Flag if percentage change > {revenue_opex_pct_threshold * 100}% OR absolute change >= {materiality_vnd:,} VND
- **Trigger**: "Revenue/OPEX >{revenue_opex_pct_threshold * 100}% or ≥{materiality_vnd/1e9}B"

**3C. DEPRECIATION ACCOUNTS** (214*, 627*):
- **Rule**: Flag if percentage change > {recurring_pct_threshold * 100}% (no materiality requirement)
- **Trigger**: "Depreciation % change > threshold"

🔍 STEP 4: GROSS MARGIN ANALYSIS
- Calculate Gross Margin = (511* Revenue - 632* COGS) / 511* Revenue
- Compare month-over-month gross margin percentage
- Flag significant drops or unusual patterns

📋 STEP 5: VIETNAMESE BUSINESS CONTEXT
For each anomaly, provide SHORT, PRACTICAL explanation:
- What this account typically represents
- Common causes in Vietnamese business environment
- Seasonal factors (Tet, fiscal year-end, monsoon)
- Regulatory considerations (VAT, tax, labor law)

⚠️ CRITICAL REQUIREMENTS:
1. Only flag accounts that meet the SPECIFIC threshold rules above
2. Use the SAME classification logic as Python mode
3. Calculate ACTUAL values from the Excel data (not zeros)
4. Keep explanations SHORT (2-3 sentences max)
5. Return ONLY valid JSON array - no markdown, no commentary

{self._get_system_prompt()}"""
        def cur(x):
            try:
                return float(x.get('current', 0) or 0)
            except Exception:
                return 0.0

        # Sum families from the filtered dict (OK in discovery mode; or switch to raw PL if you pass it here)
        items_iter = getattr(pl_accounts, 'items', lambda: [])()
        revenue_511 = sum(cur(acc) for code, acc in items_iter if str(code).startswith('511'))

        items_iter = getattr(pl_accounts, 'items', lambda: [])()
        utilities_627 = sum(cur(acc) for code, acc in items_iter if str(code).startswith('627'))

        items_iter = getattr(pl_accounts, 'items', lambda: [])()
        utilities_641 = sum(cur(acc) for code, acc in items_iter if str(code).startswith('641'))

        items_iter = getattr(pl_accounts, 'items', lambda: [])()
        interest_income_515 = sum(cur(acc) for code, acc in items_iter if str(code).startswith('515'))
        _ = interest_income_515  # Used in context analysis

        items_iter = getattr(pl_accounts, 'items', lambda: [])()
        interest_expense_635 = sum(cur(acc) for code, acc in items_iter if str(code).startswith('635'))

        # Calculate business ratios for context
        gross_margin = ((revenue_511 - sum(cur(acc) for code, acc in pl_accounts.items() if str(code).startswith('632'))) / revenue_511 * 100) if revenue_511 > 0 else 0
        utility_ratio = ((utilities_627 + utilities_641) / revenue_511 * 100) if revenue_511 > 0 else 0
        interest_coverage = (revenue_511 / interest_expense_635) if interest_expense_635 > 0 else float('inf')

        # Determine company size category
        if revenue_511 < 50_000_000_000:  # < 50B VND
            company_size = "Small Enterprise"
            materiality_suggestion = "50-100M VND"
        elif revenue_511 < 500_000_000_000:  # < 500B VND
            company_size = "Medium Enterprise"
            materiality_suggestion = "200-500M VND"
        else:
            company_size = "Large Enterprise"
            materiality_suggestion = "1-2B VND"

        prompt = f"""🔍 SENIOR AUDITOR VARIANCE ANALYSIS for {data_summary.get('subsidiary','(unknown)')}

📊 BUSINESS CONTEXT & SCALE:
- Company Category: {company_size}
- Total Revenue (511*): {revenue_511:,.0f} VND
- Gross Margin: {gross_margin:.1f}% (revenue minus 632* COGS)
- Utility Efficiency: {utility_ratio:.1f}% of revenue (627* + 641*)
- Interest Coverage: {interest_coverage:.1f}x (revenue/interest expense)
- Suggested Materiality Range: {materiality_suggestion}

🏢 VIETNAMESE BUSINESS ENVIRONMENT CONSIDERATIONS:
- Seasonal patterns (Tet holiday, fiscal year-end, monsoon impacts)
- Regulatory changes (VAT, corporate tax, labor law updates)
- Economic factors (inflation, currency fluctuation, supply chain)
- Industry-specific risks (manufacturing, services, real estate)

📈 BALANCE SHEET ACCOUNTS (period-over-period analysis):
{json.dumps(bs_accounts, indent=2)}

📊 PROFIT & LOSS ACCOUNTS (variance analysis):
{json.dumps(pl_accounts, indent=2)}

🎯 FOCUS AREAS FOR THIS ANALYSIS:
1. Revenue Quality: Timing, recognition, customer concentration
2. Operating Efficiency: Utility costs vs activity levels, margin trends
3. Financial Health: Debt service capacity, working capital management
4. Compliance Risks: Tax positions, related party transactions
5. Management Credibility: Round numbers, estimate changes, one-off items

DELIVER SPECIFIC, ACTIONABLE INSIGHTS. Think like a seasoned Vietnamese auditor who understands local business practices, regulatory environment, and common accounting issues in Vietnamese enterprises.

Return detailed JSON analysis with specific investigation steps and management questions."""
        return prompt

    # ===========================
    # Parsing (unchanged)
    # ===========================
    def _parse_llm_response(self, response: str, subsidiary: str) -> List[Dict[str, Any]]:
        import json

        def _strip_fences(text: str) -> str:
            t = (text or "").strip()
            if t.startswith("```json"):
                t = t[len("```json"):].strip()
                if t.endswith("```"):
                    t = t[:-3]
            elif t.startswith("```"):
                t = t[3:].strip()
                if t.endswith("```"):
                    t = t[:-3]
            return t.strip()

        def _find_bracket_span(text: str):
            start = text.find('[')
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return (start, i + 1)
            return None

        def _parse_attempt(s: str):
            try:
                v = json.loads(s)
                if isinstance(v, list):
                    return v
                return [v]
            except Exception:
                pass
            if (s.startswith('"') and s.endswith('"')) or ('\\"' in s):
                try:
                    unq = json.loads(s)
                    v = json.loads(unq)
                    if isinstance(v, list):
                        return v
                    return [v]
                except Exception:
                    pass
            if '""' in s and '"' in s:
                try:
                    fixed = s.replace('""', '"')
                    v = json.loads(fixed)
                    if isinstance(v, list):
                        return v
                    return [v]
                except Exception:
                    pass
            span = _find_bracket_span(s)
            if span:
                inner = s[span[0]:span[1]]
                return _parse_attempt(inner)
            raise ValueError("Unable to parse JSON from response")

        try:
            text = _strip_fences(response or "")
            try:
                anomalies_raw = _parse_attempt(text)
            except Exception:
                span = _find_bracket_span(text)
                if not span:
                    raise
                sub = text[span[0]:span[1]]
                anomalies_raw = _parse_attempt(sub)

            if anomalies_raw is None:
                anomalies_raw = []
            if not isinstance(anomalies_raw, list):
                anomalies_raw = [anomalies_raw]

            anomalies: List[Dict[str, Any]] = []
            for i, anom in enumerate(anomalies_raw):
                anom = anom or {}
                base_explanation = anom.get("explanation", "") or ""

                # Debug: Log what fields the AI actually returned
                print(f" Debug anomaly {i+1}: Available fields: {list(anom.keys())}")
                print(f"       • current_value: {anom.get('current_value', 'MISSING')}")
                print(f"       • previous_value: {anom.get('previous_value', 'MISSING')}")
                print(f"       • change_amount: {anom.get('change_amount', 'MISSING')}")
                print(f"       • change_percent: {anom.get('change_percent', 'MISSING')}")

                # Keep notes simple and clean for Excel output
                detailed_notes = base_explanation or "AI analysis completed - review variance details"

                # Map severity to priority with emojis (matching Python mode)
                severity = anom.get("severity", "Medium")
                if severity == "High" or severity == "Critical":
                    priority = "🔴 Critical"
                elif severity == "Medium" or severity == "Review":
                    priority = "🟡 Review"
                else:
                    priority = "🟢 Info"

                # Extract rule ID from analysis_type if available
                analysis_type = anom.get("analysis_type", "AI Analysis")
                rule_id = analysis_type.split("-")[0].strip() if "-" in analysis_type else "AI"

                # Format accounts field
                account = anom.get("account", f"AI_DETECTED_{i}")

                # Build reason from description and explanation
                reason = anom.get("description", "AI autonomous anomaly detection")
                if base_explanation and base_explanation != reason:
                    reason = f"{reason}. {base_explanation}"

                # Flag trigger from details
                flag_trigger = anom.get("details", "AI detected variance")
                if len(flag_trigger) > 100:
                    flag_trigger = flag_trigger[:97] + "..."

                # Python mode format
                anomalies.append({
                    "File": subsidiary,  # Map subsidiary to File
                    "Rule_ID": rule_id,
                    "Priority": priority,
                    "Issue": anom.get("description", "AI autonomous anomaly detection"),
                    "Accounts": account,
                    "Period": anom.get("period", "Current"),
                    "Reason": reason,
                    "Flag_Trigger": flag_trigger
                })

            return anomalies

        except Exception:
            return self._create_fallback_analysis(response or "", subsidiary, "GENERAL_PARSE_ERROR")

    def _split_csv_into_chunks(self, csv_data: str, max_tokens: int = 20000) -> List[str]:
        """
        Split CSV data into chunks that stay under max_tokens limit.
        Each chunk includes the header row.
        """
        if not csv_data:
            return []

        lines = csv_data.split('\n')
        if len(lines) <= 1:
            return [csv_data]  # Just header or empty

        header = lines[0]
        data_lines = lines[1:]

        chunks = []
        current_chunk_lines = [header]
        current_tokens = len(header) // 4  # Rough estimate: 4 chars per token

        for line in data_lines:
            line_tokens = len(line) // 4
            if current_tokens + line_tokens > max_tokens and len(current_chunk_lines) > 1:
                # Save current chunk and start new one
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = [header, line]
                current_tokens = (len(header) + len(line)) // 4
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens

        # Add the last chunk
        if len(current_chunk_lines) > 1:
            chunks.append('\n'.join(current_chunk_lines))

        return chunks if chunks else [csv_data]

    def _create_chunk_extraction_prompt(self, bs_csv: str, pl_csv: str, subsidiary: str, filename: str, chunk_name: str) -> str:
        """Create prompt for extracting accounts from a single chunk."""
        return f"""
CHUNKED ACCOUNT EXTRACTION

Company: {subsidiary}
File: {filename}
Chunk: {chunk_name}

Extract all account codes and their values from this data chunk.

{"=== BALANCE SHEET DATA (CHUNK) ===" if bs_csv else ""}
{bs_csv if bs_csv else ""}

{"=== PROFIT & LOSS DATA (CHUNK) ===" if pl_csv else ""}
{pl_csv if pl_csv else ""}

Return JSON array with extracted accounts:
[
  {{
    "account_code": "111000000",
    "account_name": "Cash",
    "type": "BS" or "PL",
    "values": {{"Jan 2025": 1000000, "Feb 2025": 1500000}}
  }}
]
"""

    def _parse_extraction_response(self, response_text: str) -> dict:
        """Parse JSON response from account extraction or consolidation."""
        import json

        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)

            # Handle array format (Step 1 extraction)
            if isinstance(data, list):
                return {"accounts": data}

            # Handle dict format (Step 2 consolidation)
            elif isinstance(data, dict):
                # Check if it's already in the expected format (has bs_accounts, pl_accounts, months)
                if 'bs_accounts' in data or 'pl_accounts' in data:
                    return data
                # Otherwise treat it as accounts array wrapper
                elif 'accounts' in data:
                    return data
                else:
                    return {"accounts": []}

            else:
                return {"accounts": []}

        except json.JSONDecodeError as e:
            print(f"      ❌ JSON parse error: {e}")
            print(f"      📄 Response text (first 500 chars): {cleaned[:500]}")
            return {"accounts": []}

    def _create_consolidation_prompt(self, all_extracted_accounts: List[dict], subsidiary: str, filename: str) -> str:
        """Create prompt for consolidating extracted accounts from multiple chunks."""
        import json

        accounts_json = json.dumps(all_extracted_accounts, indent=2)

        return f"""
ACCOUNT CONSOLIDATION & VALIDATION

Company: {subsidiary}
File: {filename}

You have received account data from multiple chunks. Your task is to:
1. Merge duplicate accounts (same account_code)
2. Combine all values from the same account
3. Detect all available month/period columns
4. Organize into BS and PL categories
5. Return a consolidated structure

=== RAW EXTRACTED ACCOUNTS (from chunks) ===
{accounts_json}

IMPORTANT: Return JSON in this EXACT format:
{{
  "bs_accounts": {{
    "111000000": {{"name": "Cash", "Jan 2025": 1000000, "Feb 2025": 1500000, ...}},
    "217000000": {{"name": "Investment Property", "Jan 2025": 50000000, ...}}
  }},
  "pl_accounts": {{
    "511000000": {{"name": "Revenue", "Jan 2025": 5000000, "Feb 2025": 5500000, ...}},
    "632100002": {{"name": "Depreciation", "Jan 2025": 200000, ...}}
  }},
  "months": ["Jan 2025", "Feb 2025", "Mar 2025", ...]
}}

KEY RULES:
- Use account_code as the key in bs_accounts/pl_accounts
- Include "name" field for each account
- Include all detected month columns with their values
- List all months in chronological order
- Return ONLY valid JSON, no other text
"""

    def _get_consolidation_system_prompt(self) -> str:
        """System prompt for Step 2: Consolidation."""
        return CONSOLIDATION_SYSTEM_PROMPT

    def _create_fallback_analysis(self, response: str, subsidiary: str, error_type: str) -> List[Dict[str, Any]]:
        analysis_content = response[:800] if response else "No response received"
        has_insights = any(kw in (response.lower() if response else "") for kw in
                           ['revenue', 'materiality', 'threshold', 'anomaly', 'analysis', 'significant'])

        if has_insights:
            description = "🤖 AI provided detailed analysis but format needs correction"
        else:
            description = "❌ AI model failed to generate proper analysis - check model availability"

        return [{
            "subsidiary": subsidiary,
            "account_code": f"ERROR_{error_type}",
            "rule_name": "🚨 AI Analysis Error - Check Configuration",
            "description": description,
            "details": f"""🚨 PARSING ERROR DETAILS:

Error Type: {error_type}
Response Length: {len(response) if response else 0} characters
Contains Analysis: {'Yes' if has_insights else 'No'}

📝 RAW AI RESPONSE:
{analysis_content}{'...' if response and len(response) > 800 else ''}

💡 TROUBLESHOOTING:
- Check OpenAI API key validity
- Ensure JSON output compliance
- Try reprocessing / different model / smaller input
- Check OpenAI logs
""",
            "current_value": 0,
            "previous_value": 0,
            "change_amount": 0,
            "change_percent": 0,
            "severity": "High",
            "sheet_type": "Error"
        }]

    # ===========================
    # Dedicated Revenue Analysis Methods
    # ===========================
    def _get_revenue_analysis_system_prompt(self) -> str:
        """Dedicated system prompt for comprehensive revenue impact analysis."""
        return REVENUE_ANALYSIS_SYSTEM_PROMPT

    def _create_revenue_analysis_prompt(self, bs_csv: str, pl_csv: str, subsidiary: str, filename: str, config: dict) -> str:
        """Create specialized prompt for comprehensive revenue impact analysis."""
        _ = config  # AI determines all parameters autonomously
        return create_revenue_analysis_prompt(bs_csv, pl_csv, subsidiary, filename)

    def _parse_revenue_analysis_response(self, response: str, subsidiary: str) -> List[Dict[str, Any]]:
        """Parse the AI response for comprehensive revenue impact analysis."""
        try:
            # Clean the response and parse JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            # Parse JSON array
            import json
            analysis_items = json.loads(cleaned_response)

            # Validate and enhance each item
            enhanced_items = []
            for item in analysis_items:
                if isinstance(item, dict):
                    # Ensure required fields exist
                    enhanced_item = {
                        "subsidiary": subsidiary,
                        "analysis_type": item.get("analysis_type", "general"),
                        "account": item.get("account", "Unknown"),
                        "description": item.get("description", ""),
                        "explanation": item.get("explanation", ""),
                        "current_value": float(item.get("current_value", 0)) if item.get("current_value") else 0,
                        "previous_value": float(item.get("previous_value", 0)) if item.get("previous_value") else 0,
                        "change_amount": float(item.get("change_amount", 0)) if item.get("change_amount") else 0,
                        "change_percent": float(item.get("change_percent", 0)) if item.get("change_percent") else 0,
                        "severity": item.get("severity", "Medium"),
                        "details": item.get("details", {})
                    }
                    enhanced_items.append(enhanced_item)

            return enhanced_items if enhanced_items else self._create_fallback_revenue_analysis(subsidiary)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"   ❌ JSON parsing failed: {str(e)}")
            print(f" Response sample: {response[:500]}...")
            return self._create_fallback_revenue_analysis(subsidiary)

    def _create_fallback_revenue_analysis(self, subsidiary: str) -> List[Dict[str, Any]]:
        """Create fallback analysis when AI response cannot be parsed."""
        return [{
            "subsidiary": subsidiary,
            "analysis_type": "parsing_error",
            "account": "PARSING_ERROR",
            "description": "AI response could not be parsed into comprehensive revenue analysis",
            "explanation": "The AI analysis completed but the response format could not be processed. This may indicate formatting issues in the AI output or complex data that requires manual review.",
            "current_value": 0,
            "previous_value": 0,
            "change_amount": 0,
            "change_percent": 0,
            "severity": "Medium",
            "details": {
                "error_type": "response_parsing",
                "suggestion": "Review raw AI output or reprocess with adjusted parameters"
            }
        }]

          