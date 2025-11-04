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

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class LLMFinancialAnalyzer:
    def __init__(self, model_name: str = "gpt-4o", progress_callback=None, initial_progress=0):
        """Initialize LLM analyzer with OpenAI GPT model."""
        self.progress_callback = progress_callback
        self.current_progress = initial_progress  # Start from the current progress
        self.progress_increment = 1  # Default 1% increment per API call
        # Debug information for cloud deployments
        logger.info(f"🔧 Python version: {sys.version}")
        logger.info(f"🔧 Environment: {'RENDER' if os.getenv('RENDER') else 'LOCAL'}")

        # Get OpenAI configuration from environment
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Pricing configuration (USD per 1 million tokens)
        self.input_price_per_million = float(os.getenv("OPENAI_INPUT_PRICE_PER_MILLION", "2.50"))
        self.output_price_per_million = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_MILLION", "10.00"))

        # Debug: Show if API key was loaded
        if self.openai_api_key:
            logger.info(f"✅ OpenAI API key loaded: {self.openai_api_key[:10]}...{self.openai_api_key[-4:]}")
        else:
            logger.error("❌ OpenAI API key not found in environment variables")
            logger.info(f"🔍 .env file path: {env_path}")
            logger.info(f"🔍 .env file exists: {env_path.exists()}")

        if not self.openai_api_key or self.openai_api_key == "your_openai_api_key_here":
            raise ValueError(
                "OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )

        # Initialize OpenAI client with comprehensive error handling for deployment environments
        client_kwargs = {"api_key": self.openai_api_key}

        # Try multiple initialization approaches for different environments
        initialization_attempts = [
            lambda: OpenAI(**client_kwargs),
            lambda: OpenAI(api_key=self.openai_api_key),  # Explicit API key only
            lambda: self._init_openai_minimal(),  # Minimal initialization for cloud environments
            lambda: self._init_openai_aggressive(),  # Most aggressive approach for stubborn cases
        ]

        self.openai_client = None
        last_error = None

        for attempt_num, init_func in enumerate(initialization_attempts, 1):
            try:
                print(f"🔄 Attempting OpenAI client initialization (attempt {attempt_num})...")
                self.openai_client = init_func()
                print(f"✅ OpenAI client initialized successfully on attempt {attempt_num}")
                break
            except TypeError as e:
                last_error = e
                error_msg = str(e).lower()
                print(f"⚠️  Attempt {attempt_num} failed: {e}")

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
                print(f"⚠️  Attempt {attempt_num} failed with unexpected error: {e}")
                continue

        if self.openai_client is None:
            raise RuntimeError(f"Failed to initialize OpenAI client after {len(initialization_attempts)} attempts. Last error: {last_error}")
        logger.info(f"🤖 Using OpenAI model: {self.openai_model}")
        logger.info(f"🔑 API key configured: {self.openai_api_key[:8]}...{self.openai_api_key[-4:]}")

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
    # Cost Calculation
    # ===========================
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> dict:
        """
        Calculate the cost estimate based on token usage and pricing configuration.

        Args:
            prompt_tokens: Number of input/prompt tokens
            completion_tokens: Number of output/completion tokens

        Returns:
            Dictionary with cost breakdown
        """
        input_cost = (prompt_tokens / 1_000_000) * self.input_price_per_million
        output_cost = (completion_tokens / 1_000_000) * self.output_price_per_million
        total_cost = input_cost + output_cost

        return {
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'total_cost': round(total_cost, 6),
            'model': self.openai_model,
            'currency': 'USD'
        }

    # ===========================
    # OpenAI API Methods
    # ===========================
    def _call_openai(self, system_prompt: str, user_prompt: str) -> dict:
        """Call OpenAI API."""
        try:
            total_chars = len(system_prompt) + len(user_prompt)
            estimated_tokens = total_chars // 4

            print(f"   🔄 Making OpenAI API call...")
            print(f"      • Model: {self.openai_model}")
            print(f"      • System prompt: {len(system_prompt):,} chars")
            print(f"      • User prompt: {len(user_prompt):,} chars")
            print(f"      • Estimated tokens: ~{estimated_tokens:,}")

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            print(f"   ✅ OpenAI API call completed successfully")

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
                cost_info = self._calculate_cost(prompt_tokens, completion_tokens)

                return {
                    "message": {
                        "content": content
                    },
                    "prompt_eval_count": prompt_tokens,
                    "eval_count": completion_tokens,
                    "total_tokens": usage.get('total_tokens', 0),
                    "cost": cost_info
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

                # Get token counts and calculate cost
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0
                cost_info = self._calculate_cost(prompt_tokens, completion_tokens)

                # Return in consistent format
                return {
                    "message": {
                        "content": content
                    },
                    "prompt_eval_count": prompt_tokens,
                    "eval_count": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost_info
                }
        except Exception as e:
            print(f"   ❌ OpenAI API call failed: {str(e)}")
            print(f"      • Error type: {type(e)}")
            import traceback
            print(f"      • Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")


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
        logger.info(f"🔍 ===== STARTING RAW EXCEL ANALYSIS FOR {subsidiary} =====")
        logger.info(f"📄 File: {filename}")
        logger.info(f"📏 File Size: {len(excel_bytes):,} bytes ({len(excel_bytes)/1024:.1f} KB)")
        logger.info(f"🤖 Model: {self.openai_model}")

        try:
            print(f"\n📋 STEP 1: Loading Raw Excel Sheets")
            # Get all sheet names and use fuzzy matching
            print(f"   🔄 Reading Excel file structure...")
            excel_file = pd.ExcelFile(io.BytesIO(excel_bytes))
            sheet_names = excel_file.sheet_names
            print(f"   📋 Found {len(sheet_names)} sheets: {sheet_names}")

            # Use fuzzy matching to identify BS and PL sheets (no AI needed!)
            print(f"   🔍 Using fuzzy matching to identify BS and PL sheets...")
            bs_sheet = self._find_sheet_fuzzy(sheet_names, is_balance_sheet=True)
            pl_sheet = self._find_sheet_fuzzy(sheet_names, is_balance_sheet=False)

            print(f"   ✅ Fuzzy matching identified sheets:")
            print(f"      • BS Sheet: '{bs_sheet}'")
            print(f"      • PL Sheet: '{pl_sheet}'")

            if not bs_sheet or not pl_sheet:
                raise ValueError(f"Could not identify BS and PL sheets from: {sheet_names}. Please ensure sheets contain 'BS' or 'Balance' and 'PL' or 'Profit' in their names.")

            # Now read the identified sheets
            print(f"   🔄 Reading BS sheet: '{bs_sheet}'...")
            bs_raw = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=bs_sheet, header=None, dtype=str)
            print(f"   ✅ BS sheet loaded: {len(bs_raw)} rows, {len(bs_raw.columns)} columns")

            print(f"   🔄 Reading PL sheet: '{pl_sheet}'...")
            pl_raw = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=pl_sheet, header=None, dtype=str)
            print(f"   ✅ PL sheet loaded: {len(pl_raw)} rows, {len(pl_raw.columns)} columns")

            print(f"\n📝 STEP 2: Converting to CSV for AI Analysis")
            print(f"   🔄 Converting raw Excel data to CSV format...")

            # Convert raw DataFrames to CSV - keep all rows but optimize format
            # Remove completely empty rows and columns to reduce token usage
            bs_clean = bs_raw.dropna(how='all').dropna(axis=1, how='all')
            pl_clean = pl_raw.dropna(how='all').dropna(axis=1, how='all')

            # OPTIMIZE: Filter only relevant account rows to reduce tokens
            print(f"   🔧 Filtering relevant accounts to reduce token usage...")
            bs_filtered = self._filter_relevant_accounts(bs_clean, is_balance_sheet=True)
            pl_filtered = self._filter_relevant_accounts(pl_clean, is_balance_sheet=False)

            print(f"      • BS rows: {len(bs_clean)} → {len(bs_filtered)} (filtered)")
            print(f"      • PL rows: {len(pl_clean)} → {len(pl_filtered)} (filtered)")

            # Use more compact CSV format but INCLUDE headers so AI can see period names
            bs_csv = bs_filtered.to_csv(index=False, header=True, quoting=1, float_format='%.0f')
            pl_csv = pl_filtered.to_csv(index=False, header=True, quoting=1, float_format='%.0f')

            print(f"   ✅ CSV conversion complete (optimized format):")
            print(f"      • BS CSV: {len(bs_csv):,} characters (from {len(bs_raw)} rows to {len(bs_clean)} rows)")
            print(f"      • PL CSV: {len(pl_csv):,} characters (from {len(pl_raw)} rows to {len(pl_clean)} rows)")

            # Debug: Show sample of CSV data
            print(f"   🔍 Debug: BS CSV sample (first 500 chars):")
            print(f"      {bs_csv[:500]}...")
            print(f"   🔍 Debug: PL CSV sample (first 500 chars):")
            print(f"      {pl_csv[:500]}...")

            print(f"\n📝 STEP 3: Creating AI Analysis Prompt")

            # Check if data will exceed token limits and chunk if necessary
            estimated_prompt_length = len(bs_csv) + len(pl_csv) + 10000  # Add system prompt overhead
            estimated_tokens = estimated_prompt_length // 4

            print(f"   📊 Token estimation:")
            print(f"      • Estimated prompt length: {estimated_prompt_length:,} characters")
            print(f"      • Estimated input tokens: {estimated_tokens:,}")

            # GPT-4o has 128k context window
            # For 22-rule analysis, we use two-step AI process for large data
            if estimated_tokens > 80000:  # Leave buffer for 128k limit
                print(f"   ⚠️  Data large ({estimated_tokens:,} tokens), using two-step AI process...")
                print(f"   📋 Step 1: AI will extract and group accounts from raw CSV")
                print(f"   📋 Step 2: AI will apply 22 rules to grouped account data")
                return self._analyze_with_two_step_process(bs_csv, pl_csv, subsidiary, filename, config)

            prompt = self._create_raw_excel_prompt(bs_csv, pl_csv, subsidiary, filename, config)
            prompt_length = len(prompt)
            print(f"   ✅ Prompt generation complete:")
            print(f"      • Total prompt length: {prompt_length:,} characters")

            print(f"\n🤖 STEP 4: AI Model Processing")
            response = None
            options = None
            attempt = 1

            try:
                print(f"   🚀 Attempt {attempt}: OpenAI GPT-4o processing")
                print(f"   🔄 Sending complete raw Excel data to AI...")

                response = self._call_openai(
                    system_prompt=self._get_raw_excel_system_prompt(),
                    user_prompt=prompt
                )

                # Extract token usage information if available
                if response and 'total_tokens' in response:
                    input_tokens = response.get('total_tokens', 0)
                    output_tokens = response.get('eval_count', 0)
                    total_tokens = response.get('total_tokens', 0)
                    print(f"   📊 Token Usage:")
                    print(f"      • Input tokens: {input_tokens:,}")
                    print(f"      • Output tokens: {output_tokens:,}")
                    print(f"      • Total tokens: {total_tokens:,}")

                print(f"   ✅ AI analysis successful on attempt {attempt}")

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

            print(f"\n📄 STEP 5: Processing AI Response")
            print(f"   🔍 Debug: Response type: {type(response)}")
            print(f"   🔍 Debug: Response keys: {list(response.keys()) if response else 'None'}")

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
            cost_info = response.get('cost', {})

            print(f"   ✅ Response received successfully:")
            print(f"      • Response length: {response_length:,} characters")
            if total_tokens_used > 0:
                print(f"   💰 TOKEN USAGE & COST:")
                print(f"      • Input tokens:  {total_input_tokens:,}")
                print(f"      • Output tokens: {total_output_tokens:,}")
                print(f"      • TOTAL TOKENS:  {total_tokens_used:,}")
                if cost_info:
                    print(f"      • Input cost:    ${cost_info.get('input_cost', 0):.6f}")
                    print(f"      • Output cost:   ${cost_info.get('output_cost', 0):.6f}")
                    print(f"      • TOTAL COST:    ${cost_info.get('total_cost', 0):.6f} USD")
                print(f"      • Model: {self.openai_model}")

            print(f"   📝 Response preview: {result[:200]}...")

            # Debug: Print the full AI response
            print(f"\n📄 ===== FULL AI RESPONSE =====")
            print(result)
            print(f"===== END AI RESPONSE =====\n")

            print(f"\n🔍 STEP 6: JSON Parsing & Validation")
            anomalies = self._parse_llm_response(result, subsidiary)

            print(f"   ✅ Parsing completed successfully:")
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
                print("💰 TOKEN USAGE & COST:")
                print(f"   • Input tokens:  {total_input_tokens:,}")
                print(f"   • Output tokens: {total_output_tokens:,}")
                print(f"   • TOTAL TOKENS:  {total_tokens_used:,}")
                if cost_info:
                    print("")
                    print(f"   • Input cost:    ${cost_info.get('input_cost', 0):.6f}")
                    print(f"   • Output cost:   ${cost_info.get('output_cost', 0):.6f}")
                    print(f"   • TOTAL COST:    ${cost_info.get('total_cost', 0):.6f} USD")
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
        logger.info(f"🔍 ===== STARTING AI ANALYSIS FOR {subsidiary} =====")
        logger.info("📊 Input Data Validation:")
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
        print(f"\n📋 STEP 1: Raw Data Preparation")
        print(f"   🔄 Converting Excel data to CSV format for AI analysis...")

        # Convert to simple CSV strings that AI can easily read
        bs_csv = bs_df.to_csv(index=False)
        pl_csv = pl_df.to_csv(index=False)

        print(f"   ✅ Data conversion complete:")
        print(f"      • Balance Sheet: {len(bs_df)} rows, {len(bs_df.columns)} columns")
        print(f"      • P&L: {len(pl_df)} rows, {len(pl_df.columns)} columns")
        print(f"      • Full raw data passed to AI for comprehensive analysis")

        # Step 2: Create analysis prompt with raw data
        print(f"\n📝 STEP 2: Prompt Generation")
        print(f"   🔄 Building AI analysis prompt with full Excel data...")
        prompt = self._create_raw_data_prompt(bs_csv, pl_csv, subsidiary, config)
        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4  # Rough estimate: 4 chars per token
        print(f"   ✅ Prompt generation complete:")
        print(f"      • Prompt length: {prompt_length:,} characters")
        print(f"      • Estimated input tokens: {estimated_tokens:,}")

        # Step 3: AI Model Processing with Fallback Strategy
        print(f"\n🤖 STEP 3: AI Model Processing")
        response = None
        options = None
        attempt = 1

        try:
            print(f"   🚀 Attempt {attempt}: OpenAI GPT processing")
            print(f"   🔄 Sending request to OpenAI...")

            response = self._call_openai(
                system_prompt=self._get_system_prompt(),
                user_prompt=prompt
            )

            # Extract token usage information if available
            if response and 'total_tokens' in response:
                input_tokens = response.get('total_tokens', 0)
                output_tokens = response.get('eval_count', 0)
                total_tokens = response.get('total_tokens', 0)
                print(f"   📊 Token Usage:")
                print(f"      • Input tokens: {input_tokens:,}")
                print(f"      • Output tokens: {output_tokens:,}")
                print(f"      • Total tokens: {total_tokens:,}")

            print(f"   ✅ AI analysis successful on attempt {attempt}")

        except Exception as e1:
            attempt = 2
            print(f"   ⚠️ Attempt 1 failed: {str(e1)[:100]}...")
            print(f"   🚀 Attempt {attempt}: Retry with OpenAI GPT-4o")
            try:
                print(f"   🔄 Retrying with OpenAI API...")

                response = self._call_openai(
                    system_prompt=self._get_raw_excel_system_prompt(),
                    user_prompt=prompt
                )

                # Extract token usage information if available
                if response and 'total_tokens' in response:
                    input_tokens = response.get('total_tokens', 0)
                    output_tokens = response.get('eval_count', 0)
                    total_tokens = response.get('total_tokens', 0)
                    print(f"   📊 Token Usage:")
                    print(f"      • Input tokens: {input_tokens:,}")
                    print(f"      • Output tokens: {output_tokens:,}")
                    print(f"      • Total tokens: {total_tokens:,}")

                print(f"   ✅ AI analysis successful on attempt {attempt}")

            except Exception as e2:
                attempt = 3
                print(f"   ⚠️ Attempt 2 failed: {str(e2)[:100]}...")
                print(f"   🚀 Attempt {attempt}: Final retry with OpenAI GPT-4o")
                try:
                    print(f"   🔄 Final retry with OpenAI API...")

                    response = self._call_openai(
                        system_prompt=self._get_raw_excel_system_prompt(),
                        user_prompt=prompt
                    )

                    # Extract token usage information if available
                    if response and 'total_tokens' in response:
                        input_tokens = response.get('total_tokens', 0)
                        output_tokens = response.get('eval_count', 0)
                        total_tokens = input_tokens + output_tokens
                        print(f"   📊 Token Usage:")
                        print(f"      • Input tokens: {input_tokens:,}")
                        print(f"      • Output tokens: {output_tokens:,}")
                        print(f"      • Total tokens: {total_tokens:,}")

                    print(f"   ✅ AI analysis successful on attempt {attempt}")

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

            print(f"   ✅ Response received successfully:")
            print(f"      • Response length: {response_length:,} characters")
            print(f"      • Estimated output tokens: {estimated_output_tokens:,}")
            print(f"      • Configuration used: ctx={options.get('num_ctx') if options else 'n/a'}, predict={options.get('num_predict') if options else 'n/a'}")

            if total_tokens_used > 0:
                print(f"   💰 FINAL TOKEN SUMMARY:")
                print(f"      • Total Input Tokens: {total_input_tokens:,}")
                print(f"      • Total Output Tokens: {total_output_tokens:,}")
                print(f"      • TOTAL TOKENS USED: {total_tokens_used:,}")
                print(f"      • Model: {self.openai_model}")

                # Estimate cost for reference (OpenAI pricing for comparison)
                if total_tokens_used > 0:
                    gpt4_cost = (total_input_tokens * 0.00003) + (total_output_tokens * 0.00006)  # GPT-4 pricing
                    print(f"      • Estimated cost if using GPT-4: ${gpt4_cost:.4f}")
                print(f"   📝 Response preview: {result[:200]}...")

            # Debug: Check if response looks like JSON
            stripped = result.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                print(f"   ✅ Response appears to be JSON array format")
            elif '{' in stripped and '}' in stripped:
                print(f"   ⚠️  Response contains JSON objects but may need format correction")
            else:
                print(f"   🚨 Response does not appear to be JSON format - parsing may fail")

            print(f"\n🔍 STEP 5: JSON Parsing & Validation")
            print(f"   🔄 Parsing AI response into structured anomaly data...")

            # Debug: Print the full AI response
            print(f"\n📄 ===== FULL AI RESPONSE =====")
            print(result)
            print(f"===== END AI RESPONSE =====\n")

            anomalies = self._parse_llm_response(result, subsidiary)

            print(f"   ✅ Parsing completed successfully:")
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

    def _analyze_with_two_step_process(self, bs_csv, pl_csv, subsidiary, filename, config):
        """
        Two-step AI process for large datasets with any Excel format.

        STEP 1: AI extracts and groups accounts from raw CSV (format-agnostic)
        STEP 2: AI applies 22 rules to the grouped account data
        STEP 3: Python formats output consistently

        This works with ANY Excel format - no column name assumptions!
        """
        print(f"\n   🎯 TWO-STEP AI ANALYSIS PROCESS")
        print(f"   " + "="*70)

        # ============================================================
        # STEP 1: AI ACCOUNT EXTRACTION & GROUPING
        # ============================================================
        print(f"\n   📊 STEP 1: Account Extraction & Grouping")
        print(f"   ─────────────────────────────────────────")
        print(f"   🔄 Sending raw CSV to AI for account extraction...")

        step1_prompt = self._create_account_extraction_prompt(bs_csv, pl_csv, subsidiary, filename)

        try:
            step1_response = self._call_openai(
                system_prompt=self._get_account_extraction_system_prompt(),
                user_prompt=step1_prompt
            )

            if not step1_response or 'message' not in step1_response:
                raise RuntimeError("Step 1 failed: No response from AI")

            # Parse the grouped account data
            import json
            grouped_accounts_text = step1_response['message']['content']

            # Clean and parse JSON
            if grouped_accounts_text.startswith('```json'):
                grouped_accounts_text = grouped_accounts_text[7:]
            if grouped_accounts_text.endswith('```'):
                grouped_accounts_text = grouped_accounts_text[:-3]
            grouped_accounts_text = grouped_accounts_text.strip()

            grouped_accounts = json.loads(grouped_accounts_text)

            print(f"   ✅ Step 1 complete: Accounts extracted and grouped")
            print(f"      • BS accounts found: {len(grouped_accounts.get('bs_accounts', {}))}")
            print(f"      • PL accounts found: {len(grouped_accounts.get('pl_accounts', {}))}")
            print(f"      • Months detected: {grouped_accounts.get('months', [])}")

            # Track tokens from Step 1
            step1_input_tokens = step1_response.get('prompt_eval_count', 0)
            step1_output_tokens = step1_response.get('eval_count', 0)
            step1_cost = step1_response.get('cost', {}).get('total_cost', 0)

        except Exception as e:
            print(f"   ❌ Step 1 failed: {str(e)}")
            return [{
                "subsidiary": subsidiary,
                "account_code": "STEP1_ERROR",
                "rule_name": "Account Extraction Failed",
                "description": f"AI could not extract accounts from raw CSV: {str(e)[:100]}",
                "details": f"Step 1 error: {str(e)}",
                "current_value": 0,
                "previous_value": 0,
                "change_amount": 0,
                "change_percent": 0,
                "severity": "High",
                "sheet_type": "Error"
            }]

        # ============================================================
        # STEP 2: AI RULE APPLICATION
        # ============================================================
        print(f"\n   📋 STEP 2: Applying 22 Variance Analysis Rules")
        print(f"   ─────────────────────────────────────────")
        print(f"   🔄 Sending grouped accounts to AI for rule analysis...")

        step2_prompt = self._create_rule_application_prompt(grouped_accounts, subsidiary, filename, config)

        try:
            step2_response = self._call_openai(
                system_prompt=self._get_raw_excel_system_prompt(),  # Use the 22-rule prompt
                user_prompt=step2_prompt
            )

            if not step2_response or 'message' not in step2_response:
                raise RuntimeError("Step 2 failed: No response from AI")

            # Parse variance flags
            variances = self._parse_llm_response(step2_response['message']['content'], subsidiary)

            print(f"   ✅ Step 2 complete: Rules applied")
            print(f"      • Variances detected: {len(variances)}")

            # Track tokens from Step 2
            step2_input_tokens = step2_response.get('prompt_eval_count', 0)
            step2_output_tokens = step2_response.get('eval_count', 0)
            step2_cost = step2_response.get('cost', {}).get('total_cost', 0)

        except Exception as e:
            print(f"   ❌ Step 2 failed: {str(e)}")
            return [{
                "subsidiary": subsidiary,
                "account_code": "STEP2_ERROR",
                "rule_name": "Rule Application Failed",
                "description": f"AI could not apply 22 rules: {str(e)[:100]}",
                "details": f"Step 2 error: {str(e)}. Grouped accounts were extracted successfully in Step 1.",
                "current_value": 0,
                "previous_value": 0,
                "change_amount": 0,
                "change_percent": 0,
                "severity": "High",
                "sheet_type": "Error"
            }]

        # ============================================================
        # SUMMARY
        # ============================================================
        total_input_tokens = step1_input_tokens + step2_input_tokens
        total_output_tokens = step1_output_tokens + step2_output_tokens
        total_cost = step1_cost + step2_cost
        total_tokens = total_input_tokens + total_output_tokens

        print(f"\n   ✅ TWO-STEP ANALYSIS COMPLETE")
        print(f"   " + "="*70)
        print(f"      • Total variances: {len(variances)}")
        print(f"      • Total tokens: {total_tokens:,}")
        if total_cost > 0:
            print(f"      • Total cost: ${total_cost:.6f} USD")
        print(f"      • Step 1 tokens: {step1_input_tokens + step1_output_tokens:,}")
        print(f"      • Step 2 tokens: {step2_input_tokens + step2_output_tokens:,}")

        return variances

    def _get_sheet_detection_system_prompt(self) -> str:
        """System prompt for detecting which sheets are BS and PL based on sheet names only."""
        return """You are a financial document analyzer. Your job is to identify which Excel sheets contain Balance Sheet data and which contain Profit & Loss (Income Statement) data based ONLY on the sheet names.

🎯 YOUR TASK:
Analyze the sheet names and identify:
- Which sheet likely contains Balance Sheet / Statement of Financial Position data
- Which sheet likely contains Profit & Loss / Income Statement data

📊 IDENTIFICATION CLUES FROM SHEET NAMES:

**Balance Sheet / Statement of Financial Position:**
- Sheet names containing: "BS", "Balance", "Financial Position", "Statement of Financial Position", "Assets", "Liabilities", "Equity"
- Common patterns: "BS Breakdown", "Balance Sheet", "SOFP", "Statement of FP"
- Languages: English, Vietnamese (e.g., "Bảng cân đối", "BCĐKT")

**Profit & Loss / Income Statement:**
- Sheet names containing: "PL", "P&L", "Profit", "Loss", "Income", "Revenue", "Expenses"
- Common patterns: "PL Breakdown", "Profit & Loss", "Income Statement", "P&L"
- Languages: English, Vietnamese (e.g., "Báo cáo kết quả", "KQKD")

📋 REQUIRED OUTPUT FORMAT:
Return ONLY valid JSON with EXACT sheet names from the provided list:

{
  "bs_sheet": "exact_sheet_name_here",
  "pl_sheet": "exact_sheet_name_here"
}

⚠️ IMPORTANT: Use the EXACT sheet names as provided in the input list. Do not modify or create new names."""

    def _create_sheet_detection_prompt(self, all_sheets_csv, subsidiary, filename):
        """Create prompt for AI to detect which sheets are BS and PL."""
        prompt_parts = [f"""
SHEET DETECTION REQUEST

Company: {subsidiary}
File: {filename}

Analyze the following sheet previews and identify which is the Balance Sheet and which is the Profit & Loss:

"""]

        for sheet_name, (full_csv, preview_csv) in all_sheets_csv.items():
            prompt_parts.append(f"""
=== SHEET: "{sheet_name}" (first 200 rows) ===
{preview_csv[:10000]}
""")  # Limit to 10k chars per sheet for token efficiency

        prompt_parts.append("""

Identify which sheet is the Balance Sheet and which is the Profit & Loss.
Return JSON with exact sheet names.""")

        return "".join(prompt_parts)

    def _find_sheet_fuzzy(self, sheet_names, is_balance_sheet=True):
        """
        Find sheet name using fuzzy matching (no AI needed).
        Handles variations like:
        - "BS Breakdown", "BS breakdown", "BSbreakdown", "bs breakdown"
        - "PL Breakdown", "PL breakdown", "PLbreakdown", "pl breakdown"
        - "Balance Sheet", "BẢNG CÂN ĐỐI KẾ TOÁN"
        - "Profit Loss", "Income Statement", "BÁO CÁO KẾT QUẢ KINH DOANH"
        """
        from difflib import SequenceMatcher

        if is_balance_sheet:
            # BS sheet patterns (priority order)
            patterns = [
                "bs breakdown", "bs_breakdown", "bsbreakdown",
                "balance sheet breakdown", "balance_sheet",
                "balance sheet", "bs", "bảng cân đối kế toán"
            ]
        else:
            # PL sheet patterns (priority order)
            patterns = [
                "pl breakdown", "pl_breakdown", "plbreakdown",
                "profit loss breakdown", "profit_loss",
                "profit loss", "income statement", "p&l", "p/l", "pl",
                "báo cáo kết quả kinh doanh"
            ]

        # Normalize sheet names for comparison
        sheet_names_lower = {name.lower().replace(' ', '').replace('_', ''): name for name in sheet_names}

        # Try exact matches first (ignoring case, spaces, underscores)
        for pattern in patterns:
            pattern_normalized = pattern.lower().replace(' ', '').replace('_', '')
            for normalized, original in sheet_names_lower.items():
                if pattern_normalized == normalized:
                    return original

        # Try contains match
        for pattern in patterns:
            pattern_normalized = pattern.lower().replace(' ', '').replace('_', '')
            for normalized, original in sheet_names_lower.items():
                if pattern_normalized in normalized or normalized in pattern_normalized:
                    return original

        # Try fuzzy matching with threshold
        best_match = None
        best_score = 0.0
        threshold = 0.6  # 60% similarity required

        for pattern in patterns:
            pattern_normalized = pattern.lower().replace(' ', '').replace('_', '')
            for normalized, original in sheet_names_lower.items():
                score = SequenceMatcher(None, pattern_normalized, normalized).ratio()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = original

        if best_match:
            print(f"      ℹ️  Fuzzy matched '{best_match}' (score: {best_score:.2f})")
            return best_match

        return None

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
        """System prompt for Step 1: Account extraction from raw CSV."""
        return """You are a financial data extraction specialist. Your job is to extract and group account data from raw Excel CSV files in ANY format.

🎯 YOUR TASK:
Extract account codes, names, and values from the raw CSV data. Handle ANY Excel format:
- Account codes might be in column A, B, C, or embedded in text
- Headers might be at row 1, 5, 10, or any row
- Month columns might be named "Jan 2025", "As of Jan 2025", "M01", etc.
- Account codes might be formatted as "217", "217xxx", "(217)", "217 - Investment Property", etc.

📊 EXTRACTION STRATEGY:
1. **Identify Headers**: Find the row containing month/period names (look for date patterns, month names)
2. **Identify Account Codes**: Scan for numeric patterns that look like Vietnamese chart of accounts (111, 217, 341, 511, 632, etc.)
3. **Extract Values**: Get the corresponding values for each account in each month
4. **Group by Account Family**: Organize accounts by their 3-digit prefix (217xxx, 511xxx, etc.)

📋 REQUIRED OUTPUT FORMAT:
Return ONLY valid JSON (no markdown, no code blocks):

{
  "months": ["Jan 2025", "Feb 2025", "Mar 2025"],
  "bs_accounts": {
    "111": {"name": "Cash", "values": [1000000, 1200000, 1300000]},
    "217": {"name": "Investment Property", "values": [5000000, 7000000, 7500000]},
    "341": {"name": "Loans", "values": [9000000, 9500000, 10000000]}
  },
  "pl_accounts": {
    "511": {"name": "Revenue", "values": [2000000, 2200000, 2400000]},
    "632100001": {"name": "Amortization", "values": [100000, 105000, 110000]},
    "632100002": {"name": "Depreciation", "values": [200000, 195000, 190000]}
  }
}

🔍 IMPORTANT NOTES:
- Values array should match the months array length
- Group similar accounts together (all 217xxx under "217", all 511xxx under "511")
- Sum up sub-accounts if needed (217001 + 217002 = total for "217")
- Extract ACTUAL values from the CSV, not zeros
- Handle Vietnamese account names and formats

Return comprehensive account extraction covering all major account families."""

    def _create_account_extraction_prompt(self, bs_csv, pl_csv, subsidiary, filename):
        """Create prompt for Step 1: Account extraction."""
        return f"""
ACCOUNT EXTRACTION REQUEST

Company: {subsidiary}
File: {filename}

=== RAW BALANCE SHEET CSV ===
{bs_csv}

=== RAW PROFIT & LOSS CSV ===
{pl_csv}

=== INSTRUCTIONS ===

Extract all accounts and their values from the raw CSV data above.

Focus on these account families:
- BS: 111 (Cash), 112 (Cash Equivalents), 133 (VAT Input), 217 (Investment Property), 241 (CIP), 242 (Broker Assets), 341 (Loans), and any total rows
- PL: 511 (Revenue), 515 (Interest Income), 632 (COGS/D&A), 635 (Interest Expense), 641 (Selling Expense), 642 (G&A Expense)

Identify all available month columns and extract values for each account across all months.

Return structured JSON with grouped account data."""

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
            print(f"   🔄 Loading Excel file from bytes...")

            bs_raw, pl_raw = self._load_excel_sheets(excel_bytes)
            bs_clean, pl_clean = self._clean_data_for_ai(bs_raw, pl_raw, subsidiary)

            # Step 2: Convert to CSV for AI analysis
            print(f"\n📝 STEP 2: CSV Conversion for AI Processing")
            bs_csv = bs_clean.to_csv(index=False, header=True, quoting=1, float_format='%.0f')
            pl_csv = pl_clean.to_csv(index=False, header=True, quoting=1, float_format='%.0f')

            print(f"   ✅ CSV conversion complete:")
            print(f"      • BS CSV: {len(bs_csv):,} characters")
            print(f"      • PL CSV: {len(pl_csv):,} characters")

            # Step 3: Create specialized revenue analysis prompt
            print(f"\n📝 STEP 3: Creating Comprehensive Revenue Analysis Prompt")
            prompt = self._create_revenue_analysis_prompt(bs_csv, pl_csv, subsidiary, filename, config)
            prompt_length = len(prompt)
            print(f"   ✅ Prompt generation complete:")
            print(f"      • Total prompt length: {prompt_length:,} characters")

            # Step 4: AI Model Processing
            print(f"\n🤖 STEP 4: AI Revenue Analysis Processing")
            try:
                print(f"   🚀 Sending comprehensive revenue analysis request to AI...")

                response = self._call_openai(
                    system_prompt=self._get_revenue_analysis_system_prompt(),
                    user_prompt=prompt
                )

                # Extract token usage information
                if response and 'total_tokens' in response:
                    input_tokens = response.get('total_tokens', 0)
                    output_tokens = response.get('eval_count', 0)
                    total_tokens = response.get('total_tokens', 0)
                    print(f"   📊 Token Usage:")
                    print(f"      • Input tokens: {input_tokens:,}")
                    print(f"      • Output tokens: {output_tokens:,}")
                    print(f"      • Total tokens: {total_tokens:,}")

                print(f"   ✅ AI revenue analysis successful")

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
            print(f"\n📄 STEP 5: Processing AI Revenue Analysis Response")

            if not response or 'message' not in response or not response['message'] or 'content' not in response['message']:
                print(f"   ❌ Invalid response structure from OpenAI")
                raise RuntimeError("Empty response payload from OpenAI")

            result = response['message']['content'] or ""
            response_length = len(result)

            print(f"   ✅ Response received successfully:")
            print(f"      • Response length: {response_length:,} characters")

            # Debug: Print the full AI response
            print(f"\n📄 ===== FULL AI REVENUE ANALYSIS RESPONSE =====")
            print(result)
            print(f"===== END AI RESPONSE =====\n")

            # Parse the comprehensive revenue analysis response
            revenue_analysis = self._parse_revenue_analysis_response(result, subsidiary)

            print(f"   ✅ Parsing completed successfully:")
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
        return """You are a senior financial auditor with 15+ years experience in Vietnamese enterprises. You will analyze raw Excel financial data (BS Breakdown and PL Breakdown sheets) and apply the following 22 VARIANCE ANALYSIS RULES:

🎯 YOUR TASK:
Analyze the raw CSV data and identify variances based on the 22 rules below. For each variance found, return a JSON object with the rule details.

📋 THE 22 VARIANCE ANALYSIS RULES:

═══════════════════════════════════════════════════════════════
🔴 CRITICAL RULES (Priority: Critical)
═══════════════════════════════════════════════════════════════

**A1 - Asset capitalized but depreciation not started** [SEVERITY: Critical]
- Accounts: 217xxx (Investment Property) ↔ 632100001/632100002 (D&A)
- Logic: IF Investment Property (217xxx) increased BUT Depreciation/Amortization (ONLY 632100001 or 632100002) did NOT increase
- Flag Trigger: IP↑ BUT D&A ≤ previous
- Note: Use ONLY accounts 632100001 (Amortization) and 632100002 (Depreciation), NOT all 632xxx

**A2 - Loan drawdown but interest not recorded** [SEVERITY: Critical]
- Accounts: 341xxx (Loans) ↔ 635xxx (Interest Expense) + 241xxx (CIP Interest)
- Logic: IF Loans (341xxx) increased BUT day-adjusted Interest Expense (635xxx + 241xxx) did NOT increase
- Flag Trigger: Loan↑ BUT Day-adjusted Interest ≤ previous
- Note: Normalize interest by calendar days (Feb=28/30, Jan=31/30, etc.)

**A3 - Capex incurred but VAT not recorded** [SEVERITY: Critical]
- Accounts: 217xxx/241xxx (IP/CIP) ↔ 133xxx (VAT Input)
- Logic: IF Investment Property OR CIP increased BUT VAT Input (133xxx) did NOT increase
- Flag Trigger: Assets↑ BUT VAT input ≤ previous

**A4 - Cash movement disconnected from interest** [SEVERITY: Critical]
- Accounts: 111xxx/112xxx (Cash) ↔ 515xxx (Interest Income)
- Logic: IF Cash increased BUT day-adjusted Interest Income decreased OR Cash decreased BUT Interest Income increased
- Flag Trigger: Cash↑ BUT Interest↓ OR Cash↓ BUT Interest↑
- Note: Normalize interest by calendar days

**A5 - Lease termination but broker asset not written off** [SEVERITY: Critical]
- Accounts: 511xxx (Revenue) ↔ 242xxx (Broker Assets) ↔ 641xxx (Selling Expense)
- Logic: IF Revenue ≤ 0 BUT Broker Assets (242xxx) unchanged AND Selling Expense (641xxx) unchanged
- Flag Trigger: Revenue ≤ 0 BUT 242 unchanged AND 641 unchanged

**A7 - Asset disposal but accumulated depreciation not written off** [SEVERITY: Critical]
- Accounts: 217xxx (IP Cost) ↔ 217xxx (IP Accumulated Depreciation)
- Logic: IF IP Cost decreased BUT Accumulated Depreciation did NOT decrease
- Flag Trigger: IP cost↓ BUT Accumulated depreciation unchanged
- Note: Filter by Account Name containing "cost" vs "accum" or "depreciation"

**D1 - Balance sheet imbalance** [SEVERITY: Critical]
- Accounts: Total Assets vs Total Liabilities+Equity
- Logic: Check Balance Sheet equation: Total Assets = Total Liabilities + Equity
- Flag Trigger: Total Assets ≠ Total Liabilities+Equity (tolerance: 100M VND)
- Method: Use total rows "TỔNG CỘNG TÀI SẢN" and "TỔNG CỘNG NGUỒN VỐN" directly

**E1 - Negative Net Book Value (NBV)** [SEVERITY: Critical]
- Accounts: Account Lines 222/223, 228/229, 231/232
- Logic: Check NBV = Cost + Accumulated Depreciation (accum dep is negative) > 0
- Flag Trigger: NBV < 0 for any asset class
- Pairs: 222/223, 228/229, 231/232

═══════════════════════════════════════════════════════════════
🟡 REVIEW RULES (Priority: Review)
═══════════════════════════════════════════════════════════════

**B1 - Rental revenue volatility** [SEVERITY: Review]
- Accounts: 511710001 (Rental Revenue)
- Logic: IF current month rental revenue deviates > 2σ from 6-month average
- Flag Trigger: abs(Current - Avg) > 2σ

**B2 - Depreciation changes without asset movement** [SEVERITY: Review]
- Accounts: 632100002 (Depreciation) + 217xxx (IP)
- Logic: IF Depreciation deviates > 2σ from 6-month average BUT IP unchanged
- Flag Trigger: Depreciation deviates > 2σ AND IP unchanged

**B3 - Amortization changes** [SEVERITY: Review]
- Accounts: 632100001 (Amortization)
- Logic: IF Amortization deviates > 2σ from 6-month average
- Flag Trigger: abs(Current - Avg) > 2σ

**C1 - Gross margin by revenue stream** [SEVERITY: Review]
- Revenue Streams:
  * Utilities: 511800001 ↔ 632100011
  * Service Charges: 511600001/511600005 ↔ 632100008/632100015
  * Other Revenue: 511800002 ↔ 632199999
- Logic: IF Gross Margin % deviates > 2σ from 6-month baseline
- Flag Trigger: GM% change > 2σ
- Note: IGNORE rental/leasing revenue vs depreciation

**C2 - Unbilled reimbursable expenses** [SEVERITY: Review]
- Accounts: 641xxx/632xxx (Reimbursable COGS) ↔ 511xxx (Revenue)
- Logic: IF Reimbursable COGS increased BUT Revenue did NOT increase
- Flag Trigger: Reimbursable COGS↑ BUT Revenue unchanged

**D2 - Retained earnings reconciliation break** [SEVERITY: Review]
- Accounts: Account Line 421/4211 (Retained Earnings) ↔ P&L components
- Logic: Opening RE + Net Income ≠ Closing RE (tolerance: 1M VND)
- Flag Trigger: |Calculated RE - Actual RE| > 1M VND
- Formula: Closing RE = Opening RE + Net Income (from P&L lines 1,11,21,22,23,25,26,31,32,51,52)

═══════════════════════════════════════════════════════════════
🟢 WATCH RULES (Priority: Info)
═══════════════════════════════════════════════════════════════

**E2 - Revenue vs selling expense disconnect** [SEVERITY: Info]
- Accounts: 511xxx (Revenue) ↔ 641xxx (Selling Expenses)
- Logic: IF Revenue changed significantly BUT Selling Expense (641) unchanged
- Flag Trigger: Revenue moves > 10% BUT 641 relatively flat

**E3 - Revenue vs Advance Revenue (prepayments)** [SEVERITY: Info]
- Accounts: 511xxx (Revenue) ↔ 131xxx (A/R) ↔ 3387 (Unearned Revenue/Advances)
- Logic: Monitor relationship between revenue recognition and advance payments
- Flag Trigger: Unusual patterns in advance revenue movements

**E4 - Monthly recurring charges** [SEVERITY: Info]
- Accounts: 511 (Total Revenue) vs specific recurring revenue streams
- Logic: Check if recurring revenue streams remain stable month-over-month
- Flag Trigger: Unexpected drops or spikes in normally recurring items

**E5 - One-off revenue items** [SEVERITY: Info]
- Accounts: Non-recurring revenue accounts
- Logic: Identify and highlight one-time revenue items
- Flag Trigger: Unusual account activity that appears non-recurring

**E6 - General & admin expense volatility (642xxx)** [SEVERITY: Info]
- Accounts: 642xxx (G&A Expenses)
- Logic: IF G&A expenses deviate significantly from baseline
- Flag Trigger: Unusual volatility in administrative costs

**F1 - Operating expense volatility (641xxx excluding 641100xxx)** [SEVERITY: Info]
- Accounts: 641xxx (Operating Expenses), excluding 641100xxx
- Logic: IF Operating expenses (excl. commissions) show unusual patterns
- Flag Trigger: Significant deviation from baseline

**F2 - Broker commission volatility (641100xxx)** [SEVERITY: Info]
- Accounts: 641100xxx (Broker Commissions) ↔ 511xxx (Revenue)
- Logic: Check if commission expense scales appropriately with revenue
- Flag Trigger: Commission % of revenue changes significantly

**F3 - Personnel cost volatility (642100xxx)** [SEVERITY: Info]
- Accounts: 642100xxx (Personnel Costs)
- Logic: IF Personnel costs deviate from baseline (excluding known hiring/layoffs)
- Flag Trigger: Unexpected changes in headcount-related expenses

📊 DETAILED ANALYSIS REQUIREMENTS:

1. TOTAL REVENUE ANALYSIS (511*):
   - Calculate total 511* revenue by month across all entities
   - Identify month-over-month changes with VND amounts and percentages
   - Flag significant variance periods (>1M VND changes)

2. REVENUE BY ACCOUNT TYPE (511.xxx):
   - Break down each 511* revenue account separately
   - For each account: track monthly totals and identify biggest changes
   - For accounts with changes >1M VND: analyze which entities/customers drive the changes
   - Provide top 5 entity impacts with VND amounts and percentages

3. SG&A 641* ANALYSIS:
   - Identify all 641* accounts and track monthly totals
   - Calculate month-over-month changes for each 641* account
   - For accounts with changes >500K VND: analyze entity-level impacts
   - Provide top 5 entity impacts showing expense variance drivers

4. SG&A 642* ANALYSIS:
   - Identify all 642* accounts and track monthly totals
   - Calculate month-over-month changes for each 642* account
   - For accounts with changes >500K VND: analyze entity-level impacts
   - Provide top 5 entity impacts showing expense variance drivers

5. COMBINED SG&A RATIO ANALYSIS:
   - Calculate total SG&A (641* + 642*) by month
   - Calculate SG&A as percentage of revenue for each month
   - Track month-over-month changes in SG&A ratio
   - Flag ratio changes >2% as medium risk, >3% as high risk

6. GROSS MARGIN ANALYSIS:
   - Calculate gross margin: (Revenue - COGS)/Revenue by month
   - Track margin percentage changes month-over-month
   - Flag margin changes >1% as concerning trends

7. ENTITY-LEVEL IMPACT ANALYSIS:
   - For significant account changes: identify which entities/customers drive the variance
   - Show entity name, change amount, percentage change, previous/current values
   - Focus on entities with changes >100K VND for revenue, >50K VND for SG&A

📊 DATA EXTRACTION INSTRUCTIONS:
1. ACCOUNT DETECTION: Automatically identify account codes and names:
   - Revenue accounts: Look for 511* patterns in account codes and names
   - SG&A 641* accounts: Look for 641* patterns in account codes and names
   - SG&A 642* accounts: Look for 642* patterns in account codes and names
   - COGS accounts: Look for 632* patterns for gross margin calculation
   - Extract the numeric codes and descriptive names
   - Match account codes with their corresponding financial values across months

2. ENTITY/CUSTOMER IDENTIFICATION: Find entity-level data:
   - Look for "Entity" columns or customer/subsidiary names
   - Track values by entity for each account across months
   - Identify which entities drive account-level changes
   - Focus on entities with significant value changes

3. PERIOD IDENTIFICATION: Find all available month columns:
   - Extract ALL month column headers (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec)
   - Use actual period names from the Excel headers
   - Track up to 8 months of data for trend analysis
   - Calculate month-over-month changes across the full timeline

4. VALUE EXTRACTION: Extract actual financial amounts:
   - Look for large numbers (typically 6+ digits for VND amounts)
   - Handle different formats: 2,249,885,190.00, 46,000,000,000.00, etc.
   - Track values by account, by entity, by month
   - Sum totals across entities for account-level analysis

5. COMPREHENSIVE CALCULATIONS:
   - Total revenue (511*) by month across all entities
   - Total SG&A 641* by month across all entities
   - Total SG&A 642* by month across all entities
   - Combined SG&A (641* + 642*) by month
   - SG&A ratio: Total SG&A / Total Revenue * 100
   - Gross margin: (Revenue - COGS) / Revenue * 100
   - Month-over-month changes for all metrics

💰 MATERIALITY THRESHOLDS:
- Revenue-based: 2% of total revenue or 50M VND (whichever is lower)
- Balance-based: 0.5% of total assets or 100M VND (whichever is lower)
- Focus on ANY account with changes >10% or unusual patterns
- Always explain your materiality reasoning

⚡ CRITICAL OUTPUT REQUIREMENTS:
1. You MUST respond with ONLY valid JSON array format
2. Start with [ and end with ]
3. No markdown, no ```json blocks, no additional text
4. Provide COMPREHENSIVE ANALYSIS covering all 7 focus areas
5. Include both account-level and entity-level insights
6. Calculate ratios, trends, and risk assessments
7. Use actual values from the Excel data
8. Focus on 511*, 641*, 642* accounts with entity-level detail

📋 REQUIRED COMPREHENSIVE JSON FORMAT:
[{
  "analysis_type": "total_revenue_trend",
  "account": "511*-Total Revenue",
  "description": "Total revenue analysis across all 511* accounts",
  "explanation": "Total 511* revenue changed from [previous total] to [current total] VND. Key drivers: [list main revenue accounts]. Month-over-month trend shows [pattern].",
  "period": "Feb 2025",
  "current_value": 0,
  "previous_value": 0,
  "change_amount": 0,
  "change_percent": 0,
  "severity": "Medium",
  "details": {
    "monthly_totals": {"Jan": 0, "Feb": 0, "Mar": 0},
    "biggest_changes": [{"period": "Feb→Mar", "change": 0, "pct_change": 0}]
  }
},
{
  "analysis_type": "revenue_by_account",
  "account": "511xxx-Specific Revenue Account",
  "description": "Individual revenue account analysis with entity breakdown",
  "explanation": "Account [name] showed [change description]. Top entity impacts: [entity name] contributed [amount] VND change.",
  "period": "Feb 2025",
  "current_value": 0,
  "previous_value": 0,
  "change_amount": 0,
  "change_percent": 0,
  "severity": "Low",
  "details": {
    "monthly_totals": {"Jan": 0, "Feb": 0},
    "entity_impacts": [{"entity": "Entity Name", "change": 0, "pct_change": 0, "prev_val": 0, "curr_val": 0}]
  }
},
{
  "analysis_type": "sga_641_analysis",
  "account": "641xxx-SG&A Account",
  "description": "SG&A 641* account analysis with entity-level variance tracking",
  "explanation": "SG&A account [name] changed by [amount] VND. Entity breakdown shows [top contributors].",
  "period": "Feb 2025",
  "current_value": 0,
  "previous_value": 0,
  "change_amount": 0,
  "change_percent": 0,
  "severity": "Medium",
  "details": {
    "monthly_totals": {"Jan": 0, "Feb": 0},
    "entity_impacts": [{"entity": "Entity Name", "change": 0, "pct_change": 0, "prev_val": 0, "curr_val": 0}]
  }
},
{
  "analysis_type": "sga_642_analysis",
  "account": "642xxx-SG&A Account",
  "description": "SG&A 642* account analysis with entity-level variance tracking",
  "explanation": "SG&A account [name] changed by [amount] VND. Entity breakdown shows [top contributors].",
  "period": "Feb 2025",
  "current_value": 0,
  "previous_value": 0,
  "change_amount": 0,
  "change_percent": 0,
  "severity": "Medium",
  "details": {
    "monthly_totals": {"Jan": 0, "Feb": 0},
    "entity_impacts": [{"entity": "Entity Name", "change": 0, "pct_change": 0, "prev_val": 0, "curr_val": 0}]
  }
},
{
  "analysis_type": "combined_sga_ratio",
  "account": "641*+642*-Combined SG&A",
  "description": "Combined SG&A ratio analysis as percentage of revenue",
  "explanation": "Total SG&A (641*+642*) represents [ratio]% of revenue, changing by [change] percentage points from previous period. Risk level: [assessment].",
  "period": "Feb 2025",
  "current_value": 0,
  "previous_value": 0,
  "change_amount": 0,
  "change_percent": 0,
  "severity": "High",
  "details": {
    "sga_ratio_trend": [{"month": "Jan", "revenue": 0, "total_sga": 0, "ratio_pct": 0}],
    "ratio_changes": [{"period": "Feb→Mar", "ratio_change": 0}]
  }
}]

🎯 ACCOUNT CODE EXTRACTION EXAMPLES:
- From "1. Tien (111)" → Extract: "111-Tien"
- From "2. Cac khoan tuong duong tien (112)" → Extract: "112-Cac khoan tuong duong tien"
- From "511000000 - Revenue from sale and service provision" → Extract: "511000000-Revenue from sale and service provision"
- From "627000000 - Cost of goods sold" → Extract: "627000000-Cost of goods sold"

🚨 REQUIREMENTS:
- period: MUST be the actual current period name from Excel headers (e.g., "As of Feb 2025", "Feb 2025")
- current_value: MUST be actual amount from Excel (number, not zero)
- previous_value: MUST be actual amount from Excel (number, not zero)
- change_amount: MUST be current_value - previous_value (number)
- change_percent: MUST be actual percentage change (number)
- All values must be real numbers extracted from the Excel data
- CRITICAL: Use ACTUAL period names from the CSV headers, not "current" or "previous"

ANALYZE THOROUGHLY. The raw Excel data contains the complete picture - use all available information to provide comprehensive financial analysis."""

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
        return """You are a senior financial auditor with 15+ years experience in Vietnamese enterprises. Provide SPECIFIC, ACTIONABLE analysis with detailed business context.

🎯 ANALYSIS DEPTH REQUIREMENTS:
1. REVENUE (511*): Analyze sales patterns, customer concentration, seasonality breaks, margin trends
2. UTILITIES (627*, 641*): Check operational efficiency, cost per unit, scaling with business activity
3. INTEREST (515*, 635*): Examine debt structure changes, cash flow implications, refinancing activities
4. CROSS-ACCOUNT RELATIONSHIPS: Flag disconnects between related accounts

🔍 SPECIFIC INVESTIGATION AREAS:
- Revenue Recognition Issues: Round numbers, unusual timing, concentration risks
- Working Capital Anomalies: A/R aging, inventory turns, supplier payment delays
- Cash Flow Red Flags: Operating vs financing activity mismatches
- Related Party Transactions: Unusual intercompany balances or transfers
- Asset Impairments: Sudden writedowns, depreciation policy changes
- Tax Accounting: Deferred tax movements, provision adequacy
- Management Estimates: Allowances, accruals, fair value adjustments

🧠 MATERIALITY FRAMEWORK:
- Quantitative: 5% of net income, 0.5% of revenue, 1% of total assets (adjust for company size)
- Qualitative: Fraud indicators, compliance issues, trend reversals, related party items
- ALWAYS state your specific materiality calculation and reasoning

📋 REQUIRED ANALYSIS COMPONENTS:
For EACH anomaly provide:
1. SPECIFIC BUSINESS CONTEXT: What this account typically represents in Vietnamese companies
2. ROOT CAUSE ANALYSIS: 3-5 specific scenarios that could cause this pattern
3. RISK ASSESSMENT: Financial statement impact, operational implications, compliance risks
4. VERIFICATION PROCEDURES: Specific audit steps to investigate (document requests, confirmations, etc.)
5. MANAGEMENT QUESTIONS: Exact questions to ask management about this variance

📊 OUTPUT FORMAT:
[{
  "account": "511001-Product Sales Revenue",
  "type": "Profit & Loss",
  "severity": "High|Medium|Low",
  "description": "Specific description of the anomaly pattern",
  "explanation": "DETAILED business context: (1) What this account means (2) Why this change is concerning (3) Specific business scenarios (4) Exact verification steps (5) Management interview questions",
  "previous_value": 0,
  "current_value": 0,
  "change_amount": 0,
  "change_percent": 0,
  "materiality_threshold_used": 0,
  "threshold_reasoning": "Specific calculation: X% of net income because...",
  "business_risk": "High|Medium|Low",
  "audit_priority": "Immediate|Next Review|Monitor",
  "investigation_steps": ["Step 1", "Step 2", "Step 3"],
  "management_questions": ["Question 1", "Question 2", "Question 3"]
}]

⚡ CRITICAL OUTPUT REQUIREMENTS:
1. You MUST respond with ONLY valid JSON array format - no explanatory text before or after
2. Start your response with [ and end with ]
3. No markdown formatting, no ```json blocks, no additional commentary
4. Each anomaly must be a complete JSON object with all required fields
5. If no anomalies found, return empty array: []
6. COMPREHENSIVE ANALYSIS: Detect ALL possible anomalies - do not limit results
7. Analyze every account with significant changes or patterns

📋 REQUIRED JSON OUTPUT FORMAT:
You MUST return JSON array with these EXACT field names:

[{
  "account": "128113002-ST-BIDV-Saving Account VND-Bidv-Thanh Xuan",
  "description": "Balance changed materially — check reclass/missing offset",
  "explanation": "Cash balance increased 34.5% - verify large deposits and transfers",
  "period": "Feb 2025",
  "current_value": 5600000000,
  "previous_value": 4200000000,
  "change_amount": 1400000000,
  "change_percent": 33.33,
  "severity": "Medium"
},
{
  "account": "31110001-Payables: Suppliers: Operating expenses",
  "description": "Balance changed materially — check reclass/missing offset",
  "explanation": "Payables decreased 25% - review payment timing and accruals",
  "period": "Feb 2025",
  "current_value": 2500000000,
  "previous_value": 3333000000,
  "change_amount": -833000000,
  "change_percent": -25.0,
  "severity": "Medium"
}]

⚡ CRITICAL FIELD REQUIREMENTS:
- "period": MUST be the actual period name from the Excel data (e.g., "Feb 2025", "As of Feb 2025")
- "current_value": MUST be actual current period amount (number)
- "previous_value": MUST be actual previous period amount (number)
- "change_amount": MUST be current_value - previous_value (number)
- "change_percent": MUST be percentage change (number, not string)
- ALL numeric fields must be actual numbers, not zero

CRITICAL: Keep explanations SHORT and focused. Avoid lengthy detailed analysis in the explanation field.

🚨 IMPORTANT: Any response that is not valid JSON will cause system failure. Match the above format exactly with Vietnamese business context."""

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
                print(f"   🔍 Debug anomaly {i+1}: Available fields: {list(anom.keys())}")
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
        return """You are a senior financial auditor specializing in comprehensive revenue impact analysis for Vietnamese enterprises. You will perform detailed analysis matching the methodology of our core analysis system.

🎯 REVENUE IMPACT ANALYSIS METHODOLOGY:
You must provide a complete analysis covering these specific areas:

1. TOTAL REVENUE TREND ANALYSIS (511*):
   - Calculate total 511* revenue by month across all entities
   - Identify month-over-month changes and patterns
   - Flag significant variance periods and explain business drivers

2. REVENUE BY ACCOUNT BREAKDOWN (511.xxx):
   - Analyze each individual 511* revenue account separately
   - Track monthly performance and identify biggest changes
   - For accounts with material changes: drill down to entity-level impacts

3. SG&A 641* EXPENSE ANALYSIS:
   - Identify and analyze all 641* accounts individually
   - Calculate monthly totals and variance trends
   - For significant changes: identify entity-level drivers

4. SG&A 642* EXPENSE ANALYSIS:
   - Identify and analyze all 642* accounts individually
   - Calculate monthly totals and variance trends
   - For significant changes: identify entity-level drivers

5. COMBINED SG&A RATIO ANALYSIS:
   - Calculate total SG&A (641* + 642*) as percentage of revenue
   - Track ratio changes month-over-month
   - Assess ratio trends and flag concerning patterns

6. ENTITY-LEVEL IMPACT ANALYSIS:
   - For each significant account change: identify driving entities/customers
   - Show entity contribution to variance with VND amounts and percentages
   - Focus on material entity impacts (>100K VND revenue, >50K VND SG&A)

📊 DATA PROCESSING REQUIREMENTS:
- Extract ALL month columns (up to 8 months of data)
- Identify entity/customer columns for detailed breakdowns
- Calculate accurate totals, subtotals, and ratios
- Track month-over-month changes across the timeline
- Use actual VND amounts from the Excel data

⚡ CRITICAL OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON array format (no markdown, no code blocks)
2. Include analysis_type field for each item to categorize findings
3. Provide both summary-level and detailed analysis items
4. Include actual financial amounts and percentage changes
5. Add entity-level details in the details object for drill-down capability
6. Cover ALL major analysis areas (don't skip any of the 6 areas above)

ANALYZE COMPREHENSIVELY AND RETURN DETAILED REVENUE IMPACT INSIGHTS."""

    def _create_revenue_analysis_prompt(self, bs_csv: str, pl_csv: str, subsidiary: str, filename: str, config: dict) -> str:
        """Create specialized prompt for comprehensive revenue impact analysis."""
        _ = config  # AI determines all parameters autonomously

        return f"""
COMPREHENSIVE REVENUE IMPACT ANALYSIS REQUEST

Company: {subsidiary}
File: {filename}
Analysis Type: Detailed Revenue & SG&A Impact Analysis (511*/641*/642*)

INSTRUCTIONS:
Perform comprehensive revenue impact analysis covering:
1. Total revenue trend analysis (511* accounts)
2. Individual revenue account breakdowns with entity impacts
3. SG&A 641* account analysis with entity-level variances
4. SG&A 642* account analysis with entity-level variances
5. Combined SG&A ratio analysis (% of revenue)
6. Entity-level impact identification for all material changes

Focus on accounts 511*, 641*, 642* and their entity-level details.
Calculate monthly totals, trends, and ratios.
Identify entities/customers driving significant variances.

=== RAW BALANCE SHEET DATA (BS Breakdown Sheet) ===
{bs_csv}

=== RAW P&L DATA (PL Breakdown Sheet) ===
{pl_csv}

Return comprehensive JSON analysis covering all 6 analysis areas with entity-level detail."""

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
            print(f"   🔍 Response sample: {response[:500]}...")
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

