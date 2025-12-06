# vLLM JSON Schema with LiteLLM Router/Proxy Configuration

This guide shows how to configure the LiteLLM Router/Proxy to handle JSON schema for vLLM models using **only YAML configuration** (no Python code required).

## Overview

Currently, LiteLLM does **NOT** automatically inject JSON schema into prompts for vLLM. However, you can configure the proxy/router to:

1. Mark models as not supporting `response_schema` or `function_calling`
2. Define custom prompt templates for schema injection
3. Use pre-request hooks to modify messages

This document shows **what's possible with YAML configuration today** and **what would need to be implemented** for full automatic injection.

---

## ⚠️ Current Status

**What Works:**
- ✅ Setting `litellm.drop_params` to prevent errors
- ✅ Setting model metadata via `model_info`
- ✅ Defining custom prompt dictionaries via `litellm_settings`
- ✅ Using pre-request hooks (requires Python file)

**What Doesn't Work (Yet):**
- ❌ Automatic prompt injection for vLLM models (like Gemini has)
- ❌ The custom_prompt_dict is defined but not used by vLLM provider

---

## Method 1: Basic Configuration (Drop Unsupported Params)

The simplest approach - tell LiteLLM to drop unsupported parameters:

```yaml
model_list:
  - model_name: my-vllm-model
    litellm_params:
      model: hosted_vllm/llama-3-8b
      api_base: http://localhost:8000
      api_key: none

litellm_settings:
  # Drop unsupported params instead of erroring
  drop_params: true

  # Enable detailed logging to see what's sent
  set_verbose: true

general_settings:
  master_key: sk-1234
```

**Usage:**
```bash
litellm --config config.yaml
```

Then clients just send requests normally, and `response_format` will be silently dropped if not supported.

---

## Method 2: Model Metadata Configuration

Set model capabilities in `model_info` to mark what features are supported:

```yaml
model_list:
  - model_name: my-vllm-model
    litellm_params:
      model: hosted_vllm/llama-3-8b
      api_base: http://localhost:8000

    # Model metadata - supports any additional fields
    model_info:
      # Standard fields
      mode: chat
      max_tokens: 4096
      max_input_tokens: 8192
      input_cost_per_token: 0.0
      output_cost_per_token: 0.0

      # Feature flags (extra fields are allowed)
      supports_function_calling: false
      supports_response_schema: false
      supports_vision: false
      supports_parallel_function_calling: false

litellm_settings:
  drop_params: true
```

**Note:** These flags are stored but currently **not used by the vLLM provider** to trigger automatic injection. They will be returned in `/model/info` endpoints though.

---

## Method 3: Custom Prompt Dictionary (Prepared for Future Use)

Define custom templates in `litellm_settings`:

```yaml
model_list:
  - model_name: my-vllm-llama
    litellm_params:
      model: hosted_vllm/llama-3-8b
      api_base: http://localhost:8000
    model_info:
      supports_response_schema: false
      supports_function_calling: false

litellm_settings:
  drop_params: true

  # Custom prompt templates
  custom_prompt_dict:
    # Global template for response_schema
    response_schema_prompt:
      roles:
        user:
          pre_message: |
            You must output valid JSON matching this schema:

          post_message: |

            IMPORTANT:
            - Output ONLY the JSON object
            - Do NOT use markdown code blocks
            - Include all required fields
      initial_prompt_value: ""
      final_prompt_value: ""

    # Model-specific template
    "hosted_vllm/llama-3-8b/response_schema_prompt":
      roles:
        user:
          pre_message: |
            ### Task
            Generate JSON output conforming to this schema:

          post_message: |

            ### Requirements
            1. Valid JSON only
            2. Match schema exactly
            3. No markdown formatting
      initial_prompt_value: ""
      final_prompt_value: ""

general_settings:
  master_key: sk-1234
```

**Important:** This sets `litellm.custom_prompt_dict` globally, but the vLLM provider **doesn't currently use it**. The Gemini provider does.

---

## Method 4: Pre-Request Hook (Workaround)

For a working solution today, use a custom pre-request hook:

### Step 1: Create `json_schema_hook.py`

```python
import json
from typing import Optional
from litellm.proxy.hooks.parallel_request_limiter import _PROXY_VirtualKeyModelMaxParallelRequestsHandler
from litellm.integrations.custom_logger import CustomLogger


class JSONSchemaInjector(CustomLogger):
    """
    Injects JSON schema into messages for models that don't support response_format natively.
    """

    def __init__(self):
        super().__init__()
        # Models that need schema injection
        self.models_needing_injection = {
            "my-vllm-model",
            "my-vllm-llama",
            # Add your model names here
        }

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: dict,
        data: dict,
        call_type: str,
    ):
        """
        Called before each request to the LLM.
        Modifies the request data to inject JSON schema.
        """
        if call_type != "completion" and call_type != "acompletion":
            return data

        model = data.get("model", "")
        if model not in self.models_needing_injection:
            return data

        # Check if response_format is present
        response_format = data.get("response_format")
        if not response_format:
            return data

        # Extract schema
        schema = None
        if isinstance(response_format, dict):
            if "json_schema" in response_format:
                schema = response_format["json_schema"].get("schema")
            elif "response_schema" in response_format:
                schema = response_format["response_schema"]

        if not schema:
            return data

        # Create schema instruction message
        schema_message = self._create_schema_message(schema)

        # Inject into messages
        messages = data.get("messages", [])
        messages.append({
            "role": "user",
            "content": schema_message
        })
        data["messages"] = messages

        # Remove response_format (not supported by model)
        data.pop("response_format", None)

        print(f"✅ Injected JSON schema into {model}")
        return data

    def _create_schema_message(self, schema: dict) -> str:
        """Creates the schema instruction message"""
        return f"""You must respond with valid JSON matching this exact schema:

```json
{json.dumps(schema, indent=2)}
```

IMPORTANT RULES:
- Output ONLY the JSON object (start with {{ and end with }})
- Do NOT include markdown code blocks (no ```json)
- Do NOT add any explanations before or after the JSON
- All required fields must be present
- Use the exact property names from the schema"""

# Initialize the hook
json_schema_injector = JSONSchemaInjector()
```

### Step 2: Configure in `config.yaml`

```yaml
model_list:
  - model_name: my-vllm-model
    litellm_params:
      model: hosted_vllm/llama-3-8b
      api_base: http://localhost:8000
      api_key: none

litellm_settings:
  drop_params: false  # We handle it in the hook

  # Register the custom callback
  callbacks: ["json_schema_hook.json_schema_injector"]

general_settings:
  master_key: sk-1234
```

### Step 3: Start Proxy

```bash
# Place json_schema_hook.py in the same directory as config.yaml
litellm --config config.yaml
```

### Step 4: Test It

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

response = client.chat.completions.create(
    model="my-vllm-model",
    messages=[
        {"role": "user", "content": "Extract: John Doe, 30, engineer"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "occupation": {"type": "string"}
                },
                "required": ["name", "age"]
            }
        }
    }
)

print(response.choices[0].message.content)
```

The hook will automatically inject the schema into the prompt!

---

## Method 5: Complete Production Example

Combining all approaches:

```yaml
model_list:
  # vLLM model without structured output support
  - model_name: llama-3-8b
    litellm_params:
      model: hosted_vllm/llama-3-8b
      api_base: http://vllm-server:8000
      api_key: none
      rpm: 1000
    model_info:
      mode: chat
      max_tokens: 4096
      max_input_tokens: 8192
      supports_function_calling: false
      supports_response_schema: false
      supports_vision: false

  # vLLM model WITH structured output support (newer models)
  - model_name: qwen-2.5-coder
    litellm_params:
      model: hosted_vllm/qwen2.5-coder-7b-instruct
      api_base: http://vllm-server:8000
      api_key: none
    model_info:
      mode: chat
      supports_function_calling: true
      supports_response_schema: true  # This model supports it!

litellm_settings:
  # Don't drop params - let the hook handle it
  drop_params: false

  # Enable logging to debug
  set_verbose: false  # Set to true for debugging

  # Register custom hook for schema injection
  callbacks: ["json_schema_hook.json_schema_injector"]

  # Custom templates (for future use when implemented in vLLM provider)
  custom_prompt_dict:
    "hosted_vllm/llama-3-8b/response_schema_prompt":
      roles:
        user:
          pre_message: "### JSON Schema\n\n"
          post_message: "\n\n### Output Format\nRespond with valid JSON only."
      initial_prompt_value: ""
      final_prompt_value: ""

general_settings:
  master_key: sk-1234

  # Optional: alerting
  alerting: ["slack"]

  # Optional: enable database
  database_url: "postgresql://user:password@localhost:5432/litellm"

router_settings:
  # Load balance between models
  routing_strategy: "simple-shuffle"

  # Retry settings
  num_retries: 2
  timeout: 60
```

---

## Validation: Check Configuration Loaded

After starting the proxy, validate your config:

```bash
# Check model info
curl http://localhost:4000/model/info \
  -H "Authorization: Bearer sk-1234"

# Should show your models with metadata
```

```python
# In Python
import requests

response = requests.get(
    "http://localhost:4000/model/info",
    headers={"Authorization": "Bearer sk-1234"}
)

models = response.json()["data"]
for model in models:
    if model["model_name"] == "llama-3-8b":
        print(f"Supports response_schema: {model.get('supports_response_schema', 'not set')}")
        print(f"Supports function_calling: {model.get('supports_function_calling', 'not set')}")
```

---

## Testing the Full Flow

### Test 1: Verify Hook is Injecting Schema

```python
import openai
import json

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

schema = {
    "type": "object",
    "properties": {
        "colors": {
            "type": "array",
            "items": {"type": "string"}
        },
        "count": {"type": "integer"}
    },
    "required": ["colors", "count"]
}

# This should work even though the model doesn't support response_format
response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[
        {"role": "user", "content": "List 3 primary colors"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "color_list",
            "schema": schema
        }
    }
)

result = json.loads(response.choices[0].message.content)
print(result)
# Should output: {"colors": ["red", "blue", "yellow"], "count": 3}
```

### Test 2: Check Proxy Logs

With `set_verbose: true` in config, you'll see:

```
✅ Injected JSON schema into llama-3-8b

POST Request Sent from LiteLLM:
{
  "model": "hosted_vllm/llama-3-8b",
  "messages": [
    {"role": "user", "content": "List 3 primary colors"},
    {"role": "user", "content": "You must respond with valid JSON matching this exact schema:\n\n```json\n{...}\n```"}
  ]
}
```

---

## Limitations & Future Work

### Current Limitations:

1. **Requires Python Hook File** - Cannot do pure YAML-only automatic injection
2. **vLLM Provider Doesn't Use custom_prompt_dict** - Only Gemini does
3. **Manual Hook Registration** - Need to maintain `json_schema_hook.py`

### What Would Need Implementation:

To make this work **without custom hooks**, the vLLM provider would need:

```python
# In litellm/llms/hosted_vllm/chat/transformation.py

def transform_request(self, model, messages, optional_params, ...):
    # Check for response_schema support
    if "response_schema" in optional_params:
        supports = supports_response_schema(
            model=model,
            custom_llm_provider="hosted_vllm"
        )

        if supports is False:
            # Inject schema using custom_prompt_dict template
            schema_message = response_schema_prompt(
                model=model,
                response_schema=optional_params["response_schema"]
            )
            messages.append({
                "role": "user",
                "content": schema_message
            })
            optional_params.pop("response_schema")

    # Continue transformation...
```

This is exactly what Gemini does (see `/litellm/llms/vertex_ai/gemini/transformation.py:463-472`).

---

## Comparison Table

| Method | Pure YAML | Works Today | Automatic | Maintenance |
|--------|-----------|-------------|-----------|-------------|
| **Drop Params** | ✅ Yes | ✅ Yes | ❌ No | ✅ Low |
| **Model Metadata** | ✅ Yes | ⚠️ Partial | ❌ No | ✅ Low |
| **Custom Prompt Dict** | ✅ Yes | ❌ Not used | ❌ No | ✅ Low |
| **Pre-Request Hook** | ❌ No (needs .py) | ✅ Yes | ✅ Yes | ⚠️ Medium |
| **Native Implementation** | ✅ Yes | ❌ Future | ✅ Yes | ✅ Low |

---

## Recommended Approach

**For Production Use Today:**

Use the **Pre-Request Hook** (Method 4):
- ✅ Works immediately
- ✅ Fully automatic
- ✅ Easy to customize per model
- ⚠️ Requires maintaining Python file

**For Future:**

Once vLLM provider gets native support (like Gemini has):
- Use pure YAML configuration
- Just set `model_info.supports_response_schema: false`
- Define custom templates in `litellm_settings.custom_prompt_dict`

---

## Related Documentation

- [vLLM JSON Schema Guide](./vllm_json_schema.md) - Manual Python approach
- [vLLM JSON Schema Notebook](./vllm_json_schema_demo.ipynb) - Interactive examples
- [LiteLLM Proxy Config](../../proxy/configs.md) - General proxy configuration
- [Custom Callbacks](../../proxy/custom_callbacks.md) - Writing custom hooks

---

## Contributing

To implement native support for vLLM:

1. Modify `/litellm/llms/hosted_vllm/chat/transformation.py`
2. Add logic similar to Gemini's implementation
3. Use existing `response_schema_prompt()` function
4. Add tests in `/tests/llm_translation/test_vllm.py`

See Gemini implementation:
- `/litellm/llms/vertex_ai/gemini/transformation.py` lines 463-472
