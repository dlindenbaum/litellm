# vLLM JSON Schema with LiteLLM Embedded Router

This guide shows how to configure the **embedded LiteLLM Router** (not the proxy server) to handle JSON schema for vLLM models that don't support native `response_format`.

## Overview

The embedded Router is used directly in your Python code, unlike the proxy which runs as a standalone server. This gives you more programmatic control.

**Key Differences:**
- **Proxy**: `litellm --config config.yaml` (separate HTTP server)
- **Router**: `from litellm import Router` (embedded in your code)

---

## Method 1: Load Configuration from YAML

You can load your router configuration from a YAML file and use it programmatically.

### Step 1: Create `router_config.yaml`

```yaml
model_list:
  - model_name: llama-3-8b
    litellm_params:
      model: hosted_vllm/llama-3-8b
      api_base: http://localhost:8000
      api_key: none
    model_info:
      mode: chat
      max_tokens: 4096
      supports_response_schema: false
      supports_function_calling: false

  - model_name: qwen-coder
    litellm_params:
      model: hosted_vllm/qwen2.5-coder-7b
      api_base: http://localhost:8000
    model_info:
      supports_response_schema: true  # This model supports it!

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 2
  timeout: 60

litellm_settings:
  drop_params: false  # We'll handle via callback
  set_verbose: true
```

### Step 2: Create Router with Callback

```python
import yaml
import json
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from typing import Optional

# Custom callback to inject JSON schema
class RouterJSONSchemaInjector(CustomLogger):
    """
    Injects JSON schema into messages for models that don't support response_format.
    """

    def __init__(self, models_needing_injection: set = None):
        super().__init__()
        # Models that need schema injection (by model_name)
        self.models_needing_injection = models_needing_injection or {
            "llama-3-8b",
            # Add your model names here
        }

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: dict,
        data: dict,
        call_type: str,
    ):
        """Called before each LLM request"""
        if call_type not in ["completion", "acompletion"]:
            return data

        model = data.get("model", "")
        if model not in self.models_needing_injection:
            return data

        # Check for response_format
        response_format = data.get("response_format")
        if not response_format:
            return data

        # Extract schema
        schema = self._extract_schema(response_format)
        if not schema:
            return data

        # Create and inject schema message
        schema_message = self._create_schema_message(schema)
        messages = data.get("messages", [])
        messages.append({
            "role": "user",
            "content": schema_message
        })
        data["messages"] = messages

        # Remove response_format (not supported)
        data.pop("response_format", None)

        print(f"✅ Injected JSON schema into {model}")
        return data

    def _extract_schema(self, response_format: dict) -> Optional[dict]:
        """Extract schema from response_format"""
        if "json_schema" in response_format:
            return response_format["json_schema"].get("schema")
        elif "response_schema" in response_format:
            return response_format["response_schema"]
        return None

    def _create_schema_message(self, schema: dict) -> str:
        """Create the schema instruction message"""
        return f\"\"\"You must respond with valid JSON matching this exact schema:

```json
{json.dumps(schema, indent=2)}
```

IMPORTANT RULES:
- Output ONLY the JSON object (start with {{ and end with }})
- Do NOT include markdown code blocks (no ```json)
- Do NOT add any explanations before or after the JSON
- All required fields must be present
- Use the exact property names from the schema\"\"\"


# Load configuration from YAML
with open("router_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract config sections
model_list = config.get("model_list", [])
router_settings = config.get("router_settings", {})
litellm_settings = config.get("litellm_settings", {})

# Get models needing injection from model_info
models_needing_injection = {
    model["model_name"]
    for model in model_list
    if model.get("model_info", {}).get("supports_response_schema") is False
}

print(f"Models needing schema injection: {models_needing_injection}")

# Create callback
schema_injector = RouterJSONSchemaInjector(
    models_needing_injection=models_needing_injection
)

# Create router with callback
router = Router(
    model_list=model_list,
    routing_strategy=router_settings.get("routing_strategy", "simple-shuffle"),
    num_retries=router_settings.get("num_retries", 0),
    timeout=router_settings.get("timeout", 60),
    set_verbose=litellm_settings.get("set_verbose", False),
)

# Register callback
import litellm
litellm.callbacks = [schema_injector]

print("✅ Router initialized with JSON schema injection callback")
```

### Step 3: Use the Router

```python
from pydantic import BaseModel, Field

class UserInfo(BaseModel):
    name: str
    age: int
    email: str
    occupation: str

# Use router with response_format - schema will be auto-injected!
response = await router.acompletion(
    model="llama-3-8b",  # Model without native support
    messages=[
        {"role": "user", "content": "Extract: Jane Smith, 25, jane@example.com, Engineer"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "schema": UserInfo.model_json_schema()
        }
    }
)

# Parse result
result = UserInfo.model_validate_json(response.choices[0].message.content)
print(result)
```

---

## Method 2: Pure Python Configuration

If you prefer to avoid YAML and configure everything in Python:

```python
import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
import json

# Define model list directly
model_list = [
    {
        "model_name": "llama-3-8b",
        "litellm_params": {
            "model": "hosted_vllm/llama-3-8b",
            "api_base": "http://localhost:8000",
            "api_key": "none",
        },
        "model_info": {
            "mode": "chat",
            "max_tokens": 4096,
            "supports_response_schema": False,
            "supports_function_calling": False,
        }
    },
    {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": "gpt-4",
            "api_key": "os.environ/OPENAI_API_KEY",
        },
        "model_info": {
            "supports_response_schema": True,
        }
    }
]

# Custom callback (same as above)
class RouterJSONSchemaInjector(CustomLogger):
    def __init__(self, models_needing_injection: set):
        super().__init__()
        self.models_needing_injection = models_needing_injection

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        # Same implementation as above
        if call_type not in ["completion", "acompletion"]:
            return data

        model = data.get("model", "")
        if model not in self.models_needing_injection:
            return data

        response_format = data.get("response_format")
        if not response_format:
            return data

        # Extract and inject schema
        schema = None
        if isinstance(response_format, dict):
            if "json_schema" in response_format:
                schema = response_format["json_schema"].get("schema")
            elif "response_schema" in response_format:
                schema = response_format["response_schema"]

        if schema:
            schema_message = f\"\"\"You must respond with valid JSON matching this exact schema:

```json
{json.dumps(schema, indent=2)}
```

IMPORTANT RULES:
- Output ONLY the JSON object
- No markdown code blocks
- No explanations
- All required fields must be present\"\"\"

            messages = data.get("messages", [])
            messages.append({"role": "user", "content": schema_message})
            data["messages"] = messages
            data.pop("response_format", None)

        return data

# Determine which models need injection
models_needing_injection = {
    model["model_name"]
    for model in model_list
    if model.get("model_info", {}).get("supports_response_schema") is False
}

# Create and register callback
schema_injector = RouterJSONSchemaInjector(
    models_needing_injection=models_needing_injection
)
litellm.callbacks = [schema_injector]

# Create router
router = Router(
    model_list=model_list,
    routing_strategy="simple-shuffle",
    num_retries=2,
    timeout=60,
    set_verbose=True,
)

print(f"✅ Router initialized")
print(f"   Models needing injection: {models_needing_injection}")
print(f"   Callbacks registered: {len(litellm.callbacks)}")
```

---

## Method 3: Per-Model Custom Templates

Customize schema messages per model using a template registry:

```python
class RouterJSONSchemaInjector(CustomLogger):
    def __init__(self, models_needing_injection: set, templates: dict = None):
        super().__init__()
        self.models_needing_injection = models_needing_injection
        self.templates = templates or {}

    def _create_schema_message(self, schema: dict, model: str) -> str:
        """Create schema message using custom template if available"""

        # Check for model-specific template
        if model in self.templates:
            template = self.templates[model]
            pre = template.get("pre_message", "")
            post = template.get("post_message", "")
            return f"{pre}\n```json\n{json.dumps(schema, indent=2)}\n```\n{post}"

        # Default template
        return f\"\"\"You must respond with valid JSON matching this exact schema:

```json
{json.dumps(schema, indent=2)}
```

IMPORTANT RULES:
- Output ONLY the JSON object
- No markdown code blocks
- No explanations
- All required fields must be present\"\"\"

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type not in ["completion", "acompletion"]:
            return data

        model = data.get("model", "")
        if model not in self.models_needing_injection:
            return data

        response_format = data.get("response_format")
        if not response_format:
            return data

        # Extract schema
        schema = None
        if isinstance(response_format, dict):
            if "json_schema" in response_format:
                schema = response_format["json_schema"].get("schema")

        if schema:
            # Use custom template for this model
            schema_message = self._create_schema_message(schema, model)

            messages = data.get("messages", [])
            messages.append({"role": "user", "content": schema_message})
            data["messages"] = messages
            data.pop("response_format", None)

        return data

# Define custom templates per model
templates = {
    "llama-3-8b": {
        "pre_message": "### Task\nGenerate JSON output conforming to this schema:",
        "post_message": "\n### Requirements\n1. Valid JSON only\n2. Match schema exactly\n3. No markdown formatting"
    },
    "qwen-7b": {
        "pre_message": "<|im_start|>system\nOutput JSON with this structure:",
        "post_message": "<|im_end|>"
    }
}

# Create callback with custom templates
schema_injector = RouterJSONSchemaInjector(
    models_needing_injection={"llama-3-8b", "qwen-7b"},
    templates=templates
)
litellm.callbacks = [schema_injector]

# Create router
router = Router(model_list=model_list)
```

---

## Method 4: Using Router with Model Groups

For production with multiple deployments per model:

```python
model_list = [
    # Group 1: Llama models (no response_schema support)
    {
        "model_name": "llama-chat",  # User-facing name
        "litellm_params": {
            "model": "hosted_vllm/llama-3-8b",
            "api_base": "http://vllm-server-1:8000",
        },
        "model_info": {
            "supports_response_schema": False,
        }
    },
    {
        "model_name": "llama-chat",  # Same group
        "litellm_params": {
            "model": "hosted_vllm/llama-3-8b",
            "api_base": "http://vllm-server-2:8000",  # Different server
        },
        "model_info": {
            "supports_response_schema": False,
        }
    },

    # Group 2: Models with native support
    {
        "model_name": "smart-model",
        "litellm_params": {
            "model": "gpt-4",
            "api_key": "os.environ/OPENAI_API_KEY",
        },
        "model_info": {
            "supports_response_schema": True,
        }
    },
]

# Automatically determine which model groups need injection
model_groups_needing_injection = set()
for model in model_list:
    if model.get("model_info", {}).get("supports_response_schema") is False:
        model_groups_needing_injection.add(model["model_name"])

print(f"Model groups needing injection: {model_groups_needing_injection}")

# Create callback
schema_injector = RouterJSONSchemaInjector(
    models_needing_injection=model_groups_needing_injection
)
litellm.callbacks = [schema_injector]

# Create router with load balancing
router = Router(
    model_list=model_list,
    routing_strategy="simple-shuffle",  # Load balance within groups
    num_retries=2,
    fallbacks=[
        {"llama-chat": ["smart-model"]}  # Fallback if llama fails
    ]
)

# Use it - will load balance between vllm-server-1 and vllm-server-2
response = await router.acompletion(
    model="llama-chat",
    messages=[...],
    response_format={...}  # Auto-injected!
)
```

---

## Complete Production Example

```python
import yaml
import json
import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from pydantic import BaseModel
from typing import Optional, Set, Dict

class RouterJSONSchemaInjector(CustomLogger):
    """Production-ready JSON schema injector for Router"""

    def __init__(
        self,
        models_needing_injection: Set[str],
        templates: Optional[Dict[str, dict]] = None,
        verbose: bool = False
    ):
        super().__init__()
        self.models_needing_injection = models_needing_injection
        self.templates = templates or {}
        self.verbose = verbose

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: dict,
        data: dict,
        call_type: str,
    ):
        if call_type not in ["completion", "acompletion"]:
            return data

        model = data.get("model", "")
        if model not in self.models_needing_injection:
            return data

        response_format = data.get("response_format")
        if not response_format:
            return data

        # Extract schema
        schema = self._extract_schema(response_format)
        if not schema:
            return data

        # Inject schema
        schema_message = self._create_schema_message(schema, model)
        messages = data.get("messages", [])
        messages.append({"role": "user", "content": schema_message})
        data["messages"] = messages
        data.pop("response_format", None)

        if self.verbose:
            print(f"✅ Injected JSON schema into model={model}")
            print(f"   Schema properties: {list(schema.get('properties', {}).keys())}")

        return data

    def _extract_schema(self, response_format: dict) -> Optional[dict]:
        if "json_schema" in response_format:
            return response_format["json_schema"].get("schema")
        elif "response_schema" in response_format:
            return response_format["response_schema"]
        return None

    def _create_schema_message(self, schema: dict, model: str) -> str:
        if model in self.templates:
            template = self.templates[model]
            pre = template.get("pre_message", "")
            post = template.get("post_message", "")
            schema_str = json.dumps(schema, indent=2)
            return f"{pre}\n```json\n{schema_str}\n```\n{post}"

        # Default template
        return f\"\"\"You must respond with valid JSON matching this exact schema:

```json
{json.dumps(schema, indent=2)}
```

IMPORTANT RULES:
- Output ONLY the JSON object (start with {{ and end with }})
- Do NOT include markdown code blocks (no ```json)
- Do NOT add any explanations before or after the JSON
- All required fields must be present
- Use the exact property names from the schema\"\"\"


def create_router_from_config(config_path: str, verbose: bool = False) -> Router:
    """
    Create a Router instance from a YAML config file with automatic
    JSON schema injection for models that don't support it.
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_list = config.get("model_list", [])
    router_settings = config.get("router_settings", {})
    litellm_settings = config.get("litellm_settings", {})

    # Determine which models need injection
    models_needing_injection = {
        model["model_name"]
        for model in model_list
        if model.get("model_info", {}).get("supports_response_schema") is False
    }

    # Load custom templates if provided
    templates = litellm_settings.get("custom_prompt_dict", {})

    # Create and register callback
    schema_injector = RouterJSONSchemaInjector(
        models_needing_injection=models_needing_injection,
        templates=templates,
        verbose=verbose or litellm_settings.get("set_verbose", False)
    )
    litellm.callbacks = [schema_injector]

    # Create router
    router = Router(
        model_list=model_list,
        routing_strategy=router_settings.get("routing_strategy", "simple-shuffle"),
        num_retries=router_settings.get("num_retries", 0),
        timeout=router_settings.get("timeout", 60),
        fallbacks=router_settings.get("fallbacks", []),
        set_verbose=litellm_settings.get("set_verbose", False),
    )

    if verbose:
        print(f"✅ Router created from {config_path}")
        print(f"   Total models: {len(model_list)}")
        print(f"   Models needing schema injection: {models_needing_injection}")
        print(f"   Routing strategy: {router_settings.get('routing_strategy')}")

    return router


# Usage
if __name__ == "__main__":
    # Create router from config
    router = create_router_from_config("router_config.yaml", verbose=True)

    # Use with Pydantic model
    class Article(BaseModel):
        title: str
        author: str
        summary: str

    # Make request
    response = await router.acompletion(
        model="llama-3-8b",
        messages=[
            {"role": "user", "content": "Summarize: AI and Machine Learning by John Doe..."}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "article",
                "schema": Article.model_json_schema()
            }
        }
    )

    # Parse and validate
    result = Article.model_validate_json(response.choices[0].message.content)
    print(result)
```

---

## Validation & Testing

### Test Schema Injection

```python
import asyncio

async def test_schema_injection():
    """Verify that schema injection is working"""

    # Simple test schema
    test_schema = {
        "type": "object",
        "properties": {
            "colors": {"type": "array", "items": {"type": "string"}},
            "count": {"type": "integer"}
        },
        "required": ["colors", "count"]
    }

    print("Testing schema injection...")

    response = await router.acompletion(
        model="llama-3-8b",
        messages=[
            {"role": "user", "content": "List 3 primary colors"}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "color_test",
                "schema": test_schema
            }
        }
    )

    # Validate response
    try:
        result = json.loads(response.choices[0].message.content)
        assert "colors" in result, "Missing 'colors' field"
        assert "count" in result, "Missing 'count' field"
        assert isinstance(result["colors"], list), "'colors' is not a list"
        assert isinstance(result["count"], int), "'count' is not an integer"
        print("✅ Schema injection test PASSED")
        print(f"   Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Schema injection test FAILED: {e}")
        print(f"   Raw response: {response.choices[0].message.content}")
        return False

# Run test
asyncio.run(test_schema_injection())
```

---

## Comparison: Router vs Proxy

| Feature | Router (Embedded) | Proxy (Server) |
|---------|-------------------|----------------|
| **Deployment** | In your Python app | Standalone HTTP server |
| **Configuration** | Python dict or YAML loaded in code | YAML config file |
| **Callbacks** | `litellm.callbacks = [...]` | `litellm_settings.callbacks` in YAML |
| **Schema Injection** | Custom callback required | Custom callback required |
| **Load Balancing** | ✅ Yes | ✅ Yes |
| **HTTP API** | ❌ No (direct function calls) | ✅ Yes |
| **Best For** | Python applications | Multi-language clients |

---

## Troubleshooting

### Callback not being called

```python
# Verify callback is registered
print(f"Callbacks registered: {litellm.callbacks}")

# Enable verbose logging
router = Router(model_list=model_list, set_verbose=True)
```

### Schema not being injected

```python
# Add debug logging to callback
class RouterJSONSchemaInjector(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        print(f"DEBUG: pre_call_hook called")
        print(f"  call_type: {call_type}")
        print(f"  model: {data.get('model')}")
        print(f"  has response_format: {'response_format' in data}")
        # ... rest of implementation
```

### Model name doesn't match

```python
# Make sure model_name in request matches model_name in config
# Request: model="llama-3-8b"
# Config: model_name: "llama-3-8b"  <- Must match exactly!
```

---

## Related Documentation

- [vLLM JSON Schema Guide](./vllm_json_schema.md) - Core concepts
- [vLLM JSON Schema Notebook](./vllm_json_schema_demo.ipynb) - Interactive examples
- [vLLM Router/Proxy YAML Config](./vllm_json_schema_router.md) - Proxy server approach
- [LiteLLM Router Docs](../../routing.md) - General router documentation
- [Custom Callbacks](https://docs.litellm.ai/docs/observability/custom_callback) - Writing callbacks
