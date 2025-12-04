import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# vLLM - Structured Output & JSON Schema

This guide covers how to get structured JSON output from vLLM models that **don't natively support** OpenAI's `response_format` or `tools` parameters.

## ⚠️ Current Implementation Status

**IMPORTANT**: As of this writing, LiteLLM does **NOT** automatically inject JSON schema into prompts for vLLM models that don't support `response_format`.

### What Currently Exists

- ✅ **Vertex AI Gemini** - Has automatic prompt injection when model doesn't support `response_schema`
- ✅ **Function calling fallback** - Some providers can convert to tool calls
- ❌ **vLLM automatic injection** - Not yet implemented

### What Happens Now with vLLM

```python
# If you do this with a vLLM model that doesn't support response_format:
response = litellm.completion(
    model="hosted_vllm/my-model",
    messages=[{"role": "user", "content": "Extract data"}],
    response_format={"type": "json_schema", "json_schema": {...}}
)

# What actually happens:
# 1. LiteLLM inherits OpenAI behavior (HostedVLLMChatConfig extends OpenAIGPTConfig)
# 2. response_format is passed straight through to vLLM server
# 3. If vLLM model doesn't support it → ignored or error
# 4. NO automatic prompt injection occurs
```

### Code Location

The automatic prompt injection that works for Gemini is in:
- `/litellm/llms/vertex_ai/gemini/transformation.py` lines 463-472

```python
# This logic ONLY exists for Vertex AI Gemini:
if "response_schema" in optional_params:
    supports_response_schema = get_supports_response_schema(
        model=model, custom_llm_provider=custom_llm_provider
    )
    if supports_response_schema is False:
        user_response_schema_message = response_schema_prompt(
            model=model, response_schema=optional_params.get("response_schema")
        )
        messages.append({"role": "user", "content": user_response_schema_message})
        optional_params.pop("response_schema")
```

**This logic does NOT exist in** `/litellm/llms/hosted_vllm/chat/transformation.py`

---

## Current Workarounds for vLLM

Until automatic injection is implemented, here are your options:

### Option 1: Manual Prompt Injection (Recommended)

Manually add the JSON schema to your messages:

<Tabs>
<TabItem value="basic" label="Basic Example">

```python
import litellm
import json

# Enable dropping unsupported params
litellm.drop_params = True

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "age", "email"]
}

# Manually add schema instructions to messages
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that outputs valid JSON only."
    },
    {
        "role": "user",
        "content": "Extract user information: John Doe, 30 years old, john@example.com"
    },
    {
        "role": "user",
        "content": f"""You must respond with valid JSON matching this exact schema:

```json
{json.dumps(schema, indent=2)}
```

Important:
- Output ONLY the JSON object
- Do not include markdown code blocks
- Do not add explanations
- All required fields must be present"""
    }
]

response = litellm.completion(
    model="hosted_vllm/my-model",
    api_base="http://localhost:8000",
    messages=messages,
    temperature=0
)

# Extract JSON from response
result = json.loads(response.choices[0].message.content)
print(result)
```

</TabItem>
<TabItem value="pydantic" label="With Pydantic">

```python
import litellm
import json
from pydantic import BaseModel, Field
from typing import List

class ProductReview(BaseModel):
    rating: int = Field(ge=1, le=5, description="Rating from 1-5")
    sentiment: str = Field(description="positive, negative, or neutral")
    summary: str = Field(description="Brief summary")
    pros: List[str] = Field(description="List of positive points")
    cons: List[str] = Field(description="List of negative points")

# Convert Pydantic to JSON schema
schema = ProductReview.model_json_schema()

messages = [
    {
        "role": "user",
        "content": "Review: Great product! Love the quality. A bit pricey though."
    },
    {
        "role": "user",
        "content": f"""Respond with JSON matching this schema:

```json
{json.dumps(schema, indent=2)}
```

Output only valid JSON, no other text."""
    }
]

response = litellm.completion(
    model="hosted_vllm/llama-3-8b",
    api_base="http://localhost:8000",
    messages=messages
)

# Parse and validate with Pydantic
result = ProductReview.model_validate_json(response.choices[0].message.content)
print(result)
```

</TabItem>
</Tabs>

### Option 2: Helper Function

Create a reusable helper:

```python
import litellm
import json
from typing import Union, Type, Dict
from pydantic import BaseModel

def completion_with_schema(
    model: str,
    messages: list,
    schema: Union[Type[BaseModel], dict],
    api_base: str = None,
    **kwargs
):
    """
    Workaround for vLLM models without native response_format support.
    Manually injects JSON schema into the messages.
    """
    # Convert Pydantic to dict if needed
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_dict = schema.model_json_schema()
        pydantic_class = schema
    else:
        schema_dict = schema
        pydantic_class = None

    # Add schema instruction to messages
    schema_message = {
        "role": "user",
        "content": f"""Output valid JSON matching this schema:

```json
{json.dumps(schema_dict, indent=2)}
```

Rules:
- Output ONLY the JSON object
- No markdown formatting
- No explanations
- Match the schema exactly"""
    }

    # Clone messages and add schema
    messages_with_schema = messages + [schema_message]

    # Call LiteLLM with drop_params enabled
    litellm.drop_params = True
    response = litellm.completion(
        model=model,
        api_base=api_base,
        messages=messages_with_schema,
        temperature=kwargs.get("temperature", 0),
        **{k: v for k, v in kwargs.items() if k != "temperature"}
    )

    # Parse response
    content = response.choices[0].message.content

    # Try to extract JSON if wrapped in markdown
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract from code block
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            # Try to find any JSON object
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                raise ValueError(f"No valid JSON found in response: {content}")

    # Validate with Pydantic if provided
    if pydantic_class:
        return pydantic_class.model_validate(result)

    return result

# Usage
class User(BaseModel):
    name: str
    age: int
    email: str

result = completion_with_schema(
    model="hosted_vllm/my-model",
    api_base="http://localhost:8000",
    messages=[
        {"role": "user", "content": "Extract: John Doe, 30, john@example.com"}
    ],
    schema=User
)

print(result)  # User(name='John Doe', age=30, email='john@example.com')
```

### Option 3: Use vLLM's Native Support (If Available)

Some newer vLLM versions and certain models DO support `guided_json` or `response_format`:

```python
# Check if your vLLM server supports it
response = litellm.completion(
    model="hosted_vllm/qwen2.5-coder",  # Example model with support
    api_base="http://localhost:8000",
    messages=[{"role": "user", "content": "Extract data"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        }
    }
)
# If this works without error, your model supports it natively
```

Check vLLM documentation for models with `guided_decoding` support:
- https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters

---

## What Would Need to Be Implemented

To make automatic prompt injection work for vLLM (like it does for Gemini), the following changes would be needed:

### 1. Add Support Check to vLLM Transformation

File: `/litellm/llms/hosted_vllm/chat/transformation.py`

```python
from litellm.utils import supports_response_schema
from litellm.litellm_core_utils.prompt_templates.factory import response_schema_prompt

class HostedVLLMChatConfig(OpenAIGPTConfig):

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # Existing transform logic...

        # NEW: Check for response_schema support
        if "response_schema" in optional_params:
            _supports = supports_response_schema(
                model=model,
                custom_llm_provider="hosted_vllm"
            )
            if _supports is False:
                # Inject schema into messages
                schema_message = response_schema_prompt(
                    model=model,
                    response_schema=optional_params.get("response_schema")
                )
                messages.append({
                    "role": "user",
                    "content": schema_message
                })
                optional_params.pop("response_schema")

        # Continue with rest of transformation...
```

### 2. Add Model Registry Support

Users would need to register their models with feature flags:

```python
litellm.register_model({
    "hosted_vllm/my-llama-model": {
        "supports_function_calling": False,
        "supports_response_schema": False,
    }
})
```

### 3. Add Custom Prompt Templates

Support for customizing the injection format:

```python
litellm.custom_prompt_dict = {
    "hosted_vllm/my-model/response_schema_prompt": {
        "roles": {
            "user": {
                "pre_message": "Output JSON with this schema:\n",
                "post_message": "\n\nOutput valid JSON only."
            }
        }
    }
}
```

---

## Testing Your Setup

### Check What Your Model Supports

```python
import litellm

model = "hosted_vllm/my-model"

# Check feature support
print(f"Response schema: {litellm.supports_response_schema(model)}")
print(f"Function calling: {litellm.supports_function_calling(model)}")

# Get supported parameters
params = litellm.get_supported_openai_params(model)
print(f"Supported params: {params}")

# Note: These will return True for hosted_vllm because it inherits
# from OpenAI config, but your actual model might not support them
```

### Test JSON Output Quality

```python
import litellm
import json

def test_json_output(model: str, api_base: str):
    """Test if model can follow JSON schema instructions"""

    schema = {
        "type": "object",
        "properties": {
            "colors": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3
            }
        },
        "required": ["colors"]
    }

    messages = [
        {"role": "user", "content": "List 3 primary colors"},
        {
            "role": "user",
            "content": f"Respond with JSON: {json.dumps(schema)}"
        }
    ]

    response = litellm.completion(
        model=model,
        api_base=api_base,
        messages=messages,
        temperature=0
    )

    content = response.choices[0].message.content
    print(f"Raw response: {content}")

    try:
        result = json.loads(content)
        print(f"✅ Valid JSON: {result}")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False

# Test it
test_json_output(
    model="hosted_vllm/my-model",
    api_base="http://localhost:8000"
)
```

---

## Best Practices for Manual Injection

1. **Be Explicit in Instructions**
   ```python
   content = """Output ONLY a JSON object.
   Do not include:
   - Markdown code blocks (```json)
   - Explanations or comments
   - Text before or after the JSON

   Schema: {...}"""
   ```

2. **Use Low Temperature**
   ```python
   temperature=0  # More deterministic
   ```

3. **Add System Message**
   ```python
   messages = [
       {
           "role": "system",
           "content": "You are a JSON API. You only respond with valid JSON."
       },
       # ... rest of messages
   ]
   ```

4. **Handle Markdown Wrapping**
   ```python
   import re

   def extract_json(text: str) -> dict:
       try:
           return json.loads(text)
       except json.JSONDecodeError:
           # Try markdown block
           match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
           if match:
               return json.loads(match.group(1))
           raise
   ```

5. **Validate Responses**
   ```python
   from pydantic import BaseModel, ValidationError

   try:
       result = MySchema.model_validate_json(response_text)
   except ValidationError as e:
       print(f"Schema validation failed: {e}")
   ```

---

## Comparison: What Works Where

| Feature | Vertex AI Gemini | vLLM (hosted) | Status |
|---------|------------------|---------------|--------|
| **Automatic prompt injection** | ✅ Yes | ❌ No | Gemini only |
| **Manual prompt injection** | ✅ Yes | ✅ Yes | Works everywhere |
| **Native response_format** | ⚠️ Some models | ⚠️ Some models | Model-dependent |
| **Custom prompt templates** | ✅ Yes | ❌ No | Needs implementation |
| **Feature detection** | ✅ Yes | ⚠️ Inherits OpenAI | May be inaccurate |

---

## Related Resources

- [vLLM Guided Decoding Docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters)
- [LiteLLM JSON Mode (General)](../completion/json_mode.md)
- [LiteLLM Function Calling](../completion/function_call.md)
- [vLLM Provider Overview](./vllm.md)

---

## Contributing

If you'd like to implement automatic prompt injection for vLLM, the implementation would involve:

1. Modifying `/litellm/llms/hosted_vllm/chat/transformation.py`
2. Adding support check logic (similar to Gemini)
3. Using the existing `response_schema_prompt()` function
4. Adding tests in `/tests/llm_translation/test_vllm.py`

The core logic already exists in `/litellm/litellm_core_utils/prompt_templates/factory.py` - it just needs to be wired up for vLLM.

See the Gemini implementation as a reference:
- `/litellm/llms/vertex_ai/gemini/transformation.py` lines 463-472
