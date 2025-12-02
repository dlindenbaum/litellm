import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# vLLM - Structured Output & JSON Schema

This guide covers how to get structured JSON output from vLLM models that **don't natively support** OpenAI's `response_format` or `tools` parameters.

## When Do You Need This?

You need custom template configuration when:

1. Your vLLM model doesn't support native JSON schema (`response_format`)
2. Your vLLM model doesn't support function/tool calling
3. You want to customize how JSON schema instructions are presented to the model
4. You're using older or simpler models that lack structured output capabilities

## How It Works

LiteLLM has a **fallback mechanism** that injects JSON schema instructions directly into your conversation when the model doesn't support native structured output:

```
User Request:
- messages: [{"role": "user", "content": "Extract user info"}]
- response_format: {JSON schema}

       ↓

LiteLLM detects model doesn't support response_format

       ↓

Adds schema to messages:
- messages: [
    {"role": "user", "content": "Extract user info"},
    {"role": "user", "content": "Use this JSON schema: {...}"}
  ]

       ↓

Sends to vLLM server
```

---

## Method 1: Register Model with Feature Flags (Recommended)

This is the cleanest approach for models you'll use repeatedly.

### Step 1: Register Your Model

```python
import litellm

# Register model with explicit feature support flags
litellm.register_model({
    "hosted_vllm/my-custom-model": {
        "max_tokens": 4096,
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "hosted_vllm",
        "mode": "chat",

        # Feature flags - set these to False for models without support
        "supports_function_calling": False,  # No tool/function calling
        "supports_response_schema": False,   # No native JSON schema
        "supports_vision": False,            # Set based on your model
        "supports_assistant_prefill": True,  # Most models support this
    }
})
```

### Step 2: Use Normally with response_format

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int
    email: str
    occupation: str

# LiteLLM will automatically inject schema into prompt
response = litellm.completion(
    model="hosted_vllm/my-custom-model",
    api_base="http://your-vllm-server:8000",
    messages=[
        {"role": "user", "content": "Extract: John Doe, 30, john@example.com, Engineer"}
    ],
    response_format=UserInfo  # Pydantic model
)

print(response.choices[0].message.content)
```

### Step 3: Or Use Dict Schema

```python
response = litellm.completion(
    model="hosted_vllm/my-custom-model",
    api_base="http://your-vllm-server:8000",
    messages=[
        {"role": "user", "content": "List three colors"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "color_list",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["colors"],
                "additionalProperties": False
            }
        }
    }
)
```

---

## Method 2: Custom Prompt Templates

For advanced control over how JSON schema is presented to your model.

### Understanding the Default Template

By default, LiteLLM uses this prompt:

```python
def default_response_schema_prompt(response_schema: dict) -> str:
    prompt_str = """Use this JSON schema:
    ```json
    {}
    ```""".format(response_schema)
    return prompt_str
```

This gets **appended as a user message** after your original messages.

### Creating a Custom Template

You can customize this template to better match your model's training:

<Tabs>
<TabItem value="global" label="Global Custom Template">

```python
import litellm

# Apply to all models that use response_schema fallback
litellm.custom_prompt_dict = {
    "response_schema_prompt": {
        "roles": {
            "system": {
                "pre_message": "",
                "post_message": ""
            },
            "user": {
                "pre_message": "You are a helpful assistant that outputs valid JSON.\n\n",
                "post_message": "\n\nIMPORTANT: Your response must be ONLY valid JSON matching the schema above. Do not include markdown code blocks or explanations."
            }
        },
        "initial_prompt_value": "RESPOND WITH VALID JSON ONLY.\n\n",
        "final_prompt_value": "\n\nRemember: Output ONLY the JSON object, nothing else."
    }
}

# Now all models without native support will use this template
response = litellm.completion(
    model="hosted_vllm/my-custom-model",
    messages=[{"role": "user", "content": "Extract user data"}],
    response_format={...}
)
```

</TabItem>
<TabItem value="model-specific" label="Model-Specific Template">

```python
import litellm

# Apply only to specific model
litellm.custom_prompt_dict = {
    "hosted_vllm/my-custom-model/response_schema_prompt": {
        "roles": {
            "user": {
                "pre_message": "### JSON Schema\n\n",
                "post_message": "\n\n### Instructions\n- Output valid JSON only\n- Match the schema exactly\n- Use double quotes for strings\n- Do not add comments or markdown"
            }
        },
        "initial_prompt_value": "",
        "final_prompt_value": ""
    }
}

response = litellm.completion(
    model="hosted_vllm/my-custom-model",
    messages=[{"role": "user", "content": "Extract data"}],
    response_format={...}
)
```

</TabItem>
</Tabs>

### Example: Structured Template for Better Results

```python
import litellm

# Register model
litellm.register_model({
    "hosted_vllm/llama-3-8b": {
        "supports_function_calling": False,
        "supports_response_schema": False,
    }
})

# Custom template optimized for instruction-following models
litellm.custom_prompt_dict = {
    "hosted_vllm/llama-3-8b/response_schema_prompt": {
        "roles": {
            "user": {
                "pre_message": (
                    "=== JSON OUTPUT REQUIRED ===\n\n"
                    "You must respond with a valid JSON object that strictly follows this schema:\n\n"
                ),
                "post_message": (
                    "\n\n=== OUTPUT RULES ===\n"
                    "1. Output ONLY the JSON object\n"
                    "2. Do NOT wrap in markdown code blocks\n"
                    "3. Do NOT add explanations or comments\n"
                    "4. Use proper JSON syntax (double quotes, commas, brackets)\n"
                    "5. Match the schema exactly - all required fields must be present\n"
                )
            }
        },
        "initial_prompt_value": "",
        "final_prompt_value": ""
    }
}

# Use it
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    rating: int = Field(description="Rating from 1-5")
    sentiment: str = Field(description="positive, negative, or neutral")
    summary: str = Field(description="Brief summary of the review")
    key_points: list[str] = Field(description="List of key points mentioned")

response = litellm.completion(
    model="hosted_vllm/llama-3-8b",
    api_base="http://localhost:8000",
    messages=[
        {
            "role": "user",
            "content": "Review: This product is amazing! Great quality, fast shipping, and excellent customer service. Highly recommend. 5 stars!"
        }
    ],
    response_format=ProductReview
)

print(response.choices[0].message.content)
```

---

## Method 3: Manual Prompt Injection (Simplest)

For one-off requests or maximum control:

```python
import litellm
import json

# Tell LiteLLM to drop unsupported params silently
litellm.drop_params = True

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age", "email"]
}

messages = [
    {
        "role": "user",
        "content": "Extract user information: John Doe, 30 years old, john@example.com"
    },
    {
        "role": "user",
        "content": f"Respond with JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```"
    }
]

response = litellm.completion(
    model="hosted_vllm/my-model",
    api_base="http://localhost:8000",
    messages=messages
)
```

---

## Complete Working Example

Here's a full example you can run:

```python
import litellm
from pydantic import BaseModel, Field
from typing import List
import json

# 1. Register your vLLM model
litellm.register_model({
    "hosted_vllm/my-llama-model": {
        "max_tokens": 4096,
        "litellm_provider": "hosted_vllm",
        "mode": "chat",
        "supports_function_calling": False,
        "supports_response_schema": False,
    }
})

# 2. Define custom template for better JSON output
litellm.custom_prompt_dict = {
    "hosted_vllm/my-llama-model/response_schema_prompt": {
        "roles": {
            "user": {
                "pre_message": "📋 JSON SCHEMA REQUIRED:\n\n",
                "post_message": (
                    "\n\n✅ OUTPUT REQUIREMENTS:\n"
                    "- Valid JSON only (no markdown, no explanations)\n"
                    "- Match schema exactly\n"
                    "- Include all required fields\n"
                )
            }
        },
        "initial_prompt_value": "",
        "final_prompt_value": ""
    }
}

# 3. Define your data structure
class Article(BaseModel):
    title: str = Field(description="Article title")
    author: str = Field(description="Author name")
    tags: List[str] = Field(description="List of relevant tags")
    word_count: int = Field(description="Approximate word count")
    summary: str = Field(description="Brief 1-sentence summary")

# 4. Make the request
try:
    response = litellm.completion(
        model="hosted_vllm/my-llama-model",
        api_base="http://localhost:8000",
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze this article: "
                    "'The Future of AI: A Deep Dive into Large Language Models. "
                    "By Dr. Jane Smith. "
                    "This comprehensive 2000-word article explores the latest developments "
                    "in artificial intelligence, focusing on transformer architectures, "
                    "training methodologies, and ethical considerations.'"
                )
            }
        ],
        response_format=Article,
        temperature=0.7,
        max_tokens=500
    )

    # Parse the response
    result = json.loads(response.choices[0].message.content)
    print("Structured output:")
    print(json.dumps(result, indent=2))

except Exception as e:
    print(f"Error: {e}")
```

---

## Configuration with LiteLLM Proxy

You can also configure this in your proxy `config.yaml`:

```yaml
model_list:
  - model_name: my-vllm-model
    litellm_params:
      model: hosted_vllm/llama-3-8b
      api_base: http://localhost:8000
      supports_function_calling: false
      supports_response_schema: false

general_settings:
  # Enable prompt injection for unsupported models
  add_function_to_prompt: true

  # Custom prompts (optional)
  custom_prompt_dict:
    "hosted_vllm/llama-3-8b/response_schema_prompt":
      roles:
        user:
          pre_message: "JSON Schema Required:\n\n"
          post_message: "\n\nOutput valid JSON only."
```

Then use via proxy:

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="my-vllm-model",
    messages=[{"role": "user", "content": "Extract data..."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "schema": {...}
        }
    }
)
```

---

## Validation & Error Handling

### Enable Client-Side Validation

```python
import litellm

# Validate responses match schema
litellm.enable_json_schema_validation = True

try:
    response = litellm.completion(
        model="hosted_vllm/my-model",
        messages=[...],
        response_format=MySchema
    )
except litellm.exceptions.JSONSchemaValidationError as e:
    print(f"Response didn't match schema: {e}")
```

### Handle Markdown Wrapping

Some models wrap JSON in markdown code blocks:

```python
import json
import re

def extract_json(text: str) -> dict:
    """Extract JSON from text that might have markdown code blocks"""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try to find any JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    raise ValueError("No valid JSON found in response")

# Use it
response = litellm.completion(...)
result = extract_json(response.choices[0].message.content)
```

---

## Troubleshooting

### Model ignores JSON schema

**Solution**: Make your custom template more explicit:

```python
litellm.custom_prompt_dict = {
    "response_schema_prompt": {
        "roles": {
            "user": {
                "pre_message": (
                    "CRITICAL INSTRUCTION: You MUST respond with ONLY a JSON object.\n"
                    "Required JSON Schema:\n\n"
                ),
                "post_message": (
                    "\n\nYour response MUST:\n"
                    "1. Be ONLY a JSON object (start with { and end with })\n"
                    "2. NOT include any text before or after the JSON\n"
                    "3. NOT use markdown code blocks\n"
                    "4. Match the schema exactly\n\n"
                    "Begin your JSON response now:"
                )
            }
        }
    }
}
```

### Response has extra text

Try adding a system message:

```python
messages = [
    {"role": "system", "content": "You are a JSON-only API. You respond exclusively with valid JSON objects."},
    {"role": "user", "content": "..."}
]
```

### Model doesn't support required schema fields

Use `temperature=0` for more consistent output:

```python
response = litellm.completion(
    model="...",
    messages=[...],
    response_format={...},
    temperature=0,  # More deterministic
    top_p=0.95,
    max_tokens=1000
)
```

---

## Checking Feature Support

Verify what your model supports:

```python
import litellm

model = "hosted_vllm/my-model"

# Check specific features
print(f"Function calling: {litellm.supports_function_calling(model)}")
print(f"Response schema: {litellm.supports_response_schema(model)}")
print(f"Vision: {litellm.supports_vision(model)}")

# Get all supported OpenAI params
params = litellm.get_supported_openai_params(model)
print(f"Supported params: {params}")
```

---

## Best Practices

1. **Use Pydantic models** - They provide better type safety and validation
2. **Keep schemas simple** - Complex nested schemas are harder for models to follow
3. **Set temperature=0** - More deterministic for structured output
4. **Add examples** - Include example JSON in your prompt for better results
5. **Validate responses** - Always validate with `litellm.enable_json_schema_validation = True`
6. **Test your template** - Different models respond better to different instruction styles
7. **Consider fine-tuning** - For production use, fine-tune your model on structured output tasks

---

## Related Documentation

- [JSON Mode (General)](../completion/json_mode.md)
- [Function Calling](../completion/function_call.md)
- [vLLM Provider Overview](./vllm.md)
- [Custom Prompt Templates](./vllm.md#custom-prompt-templates)
