# Model Router Python SDK (`modelrouter-sdk`)

Lightweight, drop-in Python client for the **AI Model Router** platform. Replace standard OpenAI API client calls with one line of code to get intelligent, cost-optimized LLM request routing, automated tiered fallbacks, and real-time observability.

## Installation

```bash
pip install modelrouter-sdk
```

## Quickstart

### Synchronous Usage
```python
from modelrouter import ModelRouter

client = ModelRouter(
    api_key="mr_live_...",
    base_url="http://localhost:8000",
    policy="balanced" # "balanced" | "cost_optimized" | "speed_optimized" | "quality_first"
)

# Drop-in OpenAI chat completion syntax
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Write a Python script to calculate Fibonacci numbers."}
    ]
)

print("Routed to:", response.model)
print("Output:", response.choices[0].message.content)
print("Estimated Cost:", response.usage.estimated_cost_usd)
```

### Streaming Responses
```python
stream = client.chat.completions.create(
    prompt="Explain quantum computing simply.",
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### Asynchronous Client
```python
import asyncio
from modelrouter import AsyncModelRouter

async def main():
    async_client = AsyncModelRouter(base_url="http://localhost:8000")
    
    # Dry-run route evaluation
    decision = await async_client.route("Sort a list in Python")
    print(f"Selected Model: {decision.selected_model_name} (Confidence: {decision.confidence*100:.0f}%)")
    
    # Execute routed request
    resp = await async_client.chat.complete(prompt="Write a hello world in Rust")
    print(resp.choices[0].message.content)

asyncio.run(main())
```

The async client also supports `async with AsyncModelRouter(...) as client:`.
It returns the same client and awaits `close()` on normal or exceptional exit,
without suppressing exceptions. HTTP connections currently belong to each request
and close there; `close()` is safe to call repeatedly and owns no persistent pool.


### Provider and gateway errors

HTTP 503 raises `ProviderUnavailableError`; HTTP 504 raises `GatewayTimeoutError`.
Both are exported from `modelrouter` and inherit `APIStatusError`, so existing
`except APIStatusError` handlers still catch them. Their `status_code` and
`response_body` follow the existing response handling: JSON objects are parsed,
while other bodies remain text. The mapping also applies to streaming requests
and health checks. Other health response behavior is unchanged.
