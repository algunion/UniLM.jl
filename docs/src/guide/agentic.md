# [Agentic Workflows](@id agentic_guide)

`respond` is the unified *agentic verb*. The same call targets OpenAI's Responses
API by default, or Google's Gemini Interactions API by setting
`service=GEMINIServiceEndpoint` — identical inputs, tools, lifecycle, and
usage/cost accounting.

```@setup agentic
using UniLM
```

## One call, two providers

```@example agentic
result = respond("Explain multiple dispatch in one sentence.")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

Swap the provider with a single keyword:

```@example agentic
result = respond("Explain multiple dispatch in one sentence."; service=GEMINIServiceEndpoint)
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

## Hosted tools (Gemini)

Gemini Interactions exposes server-side hosted tools via
[`gemini_google_search`](@ref), [`gemini_code_execution`](@ref), and
[`gemini_url_context`](@ref) — pass them in `tools=`:

```@example agentic
result = respond("What are the latest stable Julia releases?";
                 service=GEMINIServiceEndpoint, tools=[gemini_google_search()])
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", output_text(result))
end
```

## Constraining tool choice

Force a specific function with [`tool_choice_function`](@ref) (works on both
providers). The other builders — [`tool_choice_hosted`](@ref),
[`tool_choice_mcp`](@ref), [`tool_choice_custom`](@ref),
[`tool_choice_allowed`](@ref) — are OpenAI-Responses selectors and raise an
error on Gemini.

```julia
respond("What's the weather in Paris?";
        tools=[my_function_tool],
        tool_choice=tool_choice_function("get_weather"))
```

## Automated tool loop

The [`tool_loop`](@ref) driver (see the [Tool Calling guide](@ref tools_guide))
runs the call/execute/respond cycle automatically, and works across providers —
add `service=` to target Gemini:

```julia
ct = CallableTool(function_tool("get_weather", "Get weather",
        parameters=Dict("type" => "object",
                        "properties" => Dict("location" => Dict("type" => "string")))),
    (name, args) -> "22C, sunny")
result = tool_loop("What's the weather in Paris?"; service=GEMINIServiceEndpoint, tools=[ct])
# result.completed == true when the model returns a final text answer
```

## Feeding tool output back manually

Use [`tool_result`](@ref) to return a function's output on the next turn:

```julia
r2 = respond(; service=GEMINIServiceEndpoint,
             previous_response_id=r1.response.id,
             input=[tool_result("call_abc", "get_weather", "72F and sunny")])
```

## Lifecycle (background requests)

[`get_response`](@ref) and [`cancel_response`](@ref) take a `service=` keyword,
so background Gemini Interactions are managed the same way as OpenAI:

```julia
status = get_response("<interaction_id>"; service=GEMINIServiceEndpoint)
cancel_response("<interaction_id>"; service=GEMINIServiceEndpoint)
```

## Usage & cost (cross-provider)

`token_usage` and `estimated_cost` work for Gemini too — the Interactions
decoder normalizes usage into the shared shape and `DEFAULT_PRICING` includes
`gemini-3.5-flash` (hosted-tool per-call fees are not modeled). (These accessors
are shown in plain code font, not `@ref`: their API-reference `@docs` blocks are
deferred to a later cost-tracking pass.)

```@example agentic
r = respond("What is 2+2?"; service=GEMINIServiceEndpoint)
if r isa ResponseSuccess
    println("usage: ", token_usage(r))
    println("est. cost: \$", round(estimated_cost(r); digits=6))
else
    println("Request failed — ", output_text(r))
end
```

## See Also

- [Responses API](@ref responses_guide) — OpenAI-specific `respond`/`Respond` details
- [Tool Calling](@ref tools_guide) — defining tools and the tool loop
- [Multi-Backend](@ref backend_guide) — provider setup and capabilities
