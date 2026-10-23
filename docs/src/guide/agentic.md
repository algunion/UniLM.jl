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
result = respond("Explain multiple dispatch in one plain-text sentence.", model="gpt-5.4-mini")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
end
```

Swap the provider with a single keyword:

```@example agentic
result = respond("Explain multiple dispatch in one plain-text sentence."; service=GEMINIServiceEndpoint, model="gemini-3.8-flash")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
end
```

## Messages, images, audio and documents (Gemini)

The Interactions API takes its input as a list of steps, so `respond` sends each
[`InputMessage`](@ref) as one: a user message as a `user_input` step and an assistant
message as a `model_output` step (a Dict with a `role` is read the same way). A system or
developer message throws `ArgumentError` — send that guidance as `instructions` — and so
does a message `phase`, which only OpenAI's Responses API reads. A message's content is
text, or a vector of parts: [`input_text`](@ref) parts, and media as
[`ImageAttachment`](@ref), [`AudioAttachment`](@ref) and [`DocumentAttachment`](@ref)
values, each sent as an Interactions `image`, `audio` or `document` part holding its bytes;
[`input_file`](@ref)`(document)` and an [`input_image`](@ref) with a base64 data URL are sent
the same way. An image's `detail` is its `resolution` (`"low"` or `"high"`; `"auto"` leaves
the model's default; `"original"` throws). A web URL, an OpenAI file id and a cache
breakpoint have no equivalent here, and throw `ArgumentError` before any request.

```julia
r = respond(Respond(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
    input=[InputMessage(role="user", content=[input_text("What colour is the circle? One word."),
                                              ImageAttachment("circle.png")])]))
```

`service_tier` takes `"standard"`, `"flex"`, `"priority"` or `"deferred"`
([optimization](https://ai.google.dev/gemini-api/docs/optimization)), `seed` is sent as
`generation_config.seed`, and `safety_identifier` as the request label `safety_identifier`
(at most 63 lowercase letters, digits, underscores and dashes).

## Hosted tools (Gemini)

Control Gemini thinking with the same typed configuration used by OpenAI:

```julia
r = Respond(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", input="Explain recursion briefly.",
            reasoning=Reasoning(effort="low", summary="auto"), max_output_tokens=1024)
```

This maps to Interactions `generation_config.thinking_level` and
`thinking_summaries`. Gemini 4 Argon and Gemini 3.8 Flash support low, medium, and high
effort. Gemini ignores `temperature` and `top_p` on Gemini 3.6 Flash, 3.5 Flash-Lite and
every later model, so `respond` throws `ArgumentError` for them there, and Gemini 4 Argon
takes a `max_output_tokens` above 65,536 only with `background=true`
([long decode](https://ai.google.dev/gemini-api/docs/long-decode)); see
[Multi-Backend Support](@ref backend_guide) for every rule. Continue with
`previous_response_id` to preserve server state and signatures.
[Google thinking documentation](https://ai.google.dev/gemini-api/docs/thinking).

Gemini Interactions exposes server-side hosted tools via
[`gemini_google_search`](@ref), [`gemini_code_execution`](@ref), and
[`gemini_url_context`](@ref) — pass them in `tools=`:

```@example agentic
result = respond("What is the latest stable Julia release? Answer in one plain-text sentence.";
                 service=GEMINIServiceEndpoint, tools=[gemini_google_search()], model="gemini-3.8-flash")
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
end
```

## Constraining tool choice

Force a specific function with [`tool_choice_function`](@ref) (works on both
providers). The other builders — [`tool_choice_hosted`](@ref),
[`tool_choice_mcp`](@ref), [`tool_choice_custom`](@ref),
[`tool_choice_allowed`](@ref) — are OpenAI-Responses selectors: on Gemini,
`respond` throws `ArgumentError` for them before any request.

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

The loop takes the same `cancel` and `tool_concurrency` keywords on both providers; see
[Tool Calling](@ref tools_guide) for its stop rules.

!!! tip "Check it with Jev"
    Before the loop, one Jev question can score how much expertise a request needs and
    pick the cheap or the strong model for it ([Pick the model](@ref jev_llm_pick_model)).
    Inside it, wrap the dispatcher so that a sensitive tool call runs only if the user
    asked for it ([Approve tool calls](@ref jev_llm_tool_calls)).

## Feeding tool output back manually

Use [`tool_result`](@ref) to return a function's output on the next turn:

```julia
r2 = respond(Respond(; service=GEMINIServiceEndpoint,
                     previous_response_id=r1.response.id,
                     input=[tool_result("call_abc", "get_weather", "72F and sunny")]))
```

## Lifecycle (background requests)

[`get_response`](@ref) and [`cancel_response`](@ref) take a `service=` keyword,
so background Gemini Interactions are managed the same way as OpenAI:

```julia
status = get_response("<interaction_id>"; service=GEMINIServiceEndpoint)
cancel_response("<interaction_id>"; service=GEMINIServiceEndpoint)
```

Note: a freshly created background interaction id may briefly return a
`403 permission_denied` `ResponseFailure` before the resource becomes
retrievable. Poll with a short delay before treating that as a real
permission error. The automatic retry policy deliberately does not retry
403 (permission errors are normally permanent), so this is the caller's
poll loop to handle.

## Usage & cost (cross-provider)

[`token_usage`](@ref) and [`estimated_cost`](@ref) work for Gemini too — the
Interactions decoder normalizes usage into the shared shape, counting tool-use prompt
tokens as input, and [`DEFAULT_PRICING`](@ref) includes `gemini-4-argon` and
`gemini-3.8-flash` (hosted-tool per-call fees are not modeled). An interaction is priced
as the model that served it, `response.model`, so an alias or a routed id is billed at
its target's rates, and at the tier it reports in `service_tier` (`token_usage(r).service_tier`):
input and output at half the standard rates on Flex and at 1.8 times on Priority, a cache hit at
the price the model's table lists (see [Multi-Backend Support](@ref backend_guide)).

```@example agentic
r = respond("What is 2+2?"; service=GEMINIServiceEndpoint, model="gemini-3.8-flash")
if r isa ResponseSuccess
    println("usage: ", token_usage(r))
    println("est. cost: \$", round(estimated_cost(r); digits=6))
else
    println("Request failed — ", r)
end
```

## Streaming

Streamed Gemini Interactions assemble each step from the wire's incremental events
and rebuild the output exactly as the non-streamed decode does: a streamed
`respond(...; tools=…, stream=true)` finishes with status `requires_action` and its
[`function_calls`](@ref) populated, like the non-streamed form. Thought steps carry
their signature and their `summary` (from the streamed thought summaries); consecutive
`model_output` steps — the hops of a long answer, thoughts between them or not — make one
message, whose text is their text parts joined with no separator, as Google's SDK reads it,
while a tool's step (a function call or result, a hosted tool's call or result) or a
`user_input` step ends it, and the `model_output` steps after one make a message of their own
([`output_text`](@ref) joins the messages with a newline); and a function call whose
`arguments` are `null` reads as `"{}"`. Initial text in
`step.start` and subsequent deltas are both included, in step order. Thought summaries
stay separate from answer text ([`reasoning_summaries`](@ref) reads them), while
signatures remain available in the provider's raw steps.

An interaction body — streamed or not — without a non-empty `id` or `status` is not an
interaction: `respond`, `get_response` and `cancel_response` return a
`ResponseCallError` for it instead of an empty success. The faults an interaction records
in its `errors` array reach `response.error`, the first of them; `response.raw["errors"]`
holds them all.

## See Also

- [Responses API](@ref responses_guide) — OpenAI-specific `respond`/`Respond` details
- [Tool Calling](@ref tools_guide) — defining tools and the tool loop
- [Multi-Backend](@ref backend_guide) — provider setup and capabilities
