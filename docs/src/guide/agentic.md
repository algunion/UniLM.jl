# [Agentic respond](@id agentic_guide)

```@setup agentic
using UniLM
```

[`respond`](@ref) sends one request to a provider's agentic API — the input, `instructions`, tools and
options of a [`Respond`](@ref) — and returns the response as output items: the answer's messages,
function calls, reasoning, the work of the tools the provider runs. A provider that stores responses
lets the next request name the last one instead of sending the conversation again. `respond` runs on an
endpoint whose capabilities hold `:responses` or `:agentic`
([Capabilities by provider](@ref capabilities_guide)), and throws `ArgumentError` before anything is
sent on one whose declared capabilities hold neither; a `Chat` serves every language-model provider
([Conversations with Chat](@ref chat_guide)).

```@example agentic
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref agentic_providers) links each provider's rules.

## [A first response](@id agentic_first)

`respond(input; kwargs...)` builds the `Respond` from its keywords and sends it, and `respond(r)` sends
one built before:

```@example agentic
result = respond("Explain multiple dispatch in one plain-text sentence."; service, model)
if result isa ResponseSuccess
    println(output_text(result))
else
    println("Request failed — ", result)
end
```

The result is a [`ResponseSuccess`](@ref) whose `response` is a [`ResponseObject`](@ref) — its `id`,
`status`, `model`, `output` items, `usage` and the whole body as `raw` —, a `ResponseFailure`, or a
`ResponseCallError` ([Results](@ref results_guide)). [`output_text`](@ref) joins the text of the
answer's messages, [`token_usage`](@ref) reads the usage in the shape every result shares, and
[`estimated_cost`](@ref) prices it:

```@example agentic
u = token_usage(result)
(; result.response.status, u.prompt_tokens, u.completion_tokens, cost = round(estimated_cost(result); digits=6))
```

`respond` with a `do` block streams the answer as it arrives ([Streaming](@ref streaming_respond)).

## [Input and instructions](@id agentic_input)

`input` is a string, the user's message, or a vector of input items: [`InputMessage`](@ref)s, whose
content is a string or a vector of parts — [`input_text`](@ref), [`input_image`](@ref),
[`input_file`](@ref) —, and items such as a [`tool_result`](@ref). `instructions` is the system
guidance. `reasoning=Reasoning(effort=…, summary="auto")` sets how much a reasoning model thinks and asks
for summaries of its reasoning, where the provider gives them, which [`reasoning_summaries`](@ref) reads.
Which roles, parts and media a provider takes, the efforts and summaries its models take, and the
`Respond` fields it refuses before the request, are on its entry. [`UniLM.encode_agentic`](@ref) returns
the body `respond` sends, with no key and no request:

```@example agentic
r = Respond(service=service, model=model, instructions="You are a concise assistant.",
            input=[InputMessage(role="user", content=[input_text("Which planet is the largest?")])])
print(UniLM.encode_agentic(service, r))
```

## [Continuing a conversation](@id agentic_conversations)

`previous_response_id` names a stored response, and the request sends only the new input, since the
provider holds the conversation so far. Whether a provider stores responses, and what UniLM does with
`previous_response_id` where it does not, is on its entry; without stored responses, the whole
conversation goes in `input`.

```julia
r1 = respond("Which planet of the solar system is the largest?"; service, model)
r2 = respond("And the smallest?"; service, model, previous_response_id=r1.response.id)
```

## [Tools](@id agentic_tools)

A function tool is a [`FunctionTool`](@ref), made by [`function_tool`](@ref), and a
[`CallableTool`](@ref) pairs one with the Julia function that answers it. [`function_calls`](@ref)
reads a response's calls, each a `Dict` with its `call_id`, `name` and `arguments` (a JSON string), and
[`tool_result`](@ref)`(call_id, name, output)` makes the input item that answers one on the next
request. [`tool_loop`](@ref) does both until the model answers in text; [Tool Calling](@ref tools_guide)
has its rules:

```julia
ct = CallableTool(function_tool("get_weather", "Get weather",
        parameters=Dict("type" => "object",
                        "properties" => Dict("location" => Dict("type" => "string")))),
    (name, args) -> "22C, sunny")
result = tool_loop("What's the weather in Paris?"; service, model, tools=[ct])
# result.completed == true when the model returns a final text answer
```

To answer a call yourself, send its result in a request that continues the response:

```julia
call = first(function_calls(r1))
r2 = respond(Respond(; service, model, previous_response_id=r1.response.id,
                     input=[tool_result(call["call_id"], call["name"], "72F and sunny")]))
```

`tool_choice` is `"auto"`, `"none"` or `"required"`, or an object a builder makes:
[`tool_choice_function`](@ref)`(name)` forces one function; [`tool_choice_hosted`](@ref),
[`tool_choice_mcp`](@ref) and [`tool_choice_custom`](@ref) select other kinds of tool, and
[`tool_choice_allowed`](@ref) limits the model to some of its tools. A provider may also run tools of its
own, such as web search or code execution, passed in `tools`. Which choices and tools a provider takes is
on its entry.

## [Background requests](@id agentic_background)

With `background=true`, `respond` returns at once with the response's `id` and `status`, and the
response runs on the provider's side; [`get_response`](@ref) reads it by its id and
[`cancel_response`](@ref) stops it. Both take a `service` keyword and make one attempt, never retried
([Timeouts & Retries](@ref timeouts_guide)), so a poll is your own loop. Whether a provider runs
background requests, and what it answers while one is queued, is on its entry.

```julia
started = respond("Write a long essay on the history of numerical computing."; service, model,
                  background=true)
status = response_status(get_response(started.response.id; service))
cancel_response(started.response.id; service)
```

## [See also](@id agentic_see_also)

- [Tool Calling](@ref tools_guide) — defining tools and the tool loops
- [Streaming](@ref streaming_guide) — `respond` with a `do` block
- [Results](@ref results_guide) — success, failure and call error
- [One API across providers](@ref backend_guide) — switching providers
- [Decisions with LLMs](@ref jev_llm_guide) — pick the model a request needs, or approve a tool call before it runs

## [On your provider](@id agentic_providers)

```@eval
Main.ProviderFacts.provider_table(Main.ProviderFacts.llm)
```
