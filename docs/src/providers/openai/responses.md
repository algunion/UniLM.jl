# [Responses & hosted tools](@id responses_guide__next)

```@setup openai
using UniLM
```

On `OPENAIServiceEndpoint` a [`Respond`](@ref) goes to OpenAI's Responses API, `POST /v1/responses`, as
its own JSON, which writes each field the `Respond` sets under the API's name. `respond(input; kwargs...)`
builds the `Respond` from its keywords and sends it, `respond(r)` sends one built before, and
`tool_loop` runs the function calls of a response until the model answers. Before anything is sent, the
request is checked against OpenAI's rules and the model's row in the [rule table](@ref openai_reasoning),
and a request that breaks one is refused ([Limits and refusals](@ref openai_limits)). The verb every
provider of an agentic API shares is on [Agentic Workflows](@ref agentic_guide).

| `Respond` field | Responses API | Rules |
| :--- | :--- | :--- |
| `input`, `instructions` | `input`: a string, or `InputMessage`s and other input items; `instructions` | [Input](@ref openai_responses_input) |
| `tools`, `tool_choice`, `parallel_tool_calls`, `max_tool_calls` | the same fields; `tool_choice` a string or a `tool_choice_…` object | [Tools](@ref openai_tools), [Hosted tools](@ref openai_hosted_tools) |
| `text` | `text`: the output format and `verbosity` | [Structured output on Responses](@ref openai_responses_structured_output) |
| `reasoning` | `reasoning`: `effort`, `summary`, `context`, `mode` | [Reasoning](@ref openai_reasoning) |
| `store`, `background`, `previous_response_id`, `conversation` | the same fields | [Stored responses](@ref openai_stored_responses), [Conversations](@ref openai_conversations) |
| `stream`, `stream_options` | the same fields | [Streaming](@ref openai_streaming) |
| `prompt_cache_options`, `prompt_cache_key`, `prompt_cache_retention` | the same fields | [Prompt caching](@ref openai_caching) |
| `temperature`, `top_p`, `top_logprobs`, `include`, `max_output_tokens`, `truncation`, `service_tier`, `metadata`, `safety_identifier`, `moderation`, `context_management` | the same fields | [Request options](@ref openai_responses_options) |
| `user`, `seed` | refused | [Limits and refusals](@ref openai_limits) |

## [Input](@id openai_responses_input)

`input` is a string, the user's message, or a vector of input items: [`InputMessage`](@ref)s, with the
role `"user"`, `"assistant"`, `"system"` or `"developer"`, and items given as `Dict`s, such as a
[`tool_result`](@ref), a [`configuration_update`](@ref) or an [`mcp_approval_response`](@ref).
`instructions` is "A system (or developer) message inserted into the model's context", and with
`previous_response_id`, "the instructions from a previous response will not be carried over"
([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create)).

An `InputMessage`'s content is a string or a vector of parts: [`input_text`](@ref), [`input_image`](@ref)
— an image URL, a base64 data URL, or the `file_id` of an uploaded file, with a `detail` level — and
[`input_file`](@ref) — a file URL, a file id, or base64 `file_data`; `input_file(document)` makes the part
of a [`DocumentAttachment`](@ref). An attachment itself is refused in a Responses input, an
`AudioAttachment` among them ([Limits and refusals](@ref openai_limits)).

An assistant message you replay, rather than chain with `previous_response_id`, keeps the `phase` the
model gave it, `"commentary"` for an intermediate update or `"final_answer"`: OpenAI asks callers, for
"models like `gpt-5.3-codex` and beyond", to "preserve and resend phase on all assistant messages —
dropping it can degrade performance"
([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create),
`input`). [`output_text`](@ref) leaves a `"commentary"` message out of the answer.
[`UniLM.encode_agentic`](@ref) shows the body `respond` sends, with no key and no request:

```@example openai
r = Respond(service=OPENAIServiceEndpoint, instructions="You are a concise assistant.",
    input=[InputMessage(role="user", content=[input_text("What is in this picture?"),
                                              input_image("https://example.com/photo.jpg"; detail="low")]),
           InputMessage(role="assistant", content="I'll describe the picture first.", phase="commentary"),
           InputMessage(role="user", content="Go on.")])
print(UniLM.encode_agentic(r.service, r))
```

## [Tools](@id openai_tools)

A function tool is a [`FunctionTool`](@ref), made by
`function_tool(name, description; parameters, strict)`; it also takes `async`, `allowed_callers`,
`defer_loading` and `output_schema`, each sent when set. A Chat `Tool` given to `Respond` becomes the equivalent
`FunctionTool`. OpenAI's guide to its GPT-6 models routes tool calling to this API: "Use Responses for
reasoning with tools"
([GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra#update-api-and-model-parameters)),
and UniLM checks no effort for tools here.

[`function_calls`](@ref) reads a response's calls, each a `Dict` with its `call_id`, `name` and
`arguments` (a JSON string), and [`tool_result`](@ref)`(call_id, name, output)` makes the input item that
answers one on the next request. [`tool_loop`](@ref) does both until the model answers in text, chaining
the requests with `previous_response_id`, or with `conversation` when the `Respond` sets one:

```@example openai
weather = CallableTool(
    function_tool("get_weather", "The current weather in a city";
        parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                        "required" => ["city"], "additionalProperties" => false), strict=true),
    (name, args) -> "18 °C and sunny in $(args["city"])")
loop = tool_loop("What is the weather in Oslo and in Lima?"; tools=[weather], service=OPENAIServiceEndpoint,
                 instructions="You are a concise assistant.")
output_text(loop.response)
```

```@example openai
[(oc.tool_name, oc.arguments) for oc in loop.tool_calls], loop.turns_used
```

The `Respond` sets no `parallel_tool_calls`, and the model asked for both cities in one response; a `Chat`
with tools sends `false` and gets a turn per call ([Tools on Chat Completions](@ref openai_chat_tools)).

`tool_choice` is `"auto"`, `"none"` or `"required"`, or an object a builder makes:
[`tool_choice_function`](@ref)`(name)`, [`tool_choice_hosted`](@ref)`(type)`,
[`tool_choice_mcp`](@ref)`(label; name)`, [`tool_choice_custom`](@ref)`(name)` or
[`tool_choice_allowed`](@ref)`(mode, tools)`. `parallel_tool_calls`, "Whether to allow the model to run
tool calls in parallel", is sent only when set, and left to the API's default otherwise. The loop runs a
response's calls only when its status is `completed` or `requires_action`, and stops, with `completed=false`, on a
call the application must carry out itself (`computer_call`, `apply_patch_call`, `custom_tool_call`, a
`shell_call` that OpenAI did not run, or an `mcp_approval_request`); the loop every provider shares is on
[Tool Calling](@ref tools_guide).

## [Hosted tools](@id openai_hosted_tools)

The other tools of a `Respond` are OpenAI's built-in tools: most run on OpenAI's side, and their results
come back in the response's output items; the last row's are calls the application carries out:

| Tool | Made by | Its results |
| :--- | :--- | :--- |
| web search | [`web_search`](@ref)`(; context_size, location, filters)` | the answer's [`url_citations`](@ref); [`web_search_results`](@ref) |
| file search | [`file_search`](@ref)`(vector_store_ids; max_results, ranking, filters)` | [`file_search_results`](@ref); the stores on [Retrieval & file search](@ref retrieval_guide__next) |
| code interpreter | [`code_interpreter`](@ref)`()` | [`code_interpreter_outputs`](@ref); its settings on [Platform APIs](@ref openai_platform_page) |
| image generation | [`image_generation_tool`](@ref)`(; size, quality, action, …)` | the images as base64, [`image_generation_results`](@ref) |
| remote MCP servers | [`mcp_tool`](@ref)`(label, url; require_approval, allowed_tools, …)` | [`mcp_call_outputs`](@ref); [`mcp_approval_requests`](@ref) |
| computer use, shell, patches, custom tools | [`computer_tool`](@ref), [`shell`](@ref), [`apply_patch_tool`](@ref), [`custom_tool`](@ref) | calls the application carries out; a shell call only when OpenAI did not run it |

```@example openai
result = respond("What is the latest stable release of the Julia programming language? " *
                 "Answer in one plain-text sentence."; service=OPENAIServiceEndpoint, tools=[web_search()])
output_text(result)
```

```@example openai
[c["url"] for c in url_citations(result)]
```

`mcp_tool` sends `require_approval="never"` unless it is given another value, while OpenAI's own
default is to "request your approval before any data is shared with a connector or remote MCP server";
with approvals, a request for one is an `mcp_approval_request` item, and an
[`mcp_approval_response`](@ref) in the next request's input answers it
([MCP servers](https://developers.openai.com/api/docs/guides/tools-connectors-mcp#approvals)). The
hosted tools bill beyond the tokens — web search per call, containers per session — which
[`estimated_cost`](@ref) does not include ([Cost and usage](@ref openai_cost)).

## [Structured output on Responses](@id openai_responses_structured_output)

`text` holds the output format: [`json_schema_format`](@ref)`(name, description, schema; strict)` for a
reply that follows a JSON Schema, [`json_object_format`](@ref)`()` for any JSON object, and
[`text_format`](@ref)`(; verbosity)` for plain text, each a [`TextConfig`](@ref) whose `verbosity` is
`"low"`, `"medium"` or `"high"`. A strict schema follows the rules of
[Structured output](@ref openai_structured_output).

```@example openai
using JSON
fmt = json_schema_format("planets", "The planets of the solar system in order",
    Dict("type" => "object",
         "properties" => Dict("planets" => Dict("type" => "array", "items" => Dict("type" => "string"))),
         "required" => ["planets"], "additionalProperties" => false); strict=true)
result = respond("List the planets of the solar system in order."; service=OPENAIServiceEndpoint, text=fmt)
JSON.parse(output_text(result))["planets"]
```

## [Stored responses](@id openai_stored_responses)

A response is stored unless its request sets `store=false`, "for at least 30 days"
([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create)),
and `get_response`, `cancel_response`, `delete_response` and `list_input_items` work on it by its id.
[`get_response`](@ref) and [`cancel_response`](@ref) return a `ResponseSuccess`; [`delete_response`](@ref),
[`list_input_items`](@ref), [`compact_response`](@ref) and [`count_input_tokens`](@ref) return OpenAI's
answer as a `Dict`; a failure of any of them is a `ResponseFailure` or a `ResponseCallError`. Each makes
one attempt, never retried ([When something fails](@ref openai_failures)).

```@example openai
r = respond("Say 'stored response test' and nothing else.", store=true, model="gpt-5.4-mini")
if r isa ResponseSuccess
    rid = r.response.id
    println("Stored response ID: ", rid)

    # Retrieve
    retrieved = get_response(rid)
    if retrieved isa ResponseSuccess
        println("Retrieved text: ", output_text(retrieved))
    end

    # List input items
    items = list_input_items(rid)
    if items isa Dict
        println("Input items: ", length(items["data"]))
    end

    # Delete
    del = delete_response(rid)
    if del isa Dict
        println("Deleted: ", del["deleted"])
    end
else
    println("Request failed — ", r)
end
```

[`count_input_tokens`](@ref) asks how many input tokens a request would use, "the exact count the model
will receive", without generating
([counting tokens](https://developers.openai.com/api/docs/guides/token-counting)); with no `model` it
counts for the default model named at the top of the [start page](@ref openai_start):

```@example openai
result = count_input_tokens(input="Tell me a joke about programming")
if result isa Dict
    println("Input tokens: ", result["input_tokens"])
else
    println("Request failed — see result for details")
end
```

```@example openai
tool = function_tool("search", "Search for information",
    parameters=Dict("type" => "object", "properties" => Dict(
        "query" => Dict("type" => "string")
    ), "required" => ["query"], "additionalProperties" => false),
    strict=true
)
result = count_input_tokens(
    input="Search for Julia language news",
    instructions="You are a helpful assistant.",
    tools=[tool]
)
if result isa Dict
    println("Tokens with tools: ", result["input_tokens"])
else
    println("Request failed — see result for details")
end
```

[`compact_response`](@ref) runs a compaction pass over a conversation's items,
`POST /v1/responses/compact`: the window it returns "includes an encrypted compaction item" that is "opaque",
to be passed on as the next request's input without pruning
([compaction](https://developers.openai.com/api/docs/guides/compaction)):

```@example openai
items = [
    Dict("role" => "user", "content" => "Hello, I want to learn about Julia."),
    Dict("type" => "message", "role" => "assistant", "status" => "completed",
         "content" => [Dict("type" => "output_text",
            "text" => "Julia is a high-performance programming language for technical computing.")])
]
result = compact_response(input=items, model="gpt-5.4-mini")
if result isa Dict
    println("Compact succeeded")
    println("Output items: ", length(result["output"]))
    println("Usage: ", result["usage"])
else
    println("Request failed — see result for details")
end
```

With `background=true`, `respond` returns as soon as OpenAI has queued the response, and
`cancel_response` stops it:

```@example openai
started = respond("Write a long essay on the history of numerical computing."; service=OPENAIServiceEndpoint,
                  background=true)
response_status(started)
```

```@example openai
response_status(cancel_response(started.response.id))
```

A background response is read back with `get_response` until it leaves the queued and in-progress
states, "Keep polling while the request is in the queued or in_progress state"
([background mode](https://developers.openai.com/api/docs/guides/background)):

```julia
r = get_response(started.response.id)
while response_status(r) in ("queued", "in_progress")
    sleep(2)
    r = get_response(started.response.id)
end
```

## [Request options](@id openai_responses_options)

- **`service_tier`.** `"auto"`, `"default"`, `"flex"`, `"scale"`, `"fast"` or its former name
  `"priority"`, and `"ultrafast"` on the models whose Ultrafast tier cell in the
  [rule table](@ref openai_reasoning) is "yes": "Ultrafast mode is the fastest service tier in the OpenAI
  API" ([Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode)). The tier OpenAI
  reports is `token_usage(result).service_tier`, which [Cost and usage](@ref openai_cost) prices.
- **`max_output_tokens`, `truncation`.** `max_output_tokens` bounds the visible output and the
  reasoning together. With `truncation="auto"`, an input over the context window loses items from the
  beginning of the conversation; with `"disabled"`, the API's default, the request "will fail with a 400
  error".
- **`include`.** Extra output data, such as `"web_search_call.action.sources"`, `"file_search_call.results"`
  or `"message.output_text.logprobs"`, the last at the efforts the model's row takes log probabilities.
- **`max_tool_calls`.** "The maximum number of total calls to built-in tools that can be processed in a
  response".
- **`metadata`, `safety_identifier`, `prompt_cache_key`.** Sent as given; `metadata` holds up to 16
  key-value pairs, and `user`, which OpenAI replaces with the other two, is refused
  ([Limits and refusals](@ref openai_limits)).
- **`moderation`.** A [`ModerationConfig`](@ref) runs a moderation model over the input and the output;
  the response's `moderation` holds the results, in `result.response.raw["moderation"]`. Text and images
  on their own are classified by [`moderate`](@ref) ([Moderation](@ref openai_moderation)).
- **`context_management`, `stream_options`.** Sent as given: compaction settings, and the stream's
  `include_obfuscation`.

The quoted fields are from OpenAI's
[Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [Retrieve a model response](https://developers.openai.com/api/reference/resources/responses/methods/retrieve).
[`refusals`](@ref) and [`reasoning_items`](@ref) read a response's output items as the hosted tools'
accessors do, and [`incomplete_details`](@ref) and [`usage_details`](@ref) its `incomplete_details` and
`usage`. [`UniLM.encode_agentic`](@ref) shows the body:

```@example openai
r = Respond(service=OPENAIServiceEndpoint, input="Count to five.", service_tier="flex", max_output_tokens=200,
            truncation="auto", metadata=Dict("ticket" => "T-1042"), safety_identifier="user-42",
            moderation=ModerationConfig(model="omni-moderation-latest", input_mode="block"))
print(UniLM.encode_agentic(r.service, r))
```

Recorded on 23 and 25 October 2026 on `gpt-6-sol` and `gpt-5.4-mini` with UniLM's development branch
after 0.26.0.
