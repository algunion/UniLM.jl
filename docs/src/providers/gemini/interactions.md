# [The Interactions API](@id gemini_interactions_page)

```@setup gemini
using UniLM
```

`respond` with `service=GEMINIServiceEndpoint` sends a [`Respond`](@ref) to Google's Interactions API
and returns a `ResponseSuccess` whose `response` is the interaction, its steps read into the
`output` that [`output_text`](@ref), [`function_calls`](@ref) and [`reasoning_summaries`](@ref) read.
Google stores an interaction "By default", so a later request can continue it with
`previous_interaction_id`; "You can opt into stateless behavior by setting `store=false`"
([Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)). A `Respond`
field with no Interactions counterpart is refused before the request is sent
([Limits and refusals](@ref gemini_limits)).

| `Respond` field | Interactions | Rules |
| :--- | :--- | :--- |
| `input` | `input`: a string, or a list of steps | [Inputs](@ref gemini_interactions_input) |
| `instructions` | `system_instruction` | [Request options](@ref gemini_interactions_options) |
| `tools`, `tool_choice` | `tools`, `generation_config.tool_choice.allowed_tools` | [Tools](@ref gemini_interactions_tools), [Hosted tools](@ref gemini_hosted_tools) |
| `text` | `response_format` | [Structured output](@ref gemini_interactions_structured_output) |
| `reasoning` | `generation_config.thinking_level`, `thinking_summaries` | [Thinking](@ref gemini_interactions_thinking) |
| `max_output_tokens`, `temperature`, `top_p`, `seed` | `generation_config.max_output_tokens`, `temperature`, `top_p`, `seed` | [Request options](@ref gemini_interactions_options) |
| `previous_response_id`, `store`, `background`, `stream` | `previous_interaction_id`, `store`, `background`, `stream` | [Conversations](@ref gemini_interactions_conversations), [Background requests](@ref gemini_background), [Streaming](@ref gemini_interactions_streaming) |
| `service_tier`, `safety_identifier` | `service_tier`, `labels.safety_identifier` | [Request options](@ref gemini_interactions_options) |
| `model` | `model` | the [thinking table](@ref gemini_reasoning) |
| the other fields | refused | [Limits and refusals](@ref gemini_limits) |

## [First interaction](@id gemini_interactions_first)

`respond` takes the input and the request's fields as keywords, and `output_text` reads the answer;
with no `model` it takes the default model named at the top of the [start page](@ref gemini_start):

```@example gemini
result = respond("In one sentence: what does a compiler do?"; service=GEMINIServiceEndpoint)
output_text(result)
```

```@example gemini
r = result.response
r.status, r.model
```

```@example gemini
u = token_usage(result)
(; u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.service_tier,
   cost = round(estimated_cost(result); digits=6))
```

`response.model` is the model that served the interaction, and the one [`estimated_cost`](@ref)
prices; `token_usage` reads the interaction's `usage` in the shape every result shares:
`prompt_tokens` is its `total_input_tokens` and `total_tool_use_tokens`, `completion_tokens` its
`total_output_tokens` and `total_thought_tokens`, `reasoning_tokens` the thought tokens,
`cached_tokens` its `total_cached_tokens`, `prompt_audio_tokens` the audio among the input, and
`service_tier` the interaction's `service_tier`, the tier that billed it. They are priced as on
generateContent ([Cost and usage](@ref gemini_cost)). `respond` returns an interaction whose `status`
is `"failed"` as a `ResponseFailure`, and one that is `"completed"`, `"requires_action"`,
`"incomplete"`, `"in_progress"` or `"cancelled"` as a `ResponseSuccess`, whose `status` says which;
`get_response` and `cancel_response` return every interaction they read as a `ResponseSuccess`
([Background requests](@ref gemini_background)).

## [Conversations](@id gemini_interactions_conversations)

`previous_response_id` continues a stored interaction: it goes as `previous_interaction_id`, and
Google holds the conversation so far, so the request sends only the new input. In this mode, Google
says, "the server automatically manages the conversation state, including all thought blocks and
signatures" ([thinking](https://ai.google.dev/gemini-api/docs/thinking)):

```@example gemini
followup = respond("And an interpreter?"; service=GEMINIServiceEndpoint,
                   previous_response_id=result.response.id)
output_text(followup)
```

Google adds that continuing this way helps its cache: "Using `previous_interaction_id` (stateful) to
continue conversations allows the system to more easily utilize implicit caching for the conversation
history" ([Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview)). With
`store=false` Google stores no interaction to continue — it "prevents using `previous_interaction_id`
for subsequent turns" ([Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview))
—, so each request carries the whole conversation as its input ([Inputs](@ref gemini_interactions_input)),
and Google asks that its thought steps go back "exactly as they were received from the model"
([thinking](https://ai.google.dev/gemini-api/docs/thinking)). UniLM
sends a step in the Interactions shape as it is; sending a result's `output` back as the next input is
untested.

## [Inputs](@id gemini_interactions_input)

The input is a string, or a list of steps, which `respond` writes from the items it is given:

- an [`InputMessage`](@ref), or a `Dict` with a `role`, becomes a `user_input` step for a user message
  and a `model_output` step for an assistant one. A system or developer message is refused — send
  that guidance as `instructions` — and so is a message `phase`.
- a message's content is text, or a vector of parts: [`input_text`](@ref) parts and the media
  attachments [`ImageAttachment`](@ref), [`AudioAttachment`](@ref) and [`DocumentAttachment`](@ref),
  each an `image`, `audio` or `document` part holding its bytes in base64;
  [`input_file`](@ref)`(document)` and an [`input_image`](@ref) with a base64 `data:` URL go the same
  way. An image's `detail` is the part's `resolution`, by the rules of
  [Attachments](@ref gemini_media).
- a [`tool_result`](@ref) item becomes a `function_result` step ([Tools](@ref gemini_interactions_tools)).
- any other item, a step already in the Interactions shape among them, goes as it is.

A web URL, a file id and a cache breakpoint have no counterpart here, and are refused
([Limits and refusals](@ref gemini_limits)).

```julia
r = respond(Respond(service=GEMINIServiceEndpoint,
    input=[InputMessage(role="user", content=[input_text("What colour is the shape? One word."),
                                              ImageAttachment("red_circle.png")])]))
```

## [Thinking](@id gemini_interactions_thinking)

A [`Reasoning`](@ref) sets the interaction's thinking: its `effort` becomes
`generation_config.thinking_level`, checked against the first column of the
[thinking table](@ref gemini_reasoning) — on this API a Gemini 2.5 model takes its levels too — and
`summary="auto"` becomes `thinking_summaries`, which asks for summaries of the thoughts. A `Reasoning`
with another `summary`, a `context` or a `mode` is refused ([Limits and refusals](@ref gemini_limits)).
"By default, only the final output is returned. You can enable thought summaries with
`thinking_summaries`" ([thinking](https://ai.google.dev/gemini-api/docs/thinking)):

```@example gemini
result = respond("A bat and a ball cost 1.10 in total, and the bat costs 1.00 more than the ball. " *
                 "How much does the ball cost? Answer with the amount only.";
                 service=GEMINIServiceEndpoint, reasoning=Reasoning(effort="high", summary="auto"))
output_text(result)
```

```@example gemini
reasoning_summaries(result)
```

```@example gemini
token_usage(result).reasoning_tokens
```

Each thought is a `thought` step in `response.output`, with its `summary` and its `signature`,
which Google describes as "An encrypted representation of the model's internal reasoning state.
Always present, even when the model performs minimal reasoning"; a thought may hold "only a signature
with no summary", as after "Simple requests, where the model didn't reason enough to generate a
summary" ([thinking](https://ai.google.dev/gemini-api/docs/thinking)), and `reasoning_summaries` is
then empty. Continuing with `previous_response_id`
leaves the signatures with Google ([Conversations](@ref gemini_interactions_conversations)).
`max_output_tokens` bounds the thinking with the answer: "If the model hits this limit while
reasoning, it stops generating with status `"incomplete"`"
([thinking](https://ai.google.dev/gemini-api/docs/thinking)). The thinking bills as output
([Cost and usage](@ref gemini_cost)).

## [Tools](@id gemini_interactions_tools)

A function tool — a `FunctionTool` from [`function_tool`](@ref), or a [`CallableTool`](@ref) holding
one — goes out as `{"type": "function", "name", "description", "parameters"}`. A `FunctionTool` with
`async`, `allowed_callers`, `defer_loading` or `output_schema` set is refused, and so is any tool
that is neither a `FunctionTool` nor a `Dict`; a `Dict` goes as it is
([Hosted tools](@ref gemini_hosted_tools)). `tool_choice` becomes
`generation_config.tool_choice.allowed_tools`:

| `tool_choice` | `allowed_tools` | With a function tool marked `strict=true` |
| :--- | :--- | :--- |
| unset | none sent | `{"mode": "validated"}` |
| `"auto"` | `{"mode": "auto"}` | `{"mode": "validated"}` |
| `"required"` | `{"mode": "any"}` | the same |
| `tool_choice_function("get_weather")` | `{"mode": "any", "tools": ["get_weather"]}` | the same |
| `"none"` | `{"mode": "none"}` | the same |

Google's modes on this API: `auto` — "Model decides whether to call a function or respond
directly"; `any` — "Model is constrained to always predict a function call"; `none` — "Model is
prohibited from making function calls"; `validated` — "Model ensures function schema adherence"
([function calling](https://ai.google.dev/gemini-api/docs/function-calling)). That page states schema
adherence for `validated` and not for `any`. The other `tool_choice` builders select tools this API
does not have, and are refused ([Limits and refusals](@ref gemini_limits)).

A function call is a `function_call` item of `response.output`, its `call_id` made by Google and its
`arguments` a JSON string (`"{}"` for a call without arguments); [`function_calls`](@ref) reads
them, and an interaction that waits for their results has the status `"requires_action"`. A
[`tool_result`](@ref)`(call_id, name, output)` item answers one: it becomes a `function_result` step,
the output sent as its JSON object when it is one and as `{"result": output}` otherwise. The
[`tool_loop`](@ref) that takes an input or a `Respond` runs the calls and sends their results, each
turn continuing the last with `previous_response_id`, until the model answers in text:

```@example gemini
weather = CallableTool(
    function_tool("get_weather", "The current weather in a city";
        parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                        "required" => ["city"])),
    (name, args) -> "16 °C and cloudy in $(args["city"])")
loop = tool_loop("What is the weather in the largest city of Peru?";
                 service=GEMINIServiceEndpoint, tools=[weather])
output_text(loop.response), loop.turns_used, loop.completed
```

The loop takes the same `cancel` and `tool_concurrency` keywords as on generateContent, and stops
as [Tool Calling](@ref tools_guide) describes. To answer a call yourself, send the result in a
request that continues the interaction:

```julia
call = first(function_calls(r1))
r2 = respond(Respond(service=GEMINIServiceEndpoint, previous_response_id=r1.response.id,
                     input=[tool_result(call["call_id"], call["name"], "16 °C and cloudy")]))
```

## [Hosted tools](@id gemini_hosted_tools)

Of the tools Google runs for the model on this API, UniLM builds three:
[`gemini_google_search`](@ref) (`{"type": "google_search"}`), [`gemini_code_execution`](@ref)
(`{"type": "code_execution"}`) and [`gemini_url_context`](@ref) (`{"type": "url_context"}`). Pass
them in `tools`:

```@example gemini
result = respond("What is the latest stable release of the Julia language? Answer in one sentence.";
                 service=GEMINIServiceEndpoint, tools=[gemini_google_search()])
output_text(result)
```

```@example gemini
[get(item, "type", "") for item in result.response.output]
```

```@example gemini
answer = only(step for step in result.response.raw["steps"] if step["type"] == "model_output")
[(a["type"], a["title"]) for a in answer["content"][1]["annotations"]]
```

A hosted tool's work is in `response.output` as the steps Google sent — `google_search_call` and
`google_search_result`, `code_execution_call` and `code_execution_result`, `url_context_call` and
`url_context_result` — each a `Dict` with Google's fields, beside the `message` items that hold the
answer: read the queries, results and code there. Google puts a grounded answer's sources in "inline
`annotations`" on its text ([Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search)),
and the message UniLM builds from a `model_output` step keeps the text alone, so
[`url_citations`](@ref) finds none in an interaction: read them from the `model_output` step in
`response.raw["steps"]`, as the last block above does. Another tool the API takes — Google's error
reference names `mcp_server`, `file_search`, `google_maps` and `computer_use` among its tool types
([Interactions errors](https://ai.google.dev/gemini-api/docs/api-errors)) — goes as a `Dict` in the
API's own shape, which UniLM sends as it is. The input a tool read
counts as prompt tokens ([First interaction](@ref gemini_interactions_first)), but the fees of the
tools themselves are not priced: Google search grounding, for one, has "5,000 free search requests
per month (shared across all Gemini 3 and newer models), then \$14 per 1,000 requests"
([pricing](https://ai.google.dev/gemini-api/docs/pricing)).

## [Structured output](@id gemini_interactions_structured_output)

A `text` format becomes the top-level `response_format`: [`json_schema_format`](@ref) sends
`{"type": "text", "mime_type": "application/json", "schema": …}`, and [`json_object_format`](@ref) the
same without a schema. Only the schema is sent, not the format's name, description or `strict`;
`verbosity` is refused ([Limits and refusals](@ref gemini_limits)).

```@example gemini
using JSON
fmt = json_schema_format("capital", "A capital city and its country",
    Dict("type" => "object",
         "properties" => Dict("city" => Dict("type" => "string"), "country" => Dict("type" => "string")),
         "required" => ["city", "country"]))
result = respond("Give the capital of Peru."; service=GEMINIServiceEndpoint, text=fmt)
JSON.parse(output_text(result))
```

Google's page for this API lists the schema keywords it does not enforce: `minProperties` and
`maxProperties`, `minLength` and `maxLength`, and `uniqueItems`, `contains`, `minContains` and
`maxContains` "are not enforced during constrained decoding and are ignored by the server"
([structured output](https://ai.google.dev/gemini-api/docs/structured-output)): check them on the
parsed answer.

## [Streaming](@id gemini_interactions_streaming)

With `stream=true`, or the `do`-block form, `respond` returns a `Task`, and `fetch` gives the same
result as an unstreamed call. The callback receives the answer's text as it arrives, then the
assembled `ResponseObject`:

```@example gemini
deltas = String[]
task = respond("Name the eight planets of the solar system in order, each with one short sentence about it.";
               service=GEMINIServiceEndpoint) do chunk, close
    chunk isa String && push!(deltas, chunk)
end
result = fetch(task)
length(deltas), first(deltas)
```

Google streams each step as `step.start`, `step.delta` and `step.stop` events, then
`interaction.completed` with the interaction's status and usage
([streaming](https://ai.google.dev/gemini-api/docs/streaming)). UniLM rebuilds each step from its
events, in order — the text of a `model_output` step, the arguments of a `function_call`, the
signature and the summaries of a `thought` — and reads the result's `output` from them as an
unstreamed decode reads the interaction's steps: a streamed request with tools ends with the status
`"requires_action"` and its [`function_calls`](@ref), as an unstreamed one does. Only answer text
reaches the callback; the thought summaries are read afterwards with [`reasoning_summaries`](@ref).
Several `model_output` steps in a row — with thoughts between them or not — are one message, its text
their texts joined with no separator, as Google's SDK reads them: it continues "across `thought` steps
and empty `model_output` steps when aggregating text from the final model turn, while still stopping at
`user_input` and tool call/result boundaries"
([python-genai 180c3453](https://github.com/googleapis/python-genai/commit/180c3453f91e4734b828534950fa7f8526934b59)).
A tool's step or a `user_input` step ends the message, and [`output_text`](@ref) joins the messages
with a newline; this holds for an unstreamed interaction too.

## [Background requests](@id gemini_background)

With `background=true`, the API "immediately returns an interaction ID", and the interaction runs on
the server ([background execution](https://ai.google.dev/gemini-api/docs/background-execution)).
[`get_response`](@ref) and [`cancel_response`](@ref) take `service=GEMINIServiceEndpoint`, and read or
cancel it by its id; [`delete_response`](@ref) deletes it. Each makes one attempt
([Timeouts & Retries](@ref timeouts_guide)), and a poll is your loop:

```julia
started = respond("Write a long essay on the history of numerical computing.";
                  service=GEMINIServiceEndpoint, background=true)
id = started.response.id
status = get_response(id; service=GEMINIServiceEndpoint)
cancel_response(id; service=GEMINIServiceEndpoint)
```

A background request needs a stored interaction — "Passing `background=True` with `store=False`
returns a `400` error" ([background execution](https://ai.google.dev/gemini-api/docs/background-execution))
— so `respond` refuses the pair before sending it ([Limits and refusals](@ref gemini_limits)). Gemini 4 Argon writes
more than 65,536 output tokens only in the background: Google's limit with `background: false` is
"65,536 tokens. Values greater than 65,536 return a 400 error", and a background request takes "Your
`max_output_tokens`, up to 1,048,576 minus your input"
([long decode](https://ai.google.dev/gemini-api/docs/long-decode)), so UniLM refuses a larger
`max_output_tokens` on `gemini-4-argon` without `background=true`. A background interaction read
right after it was created has answered `403` (`permission_denied`) for a few seconds before it
became readable (observed on 20 August 2026; Google's pages do not mention it): retry such a read in
your poll before taking it for a permission error.

Google's Interactions API reads, cancels and deletes an interaction, and lists no input items:
[`list_input_items`](@ref) sends a request to a path the API reference does not list
([Interactions API](https://ai.google.dev/api/interactions-api)), and its answer is untested.
[`compact_response`](@ref) and [`count_input_tokens`](@ref) send nothing for this endpoint, and
return a `ResponseCallError`.

## [Request options](@id gemini_interactions_options)

- `instructions` becomes `system_instruction`.
- `max_output_tokens` becomes `generation_config.max_output_tokens`, the thinking included
  ([Thinking](@ref gemini_interactions_thinking)).
- `temperature`, `top_p` and `seed` follow the rules of a `Chat` ([Request options](@ref gemini_request_options)):
  sampling only for the earlier models, a seed in the int32 range.
- `safety_identifier` becomes the request label `safety_identifier`, under the same rule as on a `Chat`.
- `service_tier` takes `"standard"`, `"flex"`, `"priority"` and `"deferred"`. Google's Interactions
  API reference lists `flex`, `standard` and `priority` as its tiers
  ([Interactions API](https://ai.google.dev/api/interactions-api)), and not `deferred`, which UniLM
  sends as given: what the API answers to it is untested, and a reply that reports it is estimated at
  the standard rates, with a warning ([Cost and usage](@ref gemini_cost)).
- `store` and `background` are sent as they are ([Background requests](@ref gemini_background)).

A body that is not an interaction — a 200 without a non-empty `id` or `status`, which the API
reference marks "required" — is a `ResponseCallError` from `respond`, `get_response` and
`cancel_response`, never an empty success. The faults an interaction records in its `errors` array
reach `response.error`, the first of them, and `response.raw["errors"]` holds them all. How a call is
retried, and the daily quota that is not, is on [When something fails](@ref gemini_failures).

Recorded on 25 October 2026 on `gemini-3.8-flash` with UniLM's development branch after 0.26.0 and the Gemini
API at `v1beta`.
