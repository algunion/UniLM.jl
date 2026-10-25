# [Platform APIs](@id openai_platform_page)

```@setup openai
using UniLM
```

Besides its model APIs, OpenAI serves APIs that hold what a request works with — files,
conversations, code-interpreter containers — and that run work beside it: batches, fine-tuning
jobs, moderation, and the webhooks that report on them. UniLM wraps each with verbs of its own, on
`OPENAIServiceEndpoint` unless a call names another `service`; `OPENAIServiceEndpoint()` returns
that same type. The vector stores of file search
are on [Retrieval & file search](@ref retrieval_guide__next), and images, speech and the Realtime
API on [Images, audio & realtime](@ref openai_media_page); which of these APIs other providers serve
is on [Capabilities by provider](@ref capabilities_guide). Every verb, type and keyword is in the
API reference: [Files](@ref files_api), [Uploads](@ref uploads_api),
[Conversations](@ref conversations_api), [Containers](@ref containers_api), [Batch](@ref batch_api),
[Fine-tuning](@ref fine_tuning_api), [Moderations](@ref moderations_api) and
[Webhooks](@ref webhooks_api). UniLM does not wrap OpenAI's Agents API, which "gives your
application access to the Codex harness through an OpenAI-managed API"
([Agents API](https://developers.openai.com/api/docs/guides/agents-api/overview)).

These calls are not recorded, so their output on this page is quoted from dated runs.

## [What the platform verbs share](@id openai_platform_shared)

- **Results.** A verb returns a typed result, and throws only for what it refuses before sending
  ([Limits and refusals](@ref openai_platform_limits)): a `*Success`; a `*Failure` for an error
  status, holding the `status`, the body as `response` and the `x-request-id` header as
  `request_id` (a `RealtimeFailure` carries none); or a `*CallError` when no usable reply came — a
  transport failure, a timeout, a cancel, a reply that does not decode —, holding the text as
  `error` and the exception as `cause`. `FileSuccess`, `FileFailure` and `FileCallError` are the Files family, and the others
  are named alike ([Results](@ref results_guide)).
- **One attempt.** Each call makes a single attempt, bounded by the call's `config`, a
  [`RequestConfig`](@ref); `max_attempts` does not change it. A create is never sent again: a POST
  that timed out, or drew a gateway 5xx after OpenAI stored its object, may still have made it, so
  a failed `upload_file` does not prove that no file was made.
- **The key** is read when the call is made: without `OPENAI_API_KEY` the result is the family's
  call error, whose `cause` is the `KeyError` naming it ([Setup](@ref openai_setup)).
- **Cancellation.** With the key set, the ambient [`CancelToken`](@ref) of [`with_cancel`](@ref) ends
  a call: a token already cancelled sends nothing, and a cancel during the call aborts it, either way
  with the call error, whose `cause` is a [`UniLMCancelled`](@ref)
  ([Cancellation](@ref concurrency_cancellation)).
- **Cost.** The results carry no usage: [`token_usage`](@ref) and [`estimated_cost`](@ref) throw for
  them. What OpenAI charges for storage, containers and batches is on its
  [pricing page](https://developers.openai.com/api/docs/pricing).
- **Deletes.** A delete that OpenAI answers with a deletion record — of a file, a vector store, a
  container or a conversation — counts only when the record says `"deleted": true`: a reply without
  it, or with `false`, is the call error, the object being in place as far as UniLM can tell.
  `delete_conversation_item` is answered with the conversation, which UniLM returns as it came.
- **Capabilities.** Each verb first checks that its `service` has the capability it needs —
  `:files`, `:uploads`, `:vector_stores`, `:conversations`, `:containers`, `:batch`,
  `:fine_tuning`, `:moderation`, `:transcription` (which `:audio` implies) or `:realtime` —, all
  of which `OPENAIServiceEndpoint` declares (`realtime_connect`, which throws rather than returning
  a result and does not observe the token, is on [Realtime sessions](@ref timeout_realtime__next)),
  and refuses an endpoint that declares its capabilities without it
  ([Limits and refusals](@ref openai_platform_limits)); an endpoint that declares none is not
  dispatched to, a `MethodError` ([Capabilities by provider](@ref capabilities_guide)).

## [Files and uploads](@id openai_files)

[`upload_file`](@ref)`(path, purpose)` sends a file to `POST /v1/files` as
`multipart/form-data`, with the purpose `"assistants"`, `"batch"`, `"fine-tune"`, `"vision"` or
`"user_data"`; the Evals API's `"evals"` is refused ([Limits and refusals](@ref openai_platform_limits)),
since OpenAI's Evals "become read-only" on
31 October 2026 and "are scheduled to shut down" on 30 November 2026
([deprecations](https://developers.openai.com/api/docs/deprecations#2026-06-03-evals-platform)).
"Individual files can be up to 512 MB, and each project can store up to 2.5 TB of files in total";
the Batch API "only supports `.jsonl` files up to 200 MB in size", and fine-tuning `.jsonl` files
([Upload file](https://developers.openai.com/api/reference/resources/files/methods/create)).
[`list_files`](@ref) takes a `purpose`, a `limit` — "between 1 and 10,000, and the default is
10,000" — and `after`, the id of the last file of the page before (its `has_more` says whether one
follows); [`retrieve_file`](@ref) reads one file's record, [`file_content`](@ref) its bytes, and
[`delete_file`](@ref) removes it, which also is to "remove it from all vector stores"
([Files](https://developers.openai.com/api/reference/resources/files)).

```julia
write("questions.jsonl", """{"custom_id": "q1", "method": "POST", "url": "/v1/responses", "body": {"model": "gpt-6-sol", "input": "Name a prime number."}}\n""")
up = upload_file("questions.jsonl", "batch")
up isa FileSuccess || error("upload failed: $up")
f = up.response
f.filename, f.purpose, f.bytes
# Run on 25 October 2026 with UniLM's development branch after 0.26.0:
# ("questions.jsonl", "batch", 127)

listed = list_files(purpose="batch", limit=5)
listed isa FileListSuccess || error("listing failed: $listed")
any(x -> x.id == f.id, listed.response.data)
# true

content = file_content(f.id)
content isa FileContentSuccess || error("the download failed: $content")
save_file_content(content, "questions_copy.jsonl")
# "questions_copy.jsonl"

delete_file(f.id)
# FileDeleteSuccess("file-UYtK7VGBUPApXVa8Uz7epn", true)
```

!!! details "Evidence"
    Runs of 25 October 2026, 16:29–16:47 UTC, with UniLM's development branch after 0.26.0: OpenAI
    answered `file_content` of a file uploaded with the purpose `"user_data"` with HTTP 502 or 503 on
    each of three tries, while files with the purpose `"batch"` downloaded; four of ten `delete_file`
    calls drew HTTP 503 and succeeded when sent again; one `upload_file` drew HTTP 502, and a listing
    afterwards showed that no file had been made.

[`save_file_content`](@ref) writes the bytes atomically: to a temporary file beside the target,
then renamed over it, so a failed write leaves an existing file whole; the file keeps its
permission bits, a symbolic link is written through, and a directory is refused. An uploaded file
goes to a `Respond` as input by its id, `input_file(; id=…)` for a document and
`input_image(; file_id=…)` for an image ([Input](@ref openai_responses_input)).

The Uploads API takes a large file in parts: "an Upload can accept at most 8 GB in total and
expires after an hour after you create it", "Each Part can be at most 64 MB", and "The number of
bytes uploaded upon completion must match the number of bytes initially specified when creating
the Upload object" ([Uploads](https://developers.openai.com/api/reference/resources/uploads)).
[`create_upload`](@ref) takes the Files purposes, `"evals"` refused as above
([Limits and refusals](@ref openai_platform_limits));
[`add_upload_part`](@ref) sends one part, the bytes you give it; [`complete_upload`](@ref) joins the
parts in the order the list gives their ids, checking an `md5` of the whole file when given one, and
returns the upload with its `file`; [`cancel_upload`](@ref) ends one. UniLM does not split a file: a file
larger than a part is sent a slice at a time, as the second block below does.

```julia
data = read("requests.jsonl")
up = create_upload(filename="requests.jsonl", purpose="batch", bytes=length(data), mime_type="text/jsonl")
up isa UploadSuccess || error("create_upload failed: $up")
up.response.status, up.response.bytes
# Run on 25 October 2026 with UniLM's development branch after 0.26.0:
# ("pending", 456)

half = length(data) ÷ 2
p1 = add_upload_part(up.response.id, data[1:half])
p2 = add_upload_part(up.response.id, data[half+1:end])
(p1 isa UploadPartSuccess && p2 isa UploadPartSuccess) || error("a part failed: $p1, $p2")
done = complete_upload(up.response.id, [p1.response.id, p2.response.id])
done isa UploadSuccess || error("complete_upload failed: $done")
done.response.status, done.response.file.filename, done.response.file.bytes, done.response.file.purpose
# ("completed", "requests.jsonl", 456, "batch")
```

A file larger than a part goes up a slice at a time, each slice a part:

```julia
parts = String[]
for slice in Iterators.partition(data, 50_000_000)
    part = add_upload_part(up.response.id, collect(slice))
    part isa UploadPartSuccess || error("a part failed: $part")
    push!(parts, part.response.id)
end
done = complete_upload(up.response.id, parts)
```

## [Conversations](@id openai_conversations_api)

A conversation of OpenAI's Conversations API holds a conversation's items on OpenAI's side, with
"its own durable identifier", and works with the Responses API
([conversation state](https://developers.openai.com/api/docs/guides/conversation-state#using-the-conversations-api)).
[`create_conversation`](@ref)`(; items, metadata)` makes one — "You may add up to 20 items at a
time" — and [`conversation_id`](@ref) reads its id, which a `Respond`'s `conversation` takes: "Items
from this conversation are prepended to `input_items` for this response request. Input items and
output items from this response are automatically added to this conversation after this response
completes" ([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create),
`conversation`).

```julia
c = create_conversation(metadata=Dict("topic" => "ferries"))
c isa ConversationSuccess || error("create failed: $c")
conv = conversation_id(c.response)
r1 = respond("Which planet of the solar system is the largest?"; conversation=conv,
             instructions="You are a concise assistant: answer in one short sentence.")
r1 isa ResponseSuccess || error("respond failed: $r1")
output_text(r1)
# Run on 25 October 2026 with UniLM's development branch after 0.26.0 on gpt-6-sol:
# "Jupiter is the largest planet in the solar system."

r2 = respond("And the smallest?"; conversation=conv,
             instructions="You are a concise assistant: answer in one short sentence.")
r2 isa ResponseSuccess || error("respond failed: $r2")
output_text(r2), r2.response.raw["conversation"]["id"] == conv, token_usage(r2).prompt_tokens
# ("Mercury is the smallest planet in the solar system.", true, 57)

items = list_conversation_items(conv; order="asc")
items isa ConversationItemListSuccess || error("listing failed: $items")
[(item.type, item.raw["role"]) for item in items.response.data]
# 4-element Vector{Tuple{String, String}}:
#  ("message", "user")
#  ("message", "assistant")
#  ("message", "user")
#  ("message", "assistant")

delete_conversation(conv)
# ConversationDeleteSuccess("conv_6ade313d1634819582bab2425f6f512a0c45c1ed494760a4", true)
```

`conversation` is sent as given, the id or a `Dict("id" => …)`, and a `Respond` refuses it together
with `previous_response_id` ([Limits and refusals](@ref openai_limits)); the conversation's turns
come back from OpenAI, so the second request sends only its question, and [`tool_loop`](@ref) chains
through the conversation when the `Respond` sets one ([Tools](@ref openai_tools)). "Conversation
objects and items in them are not subject to the 30 day TTL" of a stored response
([conversation state](https://developers.openai.com/api/docs/guides/conversation-state)).
[`retrieve_conversation`](@ref) and [`update_conversation`](@ref)`(id, metadata)` read and change the
conversation's record, [`add_conversation_items`](@ref) appends items without a response,
[`list_conversation_items`](@ref) pages through them (`limit`, `order`, `after`), and
[`delete_conversation_item`](@ref) removes one, returning the conversation. No verb reads a single
item: `ConversationItemSuccess` is a result type no verb returns.

## [Code interpreter and containers](@id openai_containers)

The code interpreter, a hosted tool of the Responses API, runs Python that the model writes, and
"requires a container object", "a fully sandboxed virtual machine that the model can run Python
code in" ([Code Interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter#containers)).
[`code_interpreter`](@ref)`()` sends an auto container, `{"type": "auto"}`, which "automatically
creates a new container, or reuses an active container that was used by a previous
`code_interpreter_call` item in the model's context"; `container` takes OpenAI's other forms:

- **a `Dict`**, sent as given: an auto container with a `memory_limit`, `"1g"` (the default),
  `"4g"`, `"16g"` or `"64g"`, which "applies for the entire life of that container";
- **a `String`**, the id of a container made with [`create_container`](@ref), sent as
  `"container": "cntr_…"`.

`file_ids`, the ids of uploaded files, go inside the container `Dict`, the auto one unless you give
another. With a container id, or beside a `Dict` that names `file_ids` itself, they are refused
([Limits and refusals](@ref openai_platform_limits)): a container made beforehand takes its files
through `create_container(; file_ids)` and [`add_container_file`](@ref).

[`UniLM.encode_agentic`](@ref) shows the tool each form sends, with no key and no request:

```@example openai
sent(tool) = UniLM.encode_agentic(OPENAIServiceEndpoint, Respond(input="Total the prices.", tools=[tool]))
println(sent(code_interpreter()))
println(sent(code_interpreter(container=Dict("type" => "auto", "memory_limit" => "4g"),
                              file_ids=["file-abc123"])))
println(sent(code_interpreter(container="cntr_abc123")))
```

```@example openai
refusal = try code_interpreter(container="cntr_abc123", file_ids=["file-abc123"]); nothing catch err; err end
refusal isa ArgumentError || error("expected the request to be refused")
print(refusal.msg)
```

A run in a container made beforehand, with `tool_choice="required"` so that the model runs the
tool, and `include=["code_interpreter_call.outputs"]` for what the code printed:

```julia
write("prices.csv", "item,price\nlantern,24.50\nrope,12.00\ncompass,38.75\n")
c = create_container(name="price-check")
c isa ContainerSuccess || error("create failed: $c")
cid = c.response.id
c.response.status, c.response.raw["memory_limit"], c.response.raw["expires_after"]
# Run on 25 October 2026 with UniLM's development branch after 0.26.0 on gpt-6-sol:
# ("running", "1g", Dict{String, Any}("anchor" => "last_active_at", "minutes" => 20))

f = add_container_file(cid, "prices.csv")
f isa ContainerSuccess || error("adding the file failed: $f")
f.response.path
# "/mnt/data/1e0d7b1b7659c070ddb7f5f6d5ca1c79-prices.csv"

r = respond("Use the python tool to read prices.csv and give the total of the price column.";
            tools=[code_interpreter(container=cid)], tool_choice="required",
            include=["code_interpreter_call.outputs"])
r isa ResponseSuccess || error("respond failed: $r")
output_text(r)
# "The total of the `price` column is **75.25**."

call = only(code_interpreter_outputs(r))
print(call["code"])
# import pandas as pd
# p='/mnt/data/1e0d7b1b7659c070ddb7f5f6d5ca1c79-prices.csv'
# df=pd.read_csv(p)
# print(df.columns.tolist())
# print(df['price'].sum())

call["container_id"] == cid, call["outputs"]
# (true, Any[Dict{String, Any}("type" => "logs", "logs" => "['item', 'price']\n75.25\n")])

round(estimated_cost(r); digits=6)
# 0.004138

delete_container(cid)
# ContainerDeleteSuccess("cntr_6ade31479f108191a660d5752ee1f4e70cd28ff898e3e3c7", true)
```

[`code_interpreter_outputs`](@ref) returns the response's `code_interpreter_call` items as OpenAI
sent them, each with the `code` the model ran, the `container_id` it ran in and, with the
`include`, its `outputs`. "A container expires if it is not used for 20 minutes. When this happens,
using the container in `v1/responses` will fail", and its data is discarded: "You should download
any files you may need from the container while it is active"
([Code Interpreter](https://developers.openai.com/api/docs/guides/tools-code-interpreter#expiration)).
[`create_container`](@ref)`(; name, file_ids, expires_after)` sends no `memory_limit`, so a
container it makes has OpenAI's default, `"1g"`; [`retrieve_container`](@ref),
[`list_containers`](@ref) and [`delete_container`](@ref) read, list and delete containers, and
[`add_container_file`](@ref)`(id, path)` uploads a file into one, returning a `ContainerSuccess`
whose `response` is the file's record, its `path` where the code sees it. UniLM reads the local
file as it sends, so a path that is no file makes the result a `ContainerCallError`. It does not
wrap the endpoints that list, read or delete the files in a container. A container bills "1 GB
\$0.03, 4 GB \$0.12, 16 GB \$0.48, 64 GB \$1.92 per 20-minute session per container"
([pricing](https://developers.openai.com/api/docs/pricing#built-in-tools)), which
[`estimated_cost`](@ref) leaves out ([Cost and usage](@ref openai_cost)).

## [Batches](@id openai_batch)

The Batch API runs requests "with 50% lower costs, a separate pool of significantly higher rate
limits, and a clear 24-hour turnaround time"
([Batch API](https://developers.openai.com/api/docs/guides/batch)). Its input is a `.jsonl` file,
uploaded with the purpose `"batch"`, each of whose lines is one request: a unique `custom_id`, the
`method`, the `url` — `/v1/responses`, `/v1/chat/completions`, `/v1/embeddings`,
`/v1/completions`, `/v1/moderations`, `/v1/images/generations` or `/v1/images/edits` — and the
`body`, "the same as the parameters for the underlying endpoint"; "each input file can only include
requests to a single model", and "A single batch may include up to 50,000 requests". The bodies can
be the ones UniLM would send, so its request rules hold for them: [`UniLM.encode_agentic`](@ref) of
a `Respond`, or [`UniLM.encode_request`](@ref) of a `Chat`.

```julia
using JSON
questions = ["What is the capital of Norway?", "What is the capital of Peru?"]
lines = [JSON.json(Dict("custom_id" => "capital-$i", "method" => "POST", "url" => "/v1/responses",
                        "body" => JSON.parse(UniLM.encode_agentic(OPENAIServiceEndpoint,
                                      Respond(input=q, instructions="Answer with the city's name only.")))))
         for (i, q) in enumerate(questions)]
write("capitals.jsonl", join(lines, "\n") * "\n")
lines[1]
# Run on 25 October 2026 with UniLM's development branch after 0.26.0:
# "{\"body\":{\"input\":\"What is the capital of Norway?\",\"instructions\":\"Answer with the city's name only.\",\"model\":\"gpt-6-sol\"},\"custom_id\":\"capital-1\",\"method\":\"POST\",\"url\":\"/v1/responses\"}"

input = upload_file("capitals.jsonl", "batch")
input isa FileSuccess || error("upload failed: $input")
batch = create_batch(input.response.id, "/v1/responses"; metadata=Dict("job" => "capitals"))
batch isa BatchSuccess || error("the batch was not created: $batch")
batch.response.status, get(batch.response.raw, "completion_window", nothing)
# ("validating", "24h")

seen = retrieve_batch(batch.response.id)   # one read
seen isa BatchSuccess || error("the read failed: $seen")
seen.response.status
# "validating"

stopped = cancel_batch(batch.response.id)
stopped isa BatchSuccess || error("the cancel failed: $stopped")
stopped.response.status
# "cancelling"
```

Read back once the cancel had settled:

```julia
settled = retrieve_batch(batch.response.id)
settled isa BatchSuccess || error("the read failed: $settled")
settled.response.status, settled.response.request_counts, settled.response.output_file_id
# Read on 25 October 2026, eleven minutes after the cancel:
# ("cancelled", Dict{String, Any}("completed" => 0, "failed" => 0, "total" => 0), nothing)
```

[`create_batch`](@ref)`(input_file_id, endpoint; completion_window="24h", metadata)` sends the
`endpoint` as given — "the completion window can only be set to `24h`" —, and
[`retrieve_batch`](@ref), [`list_batches`](@ref) and [`cancel_batch`](@ref) read, list and cancel
batches. A batch moves through OpenAI's statuses: `validating`, `in_progress`, `finalizing` and
`completed`; `failed` when "the input file has failed the validation process"; `expired` when it
"was not able to be completed within the 24-hour time window"; and `cancelling`, which "may take up
to 10 minutes", then `cancelled`. [`poll_batch`](@ref) reads a batch until it is `completed`,
`failed`, `cancelled` or `expired` ([Long-running jobs](@ref openai_long_jobs)).

The results are files: `output_file_id` names the output, which "will have one response line for
every successful request line in the input file", and `error_file_id` the errors; "the output line
order **may not match** the input line order", so match lines by `custom_id`. Either id is
`nothing` while the batch has no such file, as for the batch cancelled above, so read it before
calling [`file_content`](@ref):

```julia
done = poll_batch(batch.response.id)
done isa BatchSuccess || error("the batch did not finish: $done")
out = done.response.output_file_id
if !isnothing(out)
    content = file_content(out)
    content isa FileContentSuccess || error("the download failed: $content")
    save_file_content(content, "capitals_out.jsonl")
end
```

An expired batch's "unfinished requests within that batch are cancelled, and any responses to
completed requests are made available via the batch's output file. You will be charged for tokens
consumed from any completed requests." The `batch.completed`, `batch.cancelled`, `batch.expired`
and `batch.failed` webhooks report the end without a poll ([Webhooks](@ref openai_webhooks)).

## [Fine-tuning](@id openai_fine_tuning)

"OpenAI is winding down the fine-tuning platform. The platform is no longer accessible to new
users. Eligible existing customers can create training jobs until January 6, 2027, unless the base
model shuts down earlier", and "Fine-tuned models remain available for inference until their base
model shuts down" ([supervised fine-tuning](https://developers.openai.com/api/docs/guides/supervised-fine-tuning)).
The default model takes no fine-tuning: its model page lists `v1/fine-tuning` as "Not supported"
([GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)). No job was run for this
page, so this block has no output:

```julia
train = upload_file("train.jsonl", "fine-tune")
train isa FileSuccess || error("upload failed: $train")
job = create_fine_tuning_job(model=base_model, training_file=train.response.id,
                             method=Dict("type" => "supervised",
                                         "supervised" => Dict("hyperparameters" => Dict("n_epochs" => 2))))
job isa FineTuningSuccess || error("the job was not created: $job")
r = retrieve_fine_tuning_job(job.response.id)
r isa FineTuningSuccess || error("the read failed: $r")
r.response.status, r.response.fine_tuned_model
```

`base_model` is a model your organization may still fine-tune. [`create_fine_tuning_job`](@ref) sends
`model`, `training_file` and, when set, `validation_file`, `method` — a `Dict` that names the
`"supervised"`, `"dpo"` or `"reinforcement"` method with its hyperparameters —, `suffix` and
`metadata`. A job's status is one of "`validating_files`, `queued`, `running`, `pausing`, `paused`,
`succeeded`, `failed`, or `cancelled`" ([Fine-tuning](https://developers.openai.com/api/reference/resources/fine_tuning)),
and `fine_tuned_model` names the model it made, which a `Chat` or a `Respond` takes as its `model`.
No verb waits for a job: read it with [`retrieve_fine_tuning_job`](@ref), page through
[`list_fine_tuning_events`](@ref) and [`list_fine_tuning_checkpoints`](@ref) (`limit`, `after`),
or take the `fine_tuning.job.succeeded`, `fine_tuning.job.failed` and `fine_tuning.job.cancelled`
webhooks ([Webhooks](@ref openai_webhooks)). [`cancel_fine_tuning_job`](@ref) and
[`list_fine_tuning_jobs`](@ref) cancel and list jobs; OpenAI's pause, resume and checkpoint
permissions are not wrapped.

## [Moderation](@id openai_moderation)

[`moderate`](@ref)`(input; model="omni-moderation-latest")` classifies text and images with OpenAI's
Moderations API, `POST /v1/moderations`. "The `omni-moderation-latest` model accepts text and image
inputs. It doesn't classify audio. The moderation endpoint is free to use, and image files can be up
to 20 MB" ([moderation](https://developers.openai.com/api/docs/guides/moderation)). `input` is a
string, a vector of strings, or a vector of content parts, text and `image_url` ones, which are one
input together:

```julia
m = moderate("I will find where you live and hurt you.")
m isa ModerationSuccess || error("moderation failed: $m")
is_flagged(m), m.response.model
# Run on 25 October 2026 with UniLM's development branch after 0.26.0:
# (true, "omni-moderation-latest")

row = m.response.results[1]
sort!([k for (k, v) in row.categories if v === true])
# 3-element Vector{String}:
#  "harassment"
#  "harassment/threatening"
#  "violence"

first(sort!(collect(row.category_scores); by=last, rev=true), 3)
# 3-element Vector{Pair{String, Any}}:
#                "violence" => 0.8667556675090715
#  "harassment/threatening" => 0.5270943475735724
#              "harassment" => 0.48830641686805554

two = moderate(["The ferry leaves at nine.", "I will find where you live and hurt you."])
two isa ModerationSuccess || error("moderation failed: $two")
[r.flagged for r in two.response.results], is_flagged(two)
# (Bool[0, 1], true)
```

A [`ModerationResponse`](@ref) holds one [`ModerationResult`](@ref) per input — its `flagged`, its
`categories`, each `true` or `false`, and its `category_scores` — and the `model`; each result keeps its row as
`raw`. A reply whose rows do not match the inputs one for one, or a row without `flagged`, is a
`ModerationCallError`: a verdict that cannot be matched to its input is no verdict.
[`is_flagged`](@ref) is `true` when any row is flagged, and throws an `ArgumentError` for a
`ModerationFailure` or a `ModerationCallError` rather than answering `false`, which would let
unmoderated content through: check [`issuccess`](@ref) first, and decide what a missing verdict
means. A `Chat` or a `Respond` can also moderate its own input and output with a
[`ModerationConfig`](@ref), whose results only a `Respond`'s result keeps
([Request options](@ref openai_responses_options), [Chat Completions](@ref openai_chat_options)):
`ModerationConfig(; model, input_mode, output_mode)` needs its `model`, and sends each mode set,
`"score"` or `"block"`, as the `policy` of its side. The
`moderation` of an image request is another setting, the filtering level of the image models
([Image generation](@ref openai_images)).

## [Webhooks](@id openai_webhooks)

OpenAI's webhooks "allow you to receive real-time notifications about events in the API, such as
when a batch completes, a background response is generated, or a fine-tuning job finishes. Webhooks
are delivered to an HTTP endpoint you control, following the Standard Webhooks specification", and
each endpoint, made on OpenAI's dashboard, comes with "a signing secret to use for server-side
verification of incoming webhook requests" ([webhooks](https://developers.openai.com/api/docs/guides/webhooks)).
[`verify_webhook`](@ref)`(payload, headers, secret)` checks a delivery against it, with no request:
`payload` is the raw body, `headers` the request's headers, which hold `webhook-id`,
`webhook-timestamp` and `webhook-signature`, and `secret` the signing secret, with or without its
`whsec_` prefix. It returns `true` when a fresh `v1` signature of the delivery is valid, `false`
otherwise. A stored delivery, here with a made-up secret, verifies with `tolerance_seconds=Inf`,
which skips the time check:

```@example openai
secret = "whsec_dW5pbG0tZG9jcy1leGFtcGxlLXNlY3JldC0zMmJ5dGU="
payload = """{"object": "event", "id": "evt_6ade2f3c7a3881908e0a6a1f2c3b4d5e", "type": "batch.completed", "created_at": 1792946400, "data": {"id": "batch_6ade2f3a1b2c8190a7d6e5f4c3b2a190"}}"""
headers = Dict("webhook-id" => "wh_6ade2f3c90f48190b5e1c2a7d4f61b0e", "webhook-timestamp" => "1792946400",
               "webhook-signature" => "v1,KuH1/bBMozlJfbeUhblkBOne1gUNyD41q48+86kVGAQ=")
verify_webhook(payload, headers, secret; tolerance_seconds=Inf), verify_webhook(payload, headers, secret)
```

The second call keeps the default window, 300 seconds either side of now, and the delivery is
older, so it fails, as a replayed delivery does. [`parse_webhook`](@ref) reads a verified payload
into a [`WebhookEvent`](@ref):

```@example openai
event = parse_webhook(payload)
event.type, event.data["id"], event.type in WEBHOOK_EVENTS
```

[`WEBHOOK_EVENTS`](@ref) lists the event types: responses, batches and fine-tuning jobs that end,
incoming calls of the Realtime and Live APIs, and safety notices. It leaves out the `eval.run.*`
events, which OpenAI sends until the Evals API shuts down on 30 November 2026, so a filter on it
drops them. On a server, the payload is the request's body as it arrived, before any JSON parsing,
here in an `HTTP.serve` handler, with `handle_batch` your own function:

```julia
using HTTP
HTTP.serve("0.0.0.0", 8000) do req
    payload = String(req.body)
    verify_webhook(payload, req.headers, ENV["OPENAI_WEBHOOK_SECRET"]) || return HTTP.Response(400)
    event = parse_webhook(payload)
    event.type == "batch.completed" && Threads.@spawn handle_batch(event.data["id"])
    return HTTP.Response(200)
end
```

OpenAI asks for a quick `2xx`: "If the endpoint doesn't return a successful (`2xx`) status code, or
doesn't respond within a few seconds, the webhook request will be retried. OpenAI will continue to
attempt delivery for up to 72 hours with exponential backoff", and it "may deliver duplicate copies
of the same webhook event. You can use the `webhook-id` header as an idempotency key to
deduplicate" ([webhooks](https://developers.openai.com/api/docs/guides/webhooks#handling-webhook-requests-on-a-server)).
A negative or `NaN` tolerance is refused, and so is a secret that `base64decode` cannot decode
([Limits and refusals](@ref openai_platform_limits)). `base64decode` skips characters outside the
base64 alphabet, so a secret with stray characters, such as `whsec_***`, decodes to another key, and
every delivery then returns `false`.

## [Long-running jobs](@id openai_long_jobs)

Four kinds of work go on at OpenAI after the call that starts them returns:

| Job | Started by | Read back by | Final statuses | Stopped by | Webhook events |
| :--- | :--- | :--- | :--- | :--- | :--- |
| a vector-store file batch | `create_file_batch` | `poll_file_batch`, every 2 s for at most 300 s | `completed`, `failed`, `cancelled` | — (OpenAI's cancel is not wrapped) | — |
| a batch | `create_batch` | `poll_batch`, every 10 s for at most 86,400 s | `completed`, `failed`, `cancelled`, `expired` | `cancel_batch` | `batch.*` |
| a fine-tuning job | `create_fine_tuning_job` | `retrieve_fine_tuning_job`, called by you | `succeeded`, `failed`, `cancelled` | `cancel_fine_tuning_job` | `fine_tuning.job.*` |
| a background response | `respond(…; background=true)` | `get_response`, called by you | any but `queued` and `in_progress` | `cancel_response` | `response.*` |

[`poll_batch`](@ref) and [`poll_file_batch`](@ref) read their job every `interval` seconds until it
reaches a final status, which they return, for at most `timeout` seconds of wall-clock time (`Inf`
waits indefinitely). A failure worth another read — a status UniLM sends a request again after
([When something fails](@ref openai_failures)), or a GET that timed out — is polled through; any
other failure is returned as it came. When the time runs out, the result is the family's call
error, `BatchCallError` or `VectorStoreCallError`, with a `UniLMTimeout(:deadline, …)` as `cause`
and the last batch it read as `last_observed` (`nothing` if no read succeeded). `cancel` (the
ambient token unless given) ends a poll at once, mid-read or mid-pause, with the call error and a
`UniLMCancelled` as `cause`; `config` bounds each read.

A background response is read back with [`get_response`](@ref) and stopped with
[`cancel_response`](@ref) ([Stored responses](@ref openai_stored_responses)), here with one read
between them:

```julia
started = respond("Write a detailed history of lighthouses, century by century.";
                  background=true, max_output_tokens=4000)
started isa ResponseSuccess || error("the background request failed: $started")
id = started.response.id
response_status(started)
# Run on 25 October 2026 with UniLM's development branch after 0.26.0 on gpt-6-sol:
# "queued"

sleep(3)
response_status(get_response(id))
# "in_progress"

response_status(cancel_response(id)), response_status(cancel_response(id))
# ("cancelled", "cancelled")
```

"Keep polling while the request is in the queued or in_progress state. When it leaves these states,
it has reached a final (terminal) state", and "Cancelling twice is idempotent - subsequent calls
simply return the final `Response` object"
([background mode](https://developers.openai.com/api/docs/guides/background)). A background response
that failed comes back from `get_response` and `cancel_response` as a `ResponseSuccess` whose
[`response_status`](@ref) is `"failed"`, while `respond` returns a failed response as a
`ResponseFailure`. `background=true` with `store=false` is refused, a rule of UniLM's, not OpenAI's,
whose guide says "Background requests can use `store=false`"
([background mode](https://developers.openai.com/api/docs/guides/background)); the row is in
[Limits and refusals](@ref openai_limits).

## [Limits and refusals](@id openai_platform_limits)

The sizes OpenAI allows — a file's 512 MB, an upload part's 64 MB, a batch file's 200 MB and 50,000
requests — are OpenAI's to check, and UniLM checks none of them. These are refused with an
`ArgumentError` before anything is sent:

| Refused | Why: the message's words | What to do |
| :--- | :--- | :--- |
| a platform verb on an endpoint that declares its capabilities without the verb's | `is not supported by`: the endpoint has no such API | `OPENAIServiceEndpoint` |
| `upload_file` with a path that is no file | `file not found:` | the path of the file |
| a purpose other than `"assistants"`, `"batch"`, `"fine-tune"`, `"vision"` and `"user_data"`, on `upload_file` or `create_upload` | `the Files API takes`: the purposes of OpenAI's Files API | one of them |
| the purpose `"evals"` | `purpose "evals" is for OpenAI's Evals API`: the Evals API is read-only from 31 October 2026 and shuts down on 30 November 2026 | another purpose |
| a `poll_batch` or `poll_file_batch` `interval` that is not a finite number above 0 | `interval must be a finite number of seconds > 0` | a pause in seconds |
| a poll `timeout` that is `NaN`, not above 0, or finite and above 10⁹ seconds | `timeout must not be NaN`, `timeout must be > 0 seconds`, `timeout must be at most`: `Inf` is the way to wait without bound | a number of seconds, or `Inf` |
| `code_interpreter` `file_ids` with a container id | `code_interpreter file_ids go inside an auto container`: a container made beforehand holds the files it was made with or given | `create_container(; file_ids)` or `add_container_file` |
| `code_interpreter` `file_ids` beside a container `Dict` that names `file_ids` | `code_interpreter file_ids given twice`: the files would be sent twice | one of the two |
| `verify_webhook` with a negative or `NaN` `tolerance_seconds` | `tolerance_seconds must be nonnegative`: only `Inf` turns the replay check off | a number of seconds, or `Inf` |
| `verify_webhook` with a secret that `base64decode` cannot decode, such as `whsec_A` | `webhook signing secret is not valid base64`: a misconfigured secret, which a `false` would hide; a secret with characters outside the base64 alphabet, such as `whsec_***`, decodes, to another key, and is not refused: every delivery then returns `false` | the signing secret OpenAI gave the endpoint |
