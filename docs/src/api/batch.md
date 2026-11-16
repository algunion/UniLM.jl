# [Batch API](@id batch_api)

Create, retrieve, cancel, list and delete asynchronous batch jobs, poll them to completion, and read
their results, on OpenAI, Anthropic and Gemini. A batch runs its requests within 24 hours at the
provider's Batch rates — half its standard ones on every model each page lists, but for a cache hit on
the six Gemini models whose Batch tab keeps the standard cache-hit price (`gemini-3.1-pro-preview`,
`gemini-3.1-pro-preview-customtools`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`
and `gemini-2.5-flash-lite`) — and [`estimated_cost`](@ref) prices a batch result at them.

- [`create_batch`](@ref)`(requests; service)` takes the requests as `custom_id => Chat` pairs and
  sends each provider its own form of them: OpenAI a JSONL file of Chat Completions requests to one
  model, uploaded with purpose `"batch"`, then the batch; Anthropic a Message Batch of Messages
  requests; Gemini inline `generateContent` requests to `batchGenerateContent`. OpenAI's own file-based form,
  `create_batch(input_file_id, endpoint)`, runs any endpoint its Batch API serves.
- [`batch_results`](@ref) reads the results of an ended batch as the results [`chatrequest!`](@ref)
  returns — an [`LLMSuccess`](@ref), [`LLMFailure`](@ref) or [`LLMCallError`](@ref) per custom id —,
  decoded by the provider's own decoder, each success's usage at the tier `"batch"`. On OpenAI it
  reads a `/v1/chat/completions` batch alone: another endpoint's is a `BatchCallError` naming
  [`file_content`](@ref)`(output_file_id)`, which reads its output file as OpenAI writes it.
- [`retrieve_batch`](@ref), [`list_batches`](@ref), [`cancel_batch`](@ref) and [`poll_batch`](@ref)
  run on all three; [`delete_batch`](@ref) on Anthropic and Gemini; [`update_batch`](@ref) (a batch's
  priority) on Gemini.

Each verb first checks that its `service` has the `:batch` capability, which
`OPENAIServiceEndpoint`, `ANTHROPICServiceEndpoint` and `GEMINIServiceEndpoint` declare.

## Parsed Objects

```@docs
BatchObject
BatchList
```

## Result Types

```@docs
BatchSuccess
BatchListSuccess
BatchResultsSuccess
BatchDeleteSuccess
BatchFailure
BatchCallError
```

## Request Functions

```@docs
create_batch
retrieve_batch
cancel_batch
list_batches
poll_batch
batch_results
delete_batch
update_batch
```

## Usage

```julia
# Two requests, as Chats, on any of the three providers
ask(q) = Chat(service=ANTHROPICServiceEndpoint, model="claude-haiku-5-5", max_tokens=64,
              messages=[Message(Val(:user), q)])
batch = create_batch(["norway" => ask("Capital of Norway?"), "peru" => ask("Capital of Peru?")])
batch isa BatchSuccess || error("the batch was not created: $batch")

# Poll until it ends (Anthropic: "ended"), then read the results by custom id
done = poll_batch(batch.response.id; service=ANTHROPICServiceEndpoint, interval=60)
results = batch_results(batch.response.id; service=ANTHROPICServiceEndpoint)
if results isa BatchResultsSuccess
    for (id, r) in results.results
        r isa LLMSuccess && println(id, ": ", r.message.content, " (USD ", estimated_cost(r), ")")
    end
end

# OpenAI's file-based form: upload a JSONL request file, then create a batch from it
upload = upload_file("requests.jsonl", "batch")
openai_batch = create_batch(upload.response.id, "/v1/chat/completions"; output_expires_after=86_400)

# List jobs, cancel one, delete an ended one (Anthropic and Gemini)
list_batches(; limit=10, service=GEMINIServiceEndpoint)
cancel_batch(openai_batch.response.id)
delete_batch(batch.response.id; service=ANTHROPICServiceEndpoint)
```
