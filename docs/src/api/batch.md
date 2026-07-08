# [Batch API](@id batch_api)

Create, retrieve, cancel, and list async bulk jobs, and poll them to completion.
The Batch API processes large request sets asynchronously at roughly half the cost:
the input is a JSONL file uploaded via the Files API with `purpose="batch"`, and
output is fetched with [`file_content`](@ref) using `batch.output_file_id`. OpenAI only.

## Parsed Objects

```@docs
BatchObject
BatchList
```

## Result Types

```@docs
BatchSuccess
BatchListSuccess
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
```

## Usage

```julia
# Upload a JSONL request file, then create a batch job
upload = upload_file("requests.jsonl", "batch")
batch = create_batch(upload.response.id, "/v1/chat/completions")

# Poll until terminal (completed / failed / cancelled / expired)
done = poll_batch(batch.response.id)
if done isa BatchSuccess && done.response.status == "completed"
    output = file_content(done.response.output_file_id)
    output isa FileContentSuccess && save_file_content(output, "results.jsonl")
end

# List jobs, or cancel one
list_batches(limit=10)
cancel_batch(batch.response.id)
```
