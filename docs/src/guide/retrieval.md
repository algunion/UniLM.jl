# [Retrieval & File Search](@id retrieval_guide)

Ground a model's answer in your own documents: upload a local file, index it into a vector
store, then let the Responses [`file_search`](@ref) tool retrieve the relevant chunks at
answer time. This is the end-to-end Retrieval-Augmented Generation (RAG) pipeline in UniLM.

!!! note
    Files, vector stores, and `file_search` are **OpenAI-only** (see the
    [Files API](@ref files_api) and [Vector Stores API](@ref vector_stores_api)).
    Calling them on another endpoint fails capability validation at request time.

!!! warning
    A `file_search` tool needs a **real** vector-store id. Do not hard-code a placeholder
    such as `"vs_abc123"` — obtain the id from [`create_vector_store`](@ref) and thread it
    through every later step, exactly as shown below.

## Pipeline at a glance

1. [`upload_file`](@ref) — send the local file to the Files API (`purpose="user_data"`).
2. [`create_vector_store`](@ref) — create a store and capture its real id.
3. [`create_file_batch`](@ref) + [`poll_file_batch`](@ref) — attach the file(s) and block until indexing completes.
4. [`respond`](@ref) with `tools=[file_search([vs_id])]` — ask a question against that store.
5. [`output_text`](@ref) + [`file_search_results`](@ref) — read the grounded answer and the retrieved chunks.

The `julia` blocks below form **one continuous session** — each step reuses ids produced by
the previous one. They hit the network (and need real ids), so they are shown as plain
`julia` with expected output in comments rather than executed.

## 1. Upload the file

[`upload_file`](@ref) posts a file as `multipart/form-data`. For retrieval, use
`purpose="user_data"`. On success it returns a `FileSuccess` wrapping a
[`FileObject`](@ref); the file id you need next is `.response.id`.

```julia
using UniLM

up = upload_file("handbook.md", "user_data")

up isa FileSuccess || error("upload failed: ", up)
file_id = up.response.id
@show file_id               # file_id = "file-AbC123..."
@show up.response.filename  # "handbook.md"
@show up.response.purpose   # "user_data"
```

`upload_file` returns `FileFailure` (HTTP error) or `FileCallError` (transport error)
instead of throwing on a failed request — always check the result type before reading `.response`.

## 2. Create a vector store

[`create_vector_store`](@ref) returns a `VectorStoreSuccess` wrapping a
[`VectorStoreObject`](@ref). Capture its **real** id via `.response.id` (or the
[`vector_store_id`](@ref) accessor) — this is the id every later step depends on.

```julia
vs = create_vector_store(name="handbook")

vs isa VectorStoreSuccess || error("store creation failed: ", vs)
vs_id = vector_store_id(vs.response)   # identical to vs.response.id
@show vs_id                            # vs_id = "vs_68f0..."   ← the REAL id, not a placeholder
```

!!! tip
    You can attach already-uploaded files at creation time with
    `create_vector_store(file_ids=[file_id])`, folding step 3's attach into this call
    (indexing still runs asynchronously, so you'd still poll — see below).

## 3. Attach the file and wait for indexing

Files must be chunked and embedded before they are searchable. Attach one or more files as
a batch with [`create_file_batch`](@ref) (it takes a `Vector{String}` of file ids), then
block until the batch reaches a terminal status with [`poll_file_batch`](@ref).

```julia
batch = create_file_batch(vs_id, [file_id])
batch isa VectorStoreBatchSuccess || error("batch failed: ", batch)

done = poll_file_batch(vs_id, batch.response.id; interval=2.0, timeout=300.0)
done isa VectorStoreBatchSuccess || error("indexing did not complete: ", done)

@show done.response.status                            # "completed"
@show get(done.response.file_counts, "completed", 0)  # 1
```

[`poll_file_batch`](@ref) re-fetches the batch every `interval` seconds until its status is
`"completed"`, `"failed"`, or `"cancelled"`, returning that terminal
`VectorStoreBatchSuccess`; it returns a `VectorStoreCallError` if `timeout` elapses first.
Only proceed once the status is `"completed"` — searching a still-indexing store returns no chunks.

To attach a single file without the batch wrapper, use
[`add_vector_store_file`](@ref)`(vs_id, file_id)`; it returns a `VectorStoreFileSuccess`
immediately with the file's *initial* status, so prefer the batch + poll route whenever you
need to wait for indexing to finish.

## 4. Ask a grounded question

[`file_search`](@ref) builds the hosted tool from a **vector of store ids**. Pass the real
`vs_id` from step 2, and add `include=["file_search_call.results"]` so the API returns the
retrieved chunks alongside the answer.

```julia
result = respond(
    "What is our policy on error handling?";
    tools=[file_search([vs_id]; max_results=5)],
    include=["file_search_call.results"],
)
result isa ResponseSuccess || error("query failed: ", output_text(result))
```

The tool serialises to a small JSON object. Its wire shape is self-contained (no network),
so it can be shown live — the id here is a **placeholder for illustration only**; the
pipeline above uses the real `vs_id`:

```@setup retrieval
using UniLM
using JSON
```

```@example retrieval
tool = file_search(["vs_your_store_id"]; max_results=5)  # placeholder id — wire shape only
println("Store ids:   ", tool.vector_store_ids)
println("Max results: ", tool.max_num_results)
println("Wire JSON:   ", JSON.json(JSON.lower(tool)))
```

## 5. Read the answer and the cited chunks

The grounded answer is plain text via [`output_text`](@ref). The retrieved chunks come from
[`file_search_results`](@ref), which returns the **raw** `file_search_call` output items;
each item carries a `"results"` array (present because of the `include=` above). Those
result entries are unparsed API dicts, so read their fields with `get`:

```julia
println(output_text(result))    # the grounded, citation-backed answer

for call in file_search_results(result)
    for chunk in get(call, "results", [])
        fname = get(chunk, "filename", "")
        score = get(chunk, "score", nothing)
        text  = get(chunk, "text", "")
        println("• ", fname, "  (score=", score, ")")
        println("  ", first(text, 160))
    end
end
# • handbook.md  (score=0.83)
#   Errors are surfaced as typed result values rather than thrown exceptions ...
```

The exact keys inside each chunk (`file_id`, `filename`, `score`, `text`, `attributes`) are
defined by the OpenAI API — inspect a `call` dict to see everything returned. Without
`include=["file_search_call.results"]`, `file_search_results` still returns the
`file_search_call` items, but their `"results"` array is omitted: you get the answer, not
the chunk text.

## Cleanup

Vector stores and files persist on OpenAI's servers until you remove them:

```julia
delete_vector_store(vs_id)
delete_file(file_id)
```

## See Also

- [`upload_file`](@ref), [`FileObject`](@ref) — Files API upload and parsed object
- [`create_vector_store`](@ref), [`vector_store_id`](@ref) — store creation and id accessor
- [`create_file_batch`](@ref), [`poll_file_batch`](@ref), [`add_vector_store_file`](@ref) — attach files and wait for indexing
- [`file_search`](@ref), [`FileSearchTool`](@ref) — the Responses hosted retrieval tool
- [`respond`](@ref), [`output_text`](@ref), [`file_search_results`](@ref) — run the query and read results
- [Files API](@ref files_api), [Vector Stores API](@ref vector_stores_api) — full API reference
- [Responses API](@ref responses_guide), [Tool Calling](@ref tools_guide) — related guides
