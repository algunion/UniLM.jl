# [Retrieval & file search](@id retrieval_guide__next)

```@setup openai
using UniLM
```

File search grounds a response in your own documents. It is a tool of the Responses API that
lets the model "retrieve information in a knowledge base of previously uploaded files through
semantic and keyword search", "a hosted tool managed by OpenAI"
([file search](https://developers.openai.com/api/docs/guides/tools-file-search)): the files are
uploaded, indexed into a vector store, and searched when a `respond` call carries
[`file_search`](@ref). Five steps make the pipeline, each id passed on to the next:

| Step | Call | You get |
| :--- | :--- | :--- |
| upload a file | `upload_file(path, "user_data")` | a `FileSuccess`, the file's `id` in `response.id` |
| make a store | `create_vector_store(; name, expires_after, …)` | a `VectorStoreSuccess`, its id read by `vector_store_id` |
| index the file | `create_file_batch(vs_id, [file_id])`, then `poll_file_batch` | the batch once its status is final, `"completed"` when the file is searchable |
| ask | `respond(question; tools=[file_search([vs_id])], include=["file_search_call.results"])` | the answer, and the chunks it was grounded on |
| clean up | `delete_vector_store(vs_id)`, `delete_file(file_id)` | the deletions, confirmed |

The Files and Vector Stores calls are not recorded, so their output on this page is quoted from a
dated run; the question is executed. Each of them is a platform verb: one attempt, a typed result,
never a throw for what the service answers ([Platform APIs](@ref openai_platform_shared)).

## [Build the store](@id openai_retrieval_store)

The [policy file](../../assets/returns_policy.txt) is uploaded, a store is made for it, and the file
is attached as a batch, which [`poll_file_batch`](@ref) reads back until it is indexed:

```julia
up = upload_file("returns_policy.txt", "user_data")
up isa FileSuccess || error("upload failed: $up")
file_id = up.response.id
(up.response.filename, up.response.purpose, up.response.bytes)
# Run on 25 October 2026 with UniLM's development branch after 0.26.0:
# ("returns_policy.txt", "user_data", 800)

vs = create_vector_store(name="returns-policy", expires_after=Dict("anchor" => "last_active_at", "days" => 1))
vs isa VectorStoreSuccess || error("store creation failed: $vs")
vs_id = vector_store_id(vs.response)
# "vs_6ade2f0d72148191a5b970d85c9826e9"

batch = create_file_batch(vs_id, [file_id])
batch isa VectorStoreBatchSuccess || error("attaching failed: $batch")
batch.response.status
# "in_progress"

done = poll_file_batch(vs_id, batch.response.id)
done isa VectorStoreBatchSuccess || error("indexing did not finish: $done")
done.response.status, done.response.file_counts
# ("completed", Dict{String, Any}("cancelled" => 0, "completed" => 1, "failed" => 0, "in_progress" => 0, "total" => 1))
```

"When you add a file to a vector store it will be automatically chunked, embedded, and indexed"
([retrieval](https://developers.openai.com/api/docs/guides/retrieval#vector-stores)), which takes
a while: a batch is `"in_progress"` until it is `"completed"`, `"cancelled"` or `"failed"`, and
`file_counts` says how many of its files reached each status. `poll_file_batch` reads the batch
every 2 seconds, for at most 300, and returns it at the first of those three statuses; here it
took about 6 seconds. What it returns when the time runs out or a GET fails is on
[Long-running jobs](@ref openai_long_jobs). Search the store once the batch is `"completed"`, and
check `file_counts` for a file that failed.

- **Attaching.** [`create_file_batch`](@ref) takes the ids of files already uploaded, and so does
  `create_vector_store(file_ids=…)`, which attaches them as the store is made; that store has no
  batch to poll, so read it back with [`retrieve_vector_store`](@ref) until its `file_counts` show
  no file `"in_progress"`. [`add_vector_store_file`](@ref) attaches one file and returns its first
  status.
- **Chunking.** By default a file is split into chunks of 800 tokens that overlap by 400.
  `chunking_strategy`, on `create_vector_store`, `create_file_batch` or `add_vector_store_file`,
  sets other sizes, as
  `Dict("type" => "static", "static" => Dict("max_chunk_size_tokens" => 1200, "chunk_overlap_tokens" => 200))`:
  the size is "The minimum value is `100` and the maximum value is `4096`", and "the overlap must
  not exceed half of `max_chunk_size_tokens`"
  ([Create vector store](https://developers.openai.com/api/reference/resources/vector_stores/methods/create)).
- **Files.** "The maximum file size is 512 MB. Each file should contain no more than 5,000,000
  tokens per file" ([retrieval](https://developers.openai.com/api/docs/guides/retrieval#limits)); the
  [file search](https://developers.openai.com/api/docs/guides/tools-file-search#supported-files)
  guide lists the formats a store takes, from `.txt`, `.md` and `.pdf` to source code. Uploading
  and the purposes a file takes are on [Files and uploads](@ref openai_files).
- **Expiry.** `expires_after` with the anchor `"last_active_at"` and a number of `days`: "Once a
  vector store expires, all associated `vector_store.file` objects will be deleted and you'll no
  longer be charged for them" ([retrieval](https://developers.openai.com/api/docs/guides/retrieval#expiration-policies)).
  `name` and `metadata` are sent as given.

UniLM sets no file `attributes`, the values a search's `filters` compare
([Ask with file search](@ref openai_retrieval_ask)): neither `add_vector_store_file` nor
`create_file_batch` takes them, and no verb updates them.

## [Ask with file search](@id openai_retrieval_ask)

[`file_search`](@ref)`(vector_store_ids; max_results, ranking, filters)` makes the tool, and
`include=["file_search_call.results"]` asks for the chunks the search found, "the search results of
the file search tool call"
([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create),
`include`). The call names the store the run above made, by its id:

```@example openai
vs_id = "vs_6ade2f0d72148191a5b970d85c9826e9"   # the run's store, deleted below; yours: vector_store_id(vs.response)
result = respond("How long do I have to return a jacket, and how is the refund paid?";
                 service=OPENAIServiceEndpoint,
                 instructions="Answer from the files you search, in one or two sentences.",
                 tools=[file_search([vs_id]; max_results=3)], include=["file_search_call.results"])
output_text(result)
```

```@example openai
[(hit["filename"], round(hit["score"]; digits=2)) for call in file_search_results(result) for hit in call["results"]]
```

```@example openai
u = token_usage(result)
(; u.prompt_tokens, u.completion_tokens, cost = round(estimated_cost(result); digits=6))
```

The answer comes from the file: the response's output holds a `file_search_call` item, with the
queries the model searched for and, because of `include`, their `results`, then the message, whose
`output_text` carries `file_citation` annotations naming the files it drew on.
[`file_search_results`](@ref) returns the `file_search_call` items as OpenAI sent them, so a chunk's
fields are read from the `Dict`. Without `include`, "the file search call will not return search
results by default", while "you can see annotations (references to files) in the output text"
([file search](https://developers.openai.com/api/docs/guides/tools-file-search#include-search-results-in-the-response)).
The other tools OpenAI runs are on [Hosted tools](@ref openai_hosted_tools).

The tool's options are sent under OpenAI's names:

- **`max_results`**, sent as `max_num_results`: "The maximum number of results to return. This
  number should be between 1 and 50 inclusive"
  ([Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create),
  the file search tool). Fewer results "can help reduce both token usage and latency, but may come
  at the cost of reduced answer quality"
  ([file search](https://developers.openai.com/api/docs/guides/tools-file-search#limiting-the-number-of-results)).
- **`ranking`**, sent as `ranking_options`: a `Dict` with the `ranker` (`"auto"` or
  `"default-2024-11-15"`), a `score_threshold` between 0 and 1, where "Numbers closer to 1 will
  attempt to return only the most relevant results, but may return fewer results", and the
  `hybrid_search` weights of embedding and text matches.
- **`filters`**: a comparison filter, `Dict("type" => "eq", "key" => "region", "value" => "us")`
  with the operators `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in` and `nin`, or a compound one,
  `Dict("type" => "and", "filters" => [...])`, which compare "a specific `key` in a file's
  `attributes`" ([retrieval](https://developers.openai.com/api/docs/guides/retrieval#attribute-filtering)).

UniLM checks none of their values; [`UniLM.encode_agentic`](@ref) shows what `respond` sends:

```@example openai
r = Respond(service=OPENAIServiceEndpoint, input="Which repairs are free?",
            tools=[file_search([vs_id]; max_results=5, ranking=Dict("score_threshold" => 0.5),
                               filters=Dict("type" => "eq", "key" => "region", "value" => "us"))])
print(UniLM.encode_agentic(r.service, r))
```

A search bills beyond its tokens, at "\$2.50 / 1k calls" of the tool, and the stores' storage
beyond the first, free gigabyte at "\$0.10 / GB per day"
([pricing](https://developers.openai.com/api/docs/pricing#built-in-tools)); [`estimated_cost`](@ref)
prices the tokens alone ([Cost and usage](@ref openai_cost)). With the answer and its chunks in
hand, a decision model can read the answer against a chunk and send anything but a confident
*supports* to a person ([Check claims against the source](@ref jev_llm_check_claims)), and screen
the passages you put in a prompt yourself for instructions aimed at the model
([Screen what the model reads](@ref jev_llm_screen_input)).

## [Clean up](@id openai_retrieval_cleanup)

A store and a file stay on OpenAI's servers until they are deleted, or until a store's
`expires_after` runs out:

```julia
delete_vector_store(vs_id), delete_file(file_id)
# Run on 25 October 2026 with UniLM's development branch after 0.26.0:
# (VectorStoreDeleteSuccess("vs_6ade2f0d72148191a5b970d85c9826e9", true), FileDeleteSuccess("file-UoD6Ty5NGchHpMLtz28KyY", true))
```

Deleting a file is also to "remove it from all vector stores"
([Files](https://developers.openai.com/api/reference/resources/files)), and "Removing files from a
vector store is eventually consistent, and search results may still include content from a
removed file for a short period" ([retrieval](https://developers.openai.com/api/docs/guides/retrieval#vector-store-operations)).
A delete is confirmed by the `deleted: true` of OpenAI's reply ([Platform APIs](@ref openai_platform_shared)).
Of OpenAI's vector-store endpoints, UniLM does not wrap the store's search, update and the files it
holds — listing, reading, updating or removing one — nor a batch's cancel or its file list
([Vector stores](https://developers.openai.com/api/reference/resources/vector_stores)); every
verb it has is in the [Vector Stores API](@ref vector_stores_api) reference.

Recorded on 25 October 2026 on `gpt-6-sol` with UniLM's development branch after 0.26.0.
