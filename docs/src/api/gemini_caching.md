# [Gemini Context Caching & File Search](@id gemini_caching_api)

Two Gemini resources a request reads: a **context cache**, the start of a prompt stored ahead and named
by a [`Chat`](@ref)'s `cached_content`, and a **File Search store**, documents indexed for the
Interactions API's `file_search` tool ([`gemini_file_search`](@ref)). Both are native Gemini only
(`GEMINIServiceEndpoint`): the cache verbs check the `:context_caching` capability, the File Search verbs
`:file_search_stores`, and any other endpoint throws `ArgumentError` before any request. How each is
used and billed is on the Gemini pages: [Context caching](@ref gemini_caching) and
[File Search](@ref gemini_file_search).

A resource is named as Gemini names it: `"cachedContents/<id>"`, `"fileSearchStores/<id>"`,
`"fileSearchStores/<store>/documents/<id>"`, and an operation's
`"fileSearchStores/<store>/operations/<id>"` (an import's) or
`"fileSearchStores/<store>/upload/operations/<id>"` (an upload's). A cache or a store verb takes the name
or the id alone; a document or an operation verb takes the whole name. A name of another shape — an id
with a `/`, an empty or dot segment — throws `ArgumentError` before any request. A reply without a field
Gemini always sends is the verb's call error, never a result with the field made up.

## Context caches

| Verb | Gemini method | Takes |
| :--- | :--- | :--- |
| [`create_cache`](@ref) | `cachedContents.create` | a `Chat`: its system messages, other messages, `tools` and `tool_choice`; `ttl` (seconds) or `expire_time`, `display_name` |
| [`retrieve_cache`](@ref) | `cachedContents.get` | a name or id |
| [`list_caches`](@ref) | `cachedContents.list` | `limit` (`pageSize`), `after` (`pageToken`) |
| [`update_cache`](@ref) | `cachedContents.patch` | a new `ttl` or `expire_time`, nothing else |
| [`delete_cache`](@ref) | `cachedContents.delete` | a name or id |

```@docs
CacheObject
CacheList
CacheSuccess
CacheListSuccess
CacheDeleteSuccess
CacheFailure
CacheCallError
create_cache
retrieve_cache
list_caches
update_cache
delete_cache
```

## File Search stores

| Verb | Gemini method | Takes |
| :--- | :--- | :--- |
| [`create_file_search_store`](@ref) | `fileSearchStores.create` | `display_name`, `embedding_model` |
| [`retrieve_file_search_store`](@ref) | `fileSearchStores.get` | a name or id |
| [`list_file_search_stores`](@ref) | `fileSearchStores.list` | `limit` (1 to 20), `after` |
| [`delete_file_search_store`](@ref) | `fileSearchStores.delete` | a name or id; `force` (default `false`) |
| [`upload_to_file_search_store`](@ref) | `media.uploadToFileSearchStore` | a store and a local file, at most 100 MB; the document's settings |
| [`import_to_file_search_store`](@ref) | `fileSearchStores.importFile` | a store and a Files API file; the document's settings |
| [`retrieve_file_search_operation`](@ref) | `fileSearchStores.operations.get`, `fileSearchStores.upload.operations.get` | an operation's name |
| [`poll_file_search_operation`](@ref) | the same, until `done` | an operation's name; `interval`, `timeout` |
| [`list_file_search_documents`](@ref) | `fileSearchStores.documents.list` | a store; `limit` (1 to 20), `after` |
| [`retrieve_file_search_document`](@ref) | `fileSearchStores.documents.get` | a document's name |
| [`delete_file_search_document`](@ref) | `fileSearchStores.documents.delete` | a document's name; `force` (default `true`) |

A document's settings, on an upload and an import alike: `custom_metadata`, at most 20 keys, each a
string, a number or a vector of strings; `max_tokens_per_chunk` (1 to 512) and `max_overlap_tokens`
(≥ 0), the white-space chunking. An upload also takes the document's `display_name` and `mime_type`.

```@docs
FileSearchStoreObject
FileSearchDocumentObject
FileSearchOperation
FileSearchStoreList
FileSearchDocumentList
FileSearchStoreSuccess
FileSearchStoreListSuccess
FileSearchDocumentSuccess
FileSearchDocumentListSuccess
FileSearchOperationSuccess
FileSearchDeleteSuccess
FileSearchFailure
FileSearchCallError
create_file_search_store
retrieve_file_search_store
list_file_search_stores
delete_file_search_store
upload_to_file_search_store
import_to_file_search_store
retrieve_file_search_operation
poll_file_search_operation
list_file_search_documents
retrieve_file_search_document
delete_file_search_document
```

## Usage

```julia
using UniLM, Dates

# A cache of a long document, for gemini-3.8-flash, kept ten minutes.
doc = read("handbook.txt", String)
base = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
            messages=[Message(Val(:system), "Answer from the handbook."), Message(Val(:user), doc)])
made = create_cache(base; ttl=600, display_name="handbook")
cache = made.response                       # a CacheObject: name, total_tokens, expire_time
chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.8-flash", cached_content=cache,
            messages=[Message(Val(:user), "How many days of leave?")])
r = chatrequest!(chat)                      # r.usage.cached_tokens: the cache's tokens, at the cache-hit rate
update_cache(cache.name; ttl=1800)          # half an hour more
delete_cache(cache.name)

# A File Search store with one document, searched through the Interactions API.
store = create_file_search_store(; display_name="handbook").response
op = upload_to_file_search_store(store.name, "handbook.pdf"; custom_metadata=Dict("year" => 2026)).response
done = poll_file_search_operation(op.name)
done isa FileSearchOperationSuccess && isnothing(done.response.error)
r = respond("How many days of leave?"; service=GEMINIServiceEndpoint, model="gemini-3.8-flash",
            tools=[gemini_file_search(store.name; metadata_filter="year = 2026")])
delete_file_search_store(store.name; force=true)
```
