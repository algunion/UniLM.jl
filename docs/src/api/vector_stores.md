# [Vector Stores API](@id vector_stores_api)

Create, update and search vector stores, and manage their files and file batches;
the stores power the Responses `file_search` tool. Upload files via the Files API
first. OpenAI only.

## Objects

```@docs
VectorStoreObject
VectorStoreFileObject
VectorStoreFileBatch
VectorStoreList
VectorStoreFileList
VectorStoreFileContent
VectorStoreSearchPage
VectorStoreSearchResult
```

## Request Type

```@docs
VectorStoreFileEntry
```

## Result Types

```@docs
VectorStoreSuccess
VectorStoreListSuccess
VectorStoreFileSuccess
VectorStoreFileListSuccess
VectorStoreContentSuccess
VectorStoreSearchSuccess
VectorStoreBatchSuccess
VectorStoreDeleteSuccess
VectorStoreFailure
VectorStoreCallError
```

## Request Functions

```@docs
create_vector_store
retrieve_vector_store
update_vector_store
list_vector_stores
delete_vector_store
search_vector_store
add_vector_store_file
list_vector_store_files
retrieve_vector_store_file
update_vector_store_file
delete_vector_store_file
vector_store_file_content
create_file_batch
retrieve_file_batch
poll_file_batch
cancel_file_batch
list_file_batch_files
```

## Accessors

```@docs
vector_store_id
```

## Usage

```julia
# Create a vector store, then add files already uploaded via the Files API
store = create_vector_store(name="docs", description="Product manuals")
store isa VectorStoreSuccess && println("Created: ", vector_store_id(store.response))
vs = vector_store_id(store.response)

# Add a single file with attributes a search can filter on, or a batch, then poll the batch
add_vector_store_file(vs, "file-abc123"; attributes=Dict("team" => "support", "year" => 2026))
batch = create_file_batch(vs, ["file-abc123", "file-def456"])
batch isa VectorStoreBatchSuccess && poll_file_batch(vs, batch.response.id)

# Search the store directly, keeping the chunks of files whose attribute matches
hits = search_vector_store(vs, "How do I reset the device?"; max_num_results=5,
                           filters=Dict("type" => "eq", "key" => "team", "value" => "support"))
hits isa VectorStoreSearchSuccess && [(h.filename, h.score) for h in hits.response.data]

# The store's files: list, read, update their attributes, detach
list_vector_store_files(vs; filter="completed")
vector_store_file_content(vs, "file-abc123")
update_vector_store_file(vs, "file-abc123", Dict("team" => "sales"))
delete_vector_store_file(vs, "file-abc123")

# List, retrieve, rename, delete
stores = list_vector_stores(limit=10, order="asc")
retrieve_vector_store(vs)
update_vector_store(vs; name="manuals")
delete_vector_store(vs)
```
