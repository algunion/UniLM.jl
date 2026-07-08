# [Vector Stores API](@id vector_stores_api)

Create and manage vector stores, along with their files and file batches, that
power the Responses `file_search` tool. Upload files via the Files API first.
OpenAI only.

## Objects

```@docs
VectorStoreObject
VectorStoreFileObject
VectorStoreFileBatch
VectorStoreList
```

## Result Types

```@docs
VectorStoreSuccess
VectorStoreListSuccess
VectorStoreFileSuccess
VectorStoreBatchSuccess
VectorStoreDeleteSuccess
VectorStoreFailure
VectorStoreCallError
```

## Request Functions

```@docs
create_vector_store
retrieve_vector_store
list_vector_stores
delete_vector_store
add_vector_store_file
create_file_batch
retrieve_file_batch
poll_file_batch
```

## Accessors

```@docs
vector_store_id
```

## Usage

```julia
# Create a vector store, then add files already uploaded via the Files API
store = create_vector_store(name="docs")
store isa VectorStoreSuccess && println("Created: ", vector_store_id(store.response))
vs = vector_store_id(store.response)

# Add a single file, or a batch, then poll the batch to a terminal status
add_vector_store_file(vs, "file-abc123")
batch = create_file_batch(vs, ["file-abc123", "file-def456"])
batch isa VectorStoreBatchSuccess && poll_file_batch(vs, batch.response.id)

# List, retrieve, delete
stores = list_vector_stores(limit=10)
retrieve_vector_store(vs)
delete_vector_store(vs)
```
