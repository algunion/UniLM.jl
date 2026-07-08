# [Files API](@id files_api)

Upload, list, retrieve, and delete files, and download their content. Files feed
the Responses `file_search` and code-interpreter tools, the Batch API, and
fine-tuning. OpenAI only.

## Request Type

```@docs
FileUpload
```

## Parsed Objects

```@docs
FileObject
FileList
```

## Result Types

```@docs
FileSuccess
FileListSuccess
FileContentSuccess
FileDeleteSuccess
FileFailure
FileCallError
```

## Request Functions

```@docs
upload_file
list_files
retrieve_file
delete_file
file_content
save_file_content
```

## Usage

```julia
# Upload a file for use with file_search / batch / fine-tuning
result = upload_file("data.jsonl", "batch")
result isa FileSuccess && println("Uploaded: ", result.response.id)

# List, download, delete
files = list_files(purpose="batch")
content = file_content("file-abc123")
content isa FileContentSuccess && save_file_content(content, "out.jsonl")
delete_file("file-abc123")
```
