# [Uploads API](@id uploads_api)

Resumable, multi-part uploads for large files — beyond the single-request limit
of the [Files API](@ref files_api). Create an upload, add its parts, then
complete it to receive a [`FileObject`](@ref). OpenAI only.

## Objects

```@docs
UploadObject
UploadPartObject
```

## Result Types

```@docs
UploadSuccess
UploadPartSuccess
UploadFailure
UploadCallError
```

## Request Functions

```@docs
create_upload
add_upload_part
complete_upload
cancel_upload
```

## Usage

```julia
# Start a resumable upload for a large file
up = create_upload(filename="big.jsonl", purpose="batch",
                   bytes=filesize("big.jsonl"), mime_type="application/jsonl")

# Add one or more parts, collecting their ids
part = add_upload_part(up.response.id, read("big.jsonl"))
part_ids = [part.response.id]

# Complete to finalize (yields a FileObject), or cancel to abort
done = complete_upload(up.response.id, part_ids)
done isa UploadSuccess && println("File: ", done.response.file.id)
# cancel_upload(up.response.id)
```
