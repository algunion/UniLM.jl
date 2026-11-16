# [Files API](@id files_api)

Upload, list, retrieve, and delete files, and download their content, through each provider's own
Files API: OpenAI's (`OPENAIServiceEndpoint`, the default), Anthropic's (`ANTHROPICServiceEndpoint`)
and Gemini's (`GEMINIServiceEndpoint`). The verbs and their results are the same on all three; what
differs is below. OpenAI's files feed the Responses `file_search` and code-interpreter tools, the
Batch API, and fine-tuning. Referring to an uploaded file inside a message (an Anthropic `file`
source, a Gemini `fileData` part, a Chat `file_id`) is not available yet.

| | OpenAI | Anthropic | Gemini |
| :--- | :--- | :--- | :--- |
| `upload_file` | `upload_file(path, purpose)`, multipart | `upload_file(path; service)`, multipart; no purpose | `upload_file(path; service)`, Gemini's resumable upload; no purpose |
| `expires_after` (seconds) | 3600 to 2592000 (`expires_after`) | 3600 to 7776000 (`expires_in_seconds`) | refused: Gemini stores a file for 48 hours |
| `list_files` | `purpose`, `limit`, `after` (a file id), `order` | `limit`, `after` (the `next_page` cursor); `purpose` and `order` refused | `limit` (`pageSize`), `after` (the `nextPageToken`); `purpose` and `order` refused |
| `file_content` | the bytes | the bytes of a file a tool made; an uploaded file is Anthropic's HTTP 400 | refused before any request: Gemini serves no download of an uploaded file |
| `delete_file` | `object` `"file"` | `object` `"file_deleted"` | an empty reply: `id` is the file requested |
| `poll_file` | until the status is past `"uploaded"` | at once | until the `state` is past `"PROCESSING"` |
| `register_files` | — | — | `gs://` objects (files.register) |

A Gemini file's `id` is its `name` without the `files/` prefix; the verbs take either form. An
Anthropic Files request sends the key, `anthropic-version` and the endpoint's `workspace_id`, and
neither its `betas` nor its `user_profile_id`, which the Files reference does not name. A reply
without a field the provider's reference requires is a [`FileCallError`](@ref), never a result
with the field made up: a page without OpenAI's `data` or `has_more`, an OpenAI file whose `filename`
or `purpose` is null, a delete reply without the deleted file's id.

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
poll_file
register_files
```

## Usage

```julia
# OpenAI: upload a file for use with file_search / batch / fine-tuning, kept one day
result = upload_file("data.jsonl", "batch"; expires_after=86_400)
result isa FileSuccess && println("Uploaded: ", result.response.id)

# List, download, delete
files = list_files(purpose="batch", order="desc")
content = file_content("file-abc123")
content isa FileContentSuccess && save_file_content(content, "out.jsonl")
delete_file("file-abc123")

# Anthropic: no purpose
claude = upload_file("report.pdf"; service=ANTHROPICServiceEndpoint)

# Gemini: a large video may still be processing when the upload returns
video = upload_file("clip.mp4"; service=GEMINIServiceEndpoint)
ready = poll_file(video.response.id; service=GEMINIServiceEndpoint)
ready isa FileSuccess && println(ready.response.status, " ", ready.response.raw["uri"])
```
