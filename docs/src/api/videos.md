# [Videos API](@id videos_api)

!!! warning "Shut down on September 24, 2026"
    OpenAI shuts the Videos API and the `sora-2` models (`sora-2`, `sora-2-pro`, and
    their dated snapshots) down on September 24, 2026, with no replacement listed.
    From that date the wrappers on this page return [`VideoFailure`](@ref) (the
    provider's error response) instead of jobs. See
    [OpenAI's deprecation schedule](https://developers.openai.com/api/docs/deprecations).

Create, retrieve, and list Sora video-generation jobs, and download their
rendered content. Generation is asynchronous — create a job, poll it with
[`retrieve_video`](@ref) until it is ready, then fetch the file with
[`video_content`](@ref). OpenAI only.

## Parsed Objects

```@docs
VideoObject
VideoList
```

## Result Types

```@docs
VideoSuccess
VideoListSuccess
VideoContentSuccess
VideoFailure
VideoCallError
```

## Request Functions

```@docs
create_video
retrieve_video
list_videos
video_content
```

## Usage

```julia
# Create a Sora video-generation job
result = create_video(prompt="A cat surfing a wave", model="sora-2")
result isa VideoSuccess && println("Job: ", result.response.id)

# Poll status, list jobs, download the finished video
retrieve_video("video_abc123")
list_videos(limit=10)
content = video_content("video_abc123")
content isa VideoContentSuccess && write("cat.mp4", content.content)
```
