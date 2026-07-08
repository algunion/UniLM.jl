# [Videos API](@id videos_api)

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
