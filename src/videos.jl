# ============================================================================
# OpenAI Videos API — /v1/videos (Sora). ⚠️ Newer, fast-moving surface — verify
# the request/response shapes against the live API before relying on this.
# ============================================================================

"""
    VideoObject

A video-generation job from the Videos API: `id`, plus optional `status` and
`model`; `raw` holds the unparsed JSON response.
"""
@kwdef struct VideoObject
    id::String
    status::Union{String,Nothing} = nothing
    model::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    VideoList

A page of video jobs (raw JSON dicts) from [`list_videos`](@ref); `has_more`
signals that further pages are available.
"""
@kwdef struct VideoList
    data::Vector{Dict{String,Any}}
    has_more::Bool = false
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful create/retrieve result wrapping a [`VideoObject`](@ref)."
@kwdef struct VideoSuccess <: LLMRequestResponse; response::VideoObject; end
"Successful [`list_videos`](@ref) result wrapping a [`VideoList`](@ref)."
@kwdef struct VideoListSuccess <: LLMRequestResponse; response::VideoList; end
"Successful [`video_content`](@ref) result; `content` holds the raw video bytes."
@kwdef struct VideoContentSuccess <: LLMRequestResponse; content::Vector{UInt8}; end
"Videos API error result: HTTP `status` and the raw `response` body."
@kwdef struct VideoFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from a Videos API call (the request never completed)."
@kwdef struct VideoCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

_parse_video(d::AbstractDict) = VideoObject(id=d["id"], status=get(d, "status", nothing), model=get(d, "model", nothing), raw=Dict{String,Any}(d))
_vid_err(e) = VideoCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
_vid_resp(resp) = resp.status == 200 ?
    VideoSuccess(response=_parse_video(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
    VideoFailure(response=String(resp.body), status=resp.status)

"""    create_video(; prompt, model="sora-2", seconds=nothing, size=nothing, input_reference=nothing, service=OPENAIServiceEndpoint)"""
function create_video(; prompt::String, model::String="sora-2", seconds::Union{Int,Nothing}=nothing,
    size::Union{String,Nothing}=nothing, input_reference::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :video, "Videos API")
    try
        d = Dict{Symbol,Any}(:model => model, :prompt => prompt)
        !isnothing(seconds) && (d[:seconds] = seconds)
        !isnothing(size) && (d[:size] = size)
        !isnothing(input_reference) && (d[:input_reference] = input_reference)
        _vid_resp(HTTP.post(_api_base_url(service) * VIDEOS_PATH, body=JSON.json(d), headers=auth_header(service); status_exception=false))
    catch e
        _vid_err(e)
    end
end

"""    retrieve_video(id; service=OPENAIServiceEndpoint)"""
function retrieve_video(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :video, "Videos API")
    try
        _vid_resp(HTTP.get(_api_base_url(service) * VIDEOS_PATH * "/" * id, headers=auth_header(service); status_exception=false))
    catch e
        _vid_err(e)
    end
end

"""    list_videos(; limit=nothing, after=nothing, service=OPENAIServiceEndpoint)"""
function list_videos(; limit::Union{Int,Nothing}=nothing, after::Union{String,Nothing}=nothing, service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :video, "Videos API")
    try
        url = _api_base_url(service) * VIDEOS_PATH
        params = String[]
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = HTTP.get(url, headers=auth_header(service); status_exception=false)
        resp.status == 200 || return VideoFailure(response=String(resp.body), status=resp.status)
        data = JSON.parse(resp.body; dicttype=Dict{String,Any})
        VideoListSuccess(response=VideoList(data=Vector{Dict{String,Any}}(get(data, "data", [])), has_more=get(data, "has_more", false), raw=data))
    catch e
        _vid_err(e)
    end
end

"""    video_content(id; service=OPENAIServiceEndpoint)  → VideoContentSuccess (raw bytes)"""
function video_content(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :video, "Videos API")
    try
        resp = HTTP.get(_api_base_url(service) * VIDEOS_PATH * "/" * id * "/content", headers=auth_header(service); status_exception=false)
        resp.status == 200 ? VideoContentSuccess(content=Vector{UInt8}(resp.body)) :
            VideoFailure(response=String(resp.body), status=resp.status)
    catch e
        _vid_err(e)
    end
end
