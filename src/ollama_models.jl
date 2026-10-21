# ============================================================================
# Ollama model management (https://docs.ollama.com/api): the installed models
# (`/api/tags`), one model's metadata (`/api/show`), the loaded ones (`/api/ps`),
# loading through an empty `/api/chat` or `/api/embed` request (by the model's kind)
# and unloading through an empty `/api/chat` request, pulling from the registry
# (`/api/pull`, a newline-delimited JSON progress stream), and deleting (`/api/delete`).
# ============================================================================

"""
    OllamaModel

One model installed on an Ollama server, as [`list_models`](@ref) reports it
(`/api/tags`): `name` (e.g. `"gemma4:e4b"`), `size` on disk in bytes, content `digest`,
`modified_at` (the server's RFC 3339 text), and from its details `family`,
`parameter_size` (e.g. `"8.0B"`), `quantization` (e.g. `"Q4_K_M"`) and
`context_length`, the longest context the model supports. `runner` — the runner the
server's listing names for the model (`:mlx`, `:llamacpp`, `:ggml`), the one an
[`OllamaEndpoint`](@ref)'s `runner` must name to reach it, which can differ from the one
that serves it once loaded: [`OllamaRunningModel`](@ref)`.runner` (`/api/ps`) reports that
— and `format` (`:gguf`, `:safetensors`), as Ollama 0.40 and later report them; `nothing`
when not reported. `capabilities` holds what
the server reports, e.g. `:completion`, `:tools`, `:insert`, `:vision`, `:embedding`,
`:thinking`, `:image` and `:audio`. A detail the server does not report, or reports empty
(`""`), is `nothing`; `raw` holds the unparsed entry.
"""
struct OllamaModel
    name::String
    size::Int
    digest::String
    modified_at::String
    family::Union{String,Nothing}
    parameter_size::Union{String,Nothing}
    quantization::Union{String,Nothing}
    context_length::Union{Int,Nothing}
    runner::Union{Symbol,Nothing}
    format::Union{Symbol,Nothing}
    capabilities::Vector{Symbol}
    raw::Dict{String,Any}
end

"""
    OllamaModelInfo

What [`model_info`](@ref) reports about one model (`/api/show`): the `name` asked
for, its `capabilities` (as for [`OllamaModel`](@ref)), `context_length` (the longest
context it supports, from its metadata), the `thinking_levels` it accepts — `[false,
true]` for a model that thinks or not, names such as `"low"`, `"medium"`, `"high"`
for one with levels, empty for a model that cannot think — and `thinking_default`.
`parameters` are its Modelfile defaults with numbers parsed (`"top_k" => 64`); a
parameter set more than once, such as `stop`, collects a `Vector{String}`. `family`,
`parameter_size` and `quantization` come from its details, `nothing` when not reported
or empty (`""`). `runner` — the runner `/api/show`'s details name (`:mlx`, `:llamacpp`,
`:ggml`; `nothing` when none — e.g. `embeddinggemma:latest` on Ollama 0.40.6), which can
differ from [`OllamaModel`](@ref)`.runner` (`/api/tags`) and from the runner that serves it
once loaded ([`OllamaRunningModel`](@ref)`.runner`, `/api/ps`) — and `format` (`:gguf`,
`:safetensors`), as Ollama 0.40 and later report them; `nothing` when not reported or
empty (`""`). `raw` holds the whole reply (template, license, `model_info`, …).
"""
struct OllamaModelInfo
    name::String
    capabilities::Vector{Symbol}
    context_length::Union{Int,Nothing}
    thinking_levels::Vector{Union{Bool,String}}
    thinking_default::Union{Bool,String,Nothing}
    parameters::Dict{String,Any}
    family::Union{String,Nothing}
    parameter_size::Union{String,Nothing}
    quantization::Union{String,Nothing}
    runner::Union{Symbol,Nothing}
    format::Union{Symbol,Nothing}
    raw::Dict{String,Any}
end

"""
    OllamaRunningModel

A model loaded in an Ollama server's memory, as [`running_models`](@ref) reports it
(`/api/ps`): `name`, `size` (bytes of memory it takes), `size_vram` (the part in GPU
memory), the `context_length` it was loaded with, and `expires_at`, when it unloads
unless used again (RFC 3339 text). `runner` — the runner serving it now (`:mlx`,
`:llamacpp`, `:ggml`), as Ollama 0.40 and later report it; `nothing` when not reported
or empty (`""`). `raw` holds the unparsed entry.
"""
struct OllamaRunningModel
    name::String
    size::Int
    size_vram::Int
    context_length::Union{Int,Nothing}
    expires_at::String
    runner::Union{Symbol,Nothing}
    raw::Dict{String,Any}
end

"""
    OllamaPullProgress

One progress report of [`pull_model`](@ref): `status` (`"pulling manifest"`,
`"pulling <digest prefix>"`, `"verifying sha256 digest"`, `"writing manifest"`,
`"success"`) and, while a layer downloads, its `digest` and the `completed` and
`total` bytes (`nothing` otherwise). `completed` is an `Int` whenever `total` is:
Ollama leaves a zero count out of its report, and it is read back as `0`.
"""
struct OllamaPullProgress
    status::String
    digest::Union{String,Nothing}
    completed::Union{Int,Nothing}
    total::Union{Int,Nothing}
end

"Successful Ollama model-management result: `response` holds what the verb reports (`nothing` for [`pull_model`](@ref), [`load_model`](@ref), [`unload_model`](@ref) and [`delete_model`](@ref))."
@kwdef struct OllamaSuccess{T} <: LLMRequestResponse; response::T; end
"Ollama error result: HTTP `status`, the raw `response` body (Ollama's `{\"error\": …}`), and the `request_id` the server sent (`x-request-id`/`request-id` header), if any."
@kwdef struct OllamaFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Ollama call that produced no usable reply (no server, a timeout, a cancel, a malformed reply, or a failure reported on the pull stream); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout, a [`UniLMCancelled`](@ref) for a cancel."
@kwdef struct OllamaCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

_gigabytes(bytes::Int)::String = string(round(bytes / 1e9; digits=1), " GB")

Base.show(io::IO, m::OllamaModel) =
    print(io, "OllamaModel(", repr(m.name), ", ", _gigabytes(m.size), isnothing(m.runner) ? "" : ", $(m.runner)",
          ", ", m.capabilities, ")")
Base.show(io::IO, m::OllamaRunningModel) =
    print(io, "OllamaRunningModel(", repr(m.name), ", ", _gigabytes(m.size), ", ", _gigabytes(m.size_vram),
          " in VRAM", isnothing(m.context_length) ? "" : ", context $(m.context_length)", ")")

# ─── Reply decoding ─────────────────────────────────────────────────────────
# A required field must be there with its JSON type; an optional one may be absent or
# null. Anything else is a malformed reply — the verb's call error, never a guess.

function _opt_field(d::AbstractDict, key::String, ::Type{T})::Union{T,Nothing} where {T}
    v = get(d, key, nothing)
    isnothing(v) || v isa T || throw(ArgumentError("Ollama reply field $(repr(key)) is a $(typeof(v)), not a $T"))
    v
end

function _req_field(d::AbstractDict, key::String, ::Type{T})::T where {T}
    v = _opt_field(d, key, T)
    isnothing(v) && throw(ArgumentError("Ollama reply has no $(repr(key))"))
    v
end

# A detail the server writes as "" (MLX models leave family and sizes empty) is not reported.
_opt_text(d::AbstractDict, key::String)::Union{String,Nothing} =
    (v = _opt_field(d, key, String); isnothing(v) || isempty(v) ? nothing : v)

# A name the server reports as text reads as a Symbol, as capabilities do ("mlx" → :mlx).
_opt_symbol(d::AbstractDict, key::String)::Union{Symbol,Nothing} =
    (v = _opt_text(d, key); isnothing(v) ? nothing : Symbol(v))
# A model's weight format, in lower case as Ollama names it in its manifests ("gguf" → :gguf).
_opt_format(d::AbstractDict)::Union{Symbol,Nothing} =
    (v = _opt_text(d, "format"); isnothing(v) ? nothing : Symbol(lowercase(v)))

_ollama_details(d::AbstractDict) = something(_opt_field(d, "details", AbstractDict), Dict{String,Any}())

function _ollama_capabilities(d::AbstractDict)::Vector{Symbol}
    caps = something(_opt_field(d, "capabilities", AbstractVector), [])
    all(c -> c isa String, caps) || throw(ArgumentError("Ollama capabilities must be strings"))
    Symbol[Symbol(c) for c in caps]
end

# The `models` array of a /api/tags or /api/ps reply.
function _ollama_entries(d::AbstractDict)::Vector{Dict{String,Any}}
    entries = _req_field(d, "models", AbstractVector)
    all(e -> e isa AbstractDict, entries) || throw(ArgumentError("an Ollama \"models\" entry is not a JSON object"))
    Dict{String,Any}.(entries)
end

function _decode_model(d::Dict{String,Any})::OllamaModel
    det = _ollama_details(d)
    OllamaModel(_req_field(d, "name", String), _req_field(d, "size", Int), _req_field(d, "digest", String),
                _req_field(d, "modified_at", String), _opt_text(det, "family"), _opt_text(det, "parameter_size"),
                _opt_text(det, "quantization_level"), _opt_field(det, "context_length", Int),
                _opt_symbol(det, "runner"), _opt_format(det), _ollama_capabilities(d), d)
end

# /api/ps reports the runner on the entry itself, /api/tags in its details; the details
# are read only when the entry has none.
_decode_running(d::Dict{String,Any})::OllamaRunningModel =
    OllamaRunningModel(_req_field(d, "name", String), _req_field(d, "size", Int), _req_field(d, "size_vram", Int),
                       _opt_field(d, "context_length", Int), _req_field(d, "expires_at", String),
                       @something(_opt_symbol(d, "runner"), _opt_symbol(_ollama_details(d), "runner"), Some(nothing)), d)

function _decode_show(name::String, d::Dict{String,Any})::OllamaModelInfo
    det = _ollama_details(d)
    meta = something(_opt_field(d, "model_info", AbstractDict), Dict{String,Any}())
    arch = _opt_field(meta, "general.architecture", String)
    thinking = something(_opt_field(d, "thinking", AbstractDict), Dict{String,Any}())
    levels = something(_opt_field(thinking, "values", AbstractVector), [])
    all(v -> v isa Union{Bool,String}, levels) ||
        throw(ArgumentError("Ollama thinking levels must be booleans or strings"))
    OllamaModelInfo(name, _ollama_capabilities(d), isnothing(arch) ? nothing : _opt_field(meta, arch * ".context_length", Int),
                    Vector{Union{Bool,String}}(levels), _opt_field(thinking, "default", Union{Bool,String}),
                    _decode_parameters(something(_opt_field(d, "parameters", String), "")),
                    _opt_text(det, "family"), _opt_text(det, "parameter_size"), _opt_text(det, "quantization_level"),
                    _opt_symbol(det, "runner"), _opt_format(det), d)
end

# The `parameters` text of /api/show: one `key value` line per Modelfile default, a
# list parameter (stop) once per element, a string value Go-quoted (`stop "<eos>"`).
function _decode_parameters(text::AbstractString)::Dict{String,Any}
    given = Dict{String,Vector{String}}()
    for line in eachsplit(text, '\n')
        isempty(strip(line)) && continue
        m = match(r"^(\S+)\s+(.*?)\s*$", line)
        isnothing(m) && throw(ArgumentError("unreadable Ollama parameters line $(repr(String(line)))"))
        push!(get!(Vector{String}, given, m[1]), m[2])
    end
    Dict{String,Any}(k => length(v) == 1 ? _parameter_value(only(v)) : _go_unquoted.(v) for (k, v) in given)
end

_go_quoted(s::AbstractString) = length(s) >= 2 && startswith(s, '"') && endswith(s, '"')
_go_unquoted(s::AbstractString)::String = _go_quoted(s) ? unescape_string(chop(s; head=1, tail=1)) : String(s)

function _parameter_value(s::AbstractString)
    _go_quoted(s) && return _go_unquoted(s)
    s in ("true", "false") && return s == "true"
    something(tryparse(Int, s), tryparse(Float64, s), Some(String(s)))
end

# ─── Non-streaming verbs ────────────────────────────────────────────────────

# The model a verb names: an empty name is a local mistake, refused before any I/O.
function _ollama_model_name(name::AbstractString)::String
    isempty(strip(name)) && throw(ArgumentError("an Ollama model name is required (e.g. \"gemma4:e4b\")"))
    String(name)
end

# Every call error renders through the endpoint, so an unreachable server gets the
# `ollama serve` hint.
_ollama_callerr(service::OllamaEndpoint, e) = _callerr(OllamaCallError, e; error=_error_text(service, e))

# One idempotent exchange through the shared retry loop: a 200 is decoded by `decode` (a throw there
# makes the malformed reply a call error); `decode === nothing` for a verb whose 200 carries no body.
# Any other status is the failure. `t0` is the verb's start: a verb that sends more than one request
# passes its own, so `total_deadline` bounds the verb rather than each request.
function _ollama_call(decode::Union{Function,Nothing}, service::OllamaEndpoint, method::String, path::String, body,
                      config::Union{Nothing,RequestConfig}, cancel::Union{Nothing,CancelToken}; t0::UInt64=time_ns())
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel)
    try
        resp = _http_with_retries(cfg, t0, method, service.base_url * path, auth_header(service), body; cancel=tok)
        resp.status == 200 || return _failure(OllamaFailure, resp)
        isnothing(decode) && return OllamaSuccess(response=nothing)
        d = JSON.parse(resp.body; dicttype=Dict{String,Any})
        d isa Dict{String,Any} || throw(ArgumentError("the Ollama reply is not a JSON object"))
        OllamaSuccess(response=decode(d))
    catch e
        e isa InterruptException && rethrow()
        _ollama_callerr(service, e)
    end
end

_list_models(service::OllamaEndpoint, config::Union{Nothing,RequestConfig}, cancel::Union{Nothing,CancelToken}) =
    _ollama_call(d -> OllamaModel[_decode_model(e) for e in _ollama_entries(d)], service, "GET", "/api/tags",
                 UInt8[], config, cancel)

"""
    model_info(name; service=OllamaEndpoint(), config=nothing,
                     cancel=nothing)

What an Ollama server knows about the installed model `name` (`POST /api/show`):
an [`OllamaSuccess`](@ref) holding an [`OllamaModelInfo`](@ref) — capabilities, context
length, thinking levels, Modelfile parameters — or an [`OllamaFailure`](@ref) (404 for a
model that is not installed) or an [`OllamaCallError`](@ref). Transient statuses are
retried under `config` (a [`RequestConfig`](@ref)); `cancel` (default: the ambient
[`with_cancel`](@ref) token) ends the call with a [`UniLMCancelled`](@ref) in `cause`. The
endpoint's `runner` goes with it: Ollama shows the variant it would run with that runner,
and answers a model that has none with HTTP 404 `no compatible manifest found for runners:
<runner>` (Ollama 0.40.6). An empty `name` throws `ArgumentError` before any request.

```julia
r = model_info("gemma4:e4b")
issuccess(r) && r.response.thinking_levels   # [false, true]
```
"""
model_info(name::AbstractString; service::OllamaEndpoint=OllamaEndpoint(),
           config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _ollama_show(_ollama_model_name(name), service, config, cancel)

_ollama_show(model::String, service::OllamaEndpoint, config::Union{Nothing,RequestConfig},
             cancel::Union{Nothing,CancelToken}; t0::UInt64=time_ns()) =
    _ollama_call(d -> _decode_show(model, d), service, "POST", "/api/show",
                 JSON.json(_ollama_runner!(Dict{String,Any}("model" => model), service)), config, cancel; t0)

"""
    running_models(; service=OllamaEndpoint(), config=nothing,
                     cancel=nothing)

The models an Ollama server holds in memory now (`GET /api/ps`): an
[`OllamaSuccess`](@ref) holding a `Vector{`[`OllamaRunningModel`](@ref)`}` — empty when
none is loaded — or an [`OllamaFailure`](@ref) or [`OllamaCallError`](@ref). `config` and
`cancel` work as for [`model_info`](@ref).

```julia
r = running_models()
issuccess(r) && foreach(println, r.response)
```
"""
running_models(; service::OllamaEndpoint=OllamaEndpoint(), config::Union{Nothing,RequestConfig}=nothing,
               cancel::Union{Nothing,CancelToken}=nothing) =
    _ollama_call(d -> OllamaRunningModel[_decode_running(e) for e in _ollama_entries(d)], service, "GET", "/api/ps",
                 UInt8[], config, cancel)

# An empty chat request loads the model and answers done_reason "load"; with keep_alive 0
# it unloads the model and answers "unload". Any other answer is not the one asked for.
function _ollama_done(d::Dict{String,Any}, expected::String)::Nothing
    reason = get(d, "done_reason", nothing)
    reason == expected ||
        throw(ArgumentError("unexpected Ollama reply: done_reason $(repr(reason)) where $(repr(expected)) was expected"))
    nothing
end

# A model that completes text loads with an empty chat request; any other (an embedding or a
# decision model) is refused by /api/chat and /api/generate, and loads with an empty
# /api/embed request, which every model answers once it is loaded.
"""
    load_model(name; service=OllamaEndpoint(), config=nothing,
                     cancel=nothing)

Load `name` into memory now, so the first request does not wait for it. Loads `name` the
way its kind allows (read from [`model_info`](@ref)): a chat model with an empty chat
request, an embedding or decision model with an empty embeddings request (EmbeddingGemma 2
with the 8,192-token window its embeddings requests carry); the reply comes once the model
is loaded. The endpoint's `keep_alive` and `runner` go with it, and so does what the
model's own requests carry: a chat model's load the endpoint's [`OllamaOptions`](@ref) and
`shift`, an embedding model's those options, and a decision model's no options at all, as
System One requests carry none. Ollama loads the model with those settings, so load
through the endpoint you will use. On the llama.cpp runner a later request with other
options loads the model again; on the MLX runner a loaded model keeps its context when a
later request asks for more: [`unload_model`](@ref) first to load it with another
`num_ctx`. Returns `OllamaSuccess{Nothing}`, an [`OllamaFailure`](@ref) (404 for a model
that is not installed), or an [`OllamaCallError`](@ref), also for a reply that does not
report the load. `config` and `cancel` work as for [`model_info`](@ref); `total_deadline`
bounds the whole call. An empty `name` throws `ArgumentError` before any request, and so
does an endpoint with `keep_alive=0`: Ollama reads that request as an unload.

```julia
ollama = OllamaEndpoint(num_ctx=32_768, keep_alive=Inf)
load_model("gemma4:e4b"; service=ollama)  # 32k context, stays loaded
```
"""
function load_model(name::AbstractString; service::OllamaEndpoint=OllamaEndpoint(),
                    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    model = _ollama_model_name(name)
    ka = service.keep_alive
    isnothing(ka) || !iszero(ka) || throw(ArgumentError(
        "load_model with keep_alive=0 would unload the model; leave keep_alive unset or give it seconds"))
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel); t0 = time_ns()   # one budget, both requests
    info = _ollama_show(model, service, cfg, tok; t0)
    info isa OllamaSuccess || return info
    caps = info.response.capabilities
    chat = :completion in caps
    body = Dict{String,Any}("model" => model)
    chat ? (body["messages"] = []; body["stream"] = false) : (body["input"] = [])
    isnothing(ka) || (body["keep_alive"] = _ollama_keep_alive(ka))
    # A load sends the options the model's own requests send, since on the llama.cpp runner a request whose
    # options differ from the load's loads the model again (Ollama 0.40.6): an embeddings load those of an
    # embeddings request for the model (EmbeddingGemma 2's window included), and a decision model's none, as
    # System One requests carry none.
    opts = chat ? _ollama_options(service.options) :
           :embedding in caps ? _ollama_embed_options(service, model) : Dict{String,Any}()
    isempty(opts) || (body["options"] = opts)
    chat && !isnothing(service.shift) && (body["shift"] = service.shift)
    _ollama_runner!(body, service)
    chat ? _ollama_call(d -> _ollama_done(d, "load"), service, "POST", OLLAMA_CHAT_PATH, JSON.json(body), cfg, tok; t0) :
           _ollama_call(_ollama_embed_loaded, service, "POST", OLLAMA_EMBED_PATH, JSON.json(body), cfg, tok; t0)
end

# An empty embeddings request answers an empty list once the model is loaded.
function _ollama_embed_loaded(d::Dict{String,Any})::Nothing
    v = get(d, "embeddings", nothing)
    v isa AbstractVector && isempty(v) || throw(ArgumentError(
        "unexpected Ollama reply to an empty embeddings request: $(repr(v)) where [] was expected"))
    nothing
end

"""
    unload_model(name; service=OllamaEndpoint(), config=nothing,
                       cancel=nothing)

Free the memory `name` holds now rather than when its keep-alive runs out (an empty
chat request with `keep_alive` 0, which works for every kind of model; the endpoint's
`runner` goes with it). Returns `OllamaSuccess{Nothing}`, an
[`OllamaFailure`](@ref), or an [`OllamaCallError`](@ref), also for a reply that does not
report the unload. `config` and `cancel` work as for [`model_info`](@ref); an empty
`name` throws `ArgumentError` before any request.

```julia
unload_model("gemma4:e4b")
```
"""
function unload_model(name::AbstractString; service::OllamaEndpoint=OllamaEndpoint(),
                      config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    body = Dict{String,Any}("model" => _ollama_model_name(name), "messages" => [], "stream" => false, "keep_alive" => 0)
    _ollama_runner!(body, service)
    _ollama_call(d -> _ollama_done(d, "unload"), service, "POST", OLLAMA_CHAT_PATH, JSON.json(body), config, cancel)
end

"""
    delete_model(name; service=OllamaEndpoint(), config=nothing, cancel=nothing)

Remove the installed model `name` from an Ollama server's disk (`DELETE /api/delete`). Returns
`OllamaSuccess{Nothing}`, an [`OllamaFailure`](@ref) (404 for a model that is not installed), or an
[`OllamaCallError`](@ref). `config` and `cancel` work as for [`model_info`](@ref). An empty `name`
throws `ArgumentError` before any request. There is no undo: [`pull_model`](@ref) downloads it again.

```julia
delete_model("all-minilm:22m")
```
"""
delete_model(name::AbstractString; service::OllamaEndpoint=OllamaEndpoint(),
             config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _ollama_call(nothing, service, "DELETE", "/api/delete", JSON.json(Dict("model" => _ollama_model_name(name))),
                 config, cancel)

# ─── Pull (newline-delimited JSON progress stream) ──────────────────────────

# One /api/pull line: a progress report, or `nothing` for a line that is none. A line
# `{"error": …}` is the server's failure: the 200 status is already sent when a pull
# fails (an unknown tag), so the failure arrives in-band, and it ends the call.
function _decode_pull_line(line::AbstractString)::Union{OllamaPullProgress,Nothing}
    d = try
        JSON.parse(line; dicttype=Dict{String,Any})
    catch e
        e isa InterruptException && rethrow()
        nothing
    end
    d isa AbstractDict || return nothing
    err = get(d, "error", nothing)
    isnothing(err) || error(err isa AbstractString ? err : JSON.json(err))
    s, digest, completed, total = (get(d, k, nothing) for k in ("status", "digest", "completed", "total"))
    # Ollama omits a zero `completed` (JSON omitempty): a layer that has not started reports only its total.
    isnothing(completed) && total isa Int && (completed = 0)
    s isa String && digest isa Union{String,Nothing} && completed isa Union{Int,Nothing} &&
        total isa Union{Int,Nothing} || return nothing
    OllamaPullProgress(s, digest, completed, total)
end

# The complete lines of `chunk` (layer 1 of the SSE machine; each line is one JSON
# object), reported in order; true at "success". A line that is no report is dropped
# and counted, as the chat stream drops an undecodable payload.
function _pull_lines!(report::Function, carry::IOBuffer, chunk::String, dropped::Ref{Int})::Bool
    for line in _sse_complete_lines!(carry, chunk)
        p = _decode_pull_line(line)
        if isnothing(p)
            Threads.atomic_add!(_SSE_DROPPED_LINES, 1)
            dropped[] += 1
            @debug "Ollama pull: dropped a line that is not a progress report" line = String(line)
            continue
        end
        report(p)
        p.status == "success" && return true
    end
    false
end

"""
    pull_model(name; service=OllamaEndpoint(), progress=nothing,
               config=nothing, cancel=nothing)

Download `name` from the Ollama registry to the server (`POST /api/pull`).
`progress(p::`[`OllamaPullProgress`](@ref)`)` runs once per report the server streams,
in order, on the calling task; the last reports `"success"`. Returns
`OllamaSuccess{Nothing}` once the server reports success, an [`OllamaFailure`](@ref) for
a non-200 reply, or an [`OllamaCallError`](@ref): the server's message for a failure it
reports on the stream (an unknown tag: `"pull model manifest: file does not exist"`), a
stream that ends before success, an exception `progress` threw (in `cause`), a timeout,
or a cancel.

One attempt, never retried. The response headers must arrive within
`min(request_timeout, remaining total_deadline)` of the [`RequestConfig`](@ref); after
that only `stream_idle_timeout` — the longest gap between bytes, time spent in `progress`
excluded — bounds the download, however long it runs. `cancel` (default: the ambient
[`with_cancel`](@ref) token) stops it at once, with a [`UniLMCancelled`](@ref) in `cause`.
The endpoint's `runner` goes with it: by Ollama 0.40.6's source (untested), from a model
published for several runners Ollama pulls the variant for that runner, and a model
published for another one fails with `no compatible manifest found for runners: <runner>`.
An empty `name` throws `ArgumentError` before any request.

```julia
show_progress(p) = isnothing(p.total) ? println(p.status) :
    println(p.status, " ", p.completed, "/", p.total, " bytes")
r = pull_model("gemma4:e2b"; progress=show_progress)
issuccess(r)
```
"""
function pull_model(name::AbstractString; service::OllamaEndpoint=OllamaEndpoint(), progress=nothing,
                    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing)
    model = _ollama_model_name(name)
    cfg = _resolve_config(config); tok = _resolve_cancel(cancel); t0 = time_ns()
    idle = Ref{Union{Nothing,_IdleGuard}}(nothing)    # armed when the response headers arrive
    # Request-phase bound, recorded before it unwinds through HTTP.jl (see
    # `_with_recorded_deadline`), and what `progress` threw, kept for the same reason.
    bound = Ref{Union{Nothing,UniLMTimeout}}(nothing)
    thrown = Ref{Any}(nothing)
    dropped = Ref(0)
    failed = IOBuffer()                               # the body of a non-200 reply
    ctx = HTTP.RequestContext()
    # `progress` runs between reads: its time is not a byte gap, and what it throws ends
    # the call (see `_user_call`).
    function report(p::OllamaPullProgress)
        isnothing(progress) && return
        _enter_user!(idle[])
        try
            progress(p)
        catch e
            e isa InterruptException || (thrown[] = e)
            rethrow()
        finally
            _exit_user!(idle[])
        end
        nothing
    end
    try
        # Uncompressed, so each line arrives as the server wrote it.
        resp = _http_open("POST", service.base_url * "/api/pull", [auth_header(service); "Accept-Encoding" => "identity"];
                          cfg, t0, cancel=tok, context=ctx, decompress=false) do io
            status = _with_recorded_deadline(() -> begin
                    write(io, JSON.json(_ollama_runner!(Dict{String,Any}("model" => model), service)))
                    HTTP.closewrite(io)
                    HTTP.startread(io).status
                end, () -> _abort_request_phase(io, ctx),
                min(_remaining_s(cfg, t0), cfg.request_timeout), :request, bound)
            # From the headers on, only the byte-gap bound runs: a long healthy download
            # is not a failure.
            idle[] = _idle_guard(() -> close(io), cfg.stream_idle_timeout)
            carry = IOBuffer()
            while !iscancelled(tok) && !eof(io)
                chunk = String(readavailable(io))
                _touch!(idle[])
                status == 200 || (write(failed, chunk); continue)
                _pull_lines!(report, carry, chunk, dropped) && return   # success: the model is in place
            end
            iscancelled(tok) && throw(UniLMCancelled(:token, _elapsed_s(t0)))
            status == 200 || return
            # A last line the server never newline-terminated is still a line.
            tail = takestring!(carry)
            !isempty(tail) && _pull_lines!(report, carry, tail * "\n", dropped) && return
            # A guard's close landing between reads ends the loop at a clean EOF.
            breach = _exit_breach(idle[], bound, cfg)
            breach === nothing || throw(breach)
            error("the pull stream ended before success")
        end
        resp.status == 200 ? OllamaSuccess(response=nothing) :
            OllamaFailure(response=takestring!(failed), status=resp.status, request_id=_platform_request_id(resp))
    catch e
        _find_exception(x -> x isa InterruptException, e) !== nothing && rethrow()
        u = thrown[]
        isnothing(u) || return OllamaCallError(error=_error_text(service, u), cause=(u isa Exception ? u : nothing))
        # Cancelled: whatever HTTP.jl surfaced is the abort's echo. Otherwise a typed
        # bound (the byte gap, then the request phase) wins over its teardown noise.
        cause = iscancelled(tok) ? UniLMCancelled(:token, _elapsed_s(t0)) :
            something(_classify_stream_timeout(e, idle[], cfg, t0), bound[],
                      _map_native_timeout(e, cfg, min(_remaining_s(cfg, t0), cfg.request_timeout), t0), Some(e))
        _ollama_callerr(service, cause)
    finally
        _disarm!(idle[])
        _warn_sse_drops(dropped[], model, "model pull")
    end
end
