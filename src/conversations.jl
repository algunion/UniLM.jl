# ============================================================================
# OpenAI Conversations API — https://platform.openai.com/docs/api-reference/conversations
# Durable, server-side conversation state. The returned conversation id feeds
# `Respond(conversation=...)` for multi-turn Responses without resending history.
# ============================================================================

# ─── Parsed objects ───────────────────────────────────────────────────────────

"""
    ConversationObject

A durable, server-side conversation: `id`, optional `created_at` and `metadata`;
`raw` holds the unparsed JSON response.
"""
@kwdef struct ConversationObject
    id::String
    created_at::Union{Int,Nothing} = nothing
    metadata::Union{Dict{String,Any},Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    conversation_id(c::ConversationObject) -> String

Return the `id` of a [`ConversationObject`](@ref).
"""
conversation_id(c::ConversationObject) = c.id

"""
    ConversationItem

A single item in a conversation: `id` and optional `type`; `raw` holds the
unparsed JSON response.
"""
@kwdef struct ConversationItem
    id::String
    type::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    ConversationItemList

A page of [`ConversationItem`](@ref)s from [`list_conversation_items`](@ref);
`has_more` signals further pages, and `first_id`/`last_id` bound the page.
"""
@kwdef struct ConversationItemList
    data::Vector{ConversationItem}
    has_more::Bool = false
    first_id::Union{String,Nothing} = nothing
    last_id::Union{String,Nothing} = nothing
    raw::Dict{String,Any} = Dict{String,Any}()
end

# ─── Result types ─────────────────────────────────────────────────────────────

"Successful create/retrieve/update result wrapping a [`ConversationObject`](@ref)."
@kwdef struct ConversationSuccess <: LLMRequestResponse; response::ConversationObject; end
"Successful [`list_conversation_items`](@ref) / [`add_conversation_items`](@ref) result wrapping a [`ConversationItemList`](@ref)."
@kwdef struct ConversationItemListSuccess <: LLMRequestResponse; response::ConversationItemList; end
"Successful result wrapping a single [`ConversationItem`](@ref)."
@kwdef struct ConversationItemSuccess <: LLMRequestResponse; response::ConversationItem; end
"Successful [`delete_conversation`](@ref) / [`delete_conversation_item`](@ref) result; `deleted` confirms removal of `id`."
@kwdef struct ConversationDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Conversations API error result: HTTP `status` and the raw `response` body."
@kwdef struct ConversationFailure <: LLMRequestResponse; response::String; status::Int; end
"Local/transport error from a Conversations API call (the request never completed)."
@kwdef struct ConversationCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

_parse_conversation(d::AbstractDict) = ConversationObject(id=d["id"], created_at=get(d, "created_at", nothing),
    metadata=get(d, "metadata", nothing), raw=Dict{String,Any}(d))
_parse_conv_item(d::AbstractDict) = ConversationItem(id=get(d, "id", ""), type=get(d, "type", nothing), raw=Dict{String,Any}(d))

function _conv_http(method::String, url::String, service, cfg::RequestConfig, remaining::Float64; body::Union{String,Nothing}=nothing)
    headers = auth_header(service)
    isnothing(body) ? _http(method, url, headers; cfg, remaining) :
        _http(method, url, headers, body; cfg, remaining)
end

_conv_err(e) = ConversationCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))

# ─── Requests ─────────────────────────────────────────────────────────────────

"""
    create_conversation(; items=nothing, metadata=nothing, service=OPENAIServiceEndpoint)

Create a conversation. `items` is an optional vector of input items (e.g. `InputMessage`).
Returns `ConversationSuccess`, `ConversationFailure`, or `ConversationCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function create_conversation(; items::Union{Vector,Nothing}=nothing, metadata::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        d = Dict{Symbol,Any}()
        !isnothing(items) && (d[:items] = items)
        !isnothing(metadata) && (d[:metadata] = metadata)
        resp = _conv_http("POST", _api_base_url(service) * CONVERSATIONS_PATH, service, cfg, _remaining_s(cfg, t0); body=JSON.json(d))
        resp.status == 200 ? ConversationSuccess(response=_parse_conversation(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            ConversationFailure(response=String(resp.body), status=resp.status)
    catch e
        e isa InterruptException && rethrow()
        _conv_err(e)
    end
end

"""
    retrieve_conversation(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function retrieve_conversation(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("GET", _api_base_url(service) * CONVERSATIONS_PATH * "/" * id, service, cfg, _remaining_s(cfg, t0))
        resp.status == 200 ? ConversationSuccess(response=_parse_conversation(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            ConversationFailure(response=String(resp.body), status=resp.status)
    catch e
        e isa InterruptException && rethrow()
        _conv_err(e)
    end
end

"""
    update_conversation(id, metadata; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function update_conversation(id::String, metadata::AbstractDict; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("POST", _api_base_url(service) * CONVERSATIONS_PATH * "/" * id, service, cfg, _remaining_s(cfg, t0);
            body=JSON.json(Dict{Symbol,Any}(:metadata => metadata)))
        resp.status == 200 ? ConversationSuccess(response=_parse_conversation(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            ConversationFailure(response=String(resp.body), status=resp.status)
    catch e
        e isa InterruptException && rethrow()
        _conv_err(e)
    end
end

"""
    delete_conversation(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function delete_conversation(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("DELETE", _api_base_url(service) * CONVERSATIONS_PATH * "/" * id, service, cfg, _remaining_s(cfg, t0))
        if resp.status == 200
            d = JSON.parse(resp.body; dicttype=Dict{String,Any})
            ConversationDeleteSuccess(id=get(d, "id", id), deleted=get(d, "deleted", false))
        else
            ConversationFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        e isa InterruptException && rethrow()
        _conv_err(e)
    end
end

"""
    add_conversation_items(conversation_id, items; service=OPENAIServiceEndpoint)

Append input items to a conversation. Returns `ConversationItemListSuccess`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function add_conversation_items(conv_id::String, items::Vector; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("POST", _api_base_url(service) * CONVERSATIONS_PATH * "/" * conv_id * "/items", service, cfg, _remaining_s(cfg, t0);
            body=JSON.json(Dict{Symbol,Any}(:items => items)))
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            its = ConversationItem[_parse_conv_item(i) for i in get(data, "data", [])]
            ConversationItemListSuccess(response=ConversationItemList(data=its, has_more=get(data, "has_more", false),
                first_id=get(data, "first_id", nothing), last_id=get(data, "last_id", nothing), raw=data))
        else
            ConversationFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        e isa InterruptException && rethrow()
        _conv_err(e)
    end
end

"""
    list_conversation_items(conversation_id; limit=nothing, order=nothing, after=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function list_conversation_items(conv_id::String; limit::Union{Int,Nothing}=nothing,
    order::Union{String,Nothing}=nothing, after::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * CONVERSATIONS_PATH * "/" * conv_id * "/items"
        params = String[]
        !isnothing(limit) && push!(params, "limit=$limit")
        !isnothing(order) && push!(params, "order=$order")
        !isnothing(after) && push!(params, "after=$after")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = _conv_http("GET", url, service, cfg, _remaining_s(cfg, t0))
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            its = ConversationItem[_parse_conv_item(i) for i in get(data, "data", [])]
            ConversationItemListSuccess(response=ConversationItemList(data=its, has_more=get(data, "has_more", false),
                first_id=get(data, "first_id", nothing), last_id=get(data, "last_id", nothing), raw=data))
        else
            ConversationFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        e isa InterruptException && rethrow()
        _conv_err(e)
    end
end

"""
    delete_conversation_item(conversation_id, item_id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout/retry budget for this call.
"""
function delete_conversation_item(conv_id::String, item_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("DELETE", _api_base_url(service) * CONVERSATIONS_PATH * "/" * conv_id * "/items/" * item_id, service, cfg, _remaining_s(cfg, t0))
        if resp.status == 200
            d = JSON.parse(resp.body; dicttype=Dict{String,Any})
            ConversationDeleteSuccess(id=get(d, "id", item_id), deleted=get(d, "deleted", false))
        else
            ConversationFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        e isa InterruptException && rethrow()
        _conv_err(e)
    end
end
