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

"Successful create/retrieve/update result wrapping a [`ConversationObject`](@ref); also the result of [`delete_conversation_item`](@ref), which the service answers with the updated conversation."
@kwdef struct ConversationSuccess <: LLMRequestResponse; response::ConversationObject; end
"Successful [`list_conversation_items`](@ref) / [`add_conversation_items`](@ref) result wrapping a [`ConversationItemList`](@ref)."
@kwdef struct ConversationItemListSuccess <: LLMRequestResponse; response::ConversationItemList; end
"Successful result wrapping a single [`ConversationItem`](@ref)."
@kwdef struct ConversationItemSuccess <: LLMRequestResponse; response::ConversationItem; end
"Successful [`delete_conversation`](@ref) result: the service confirmed (`deleted` is always `true`) the removal of `id`."
@kwdef struct ConversationDeleteSuccess <: LLMRequestResponse; id::String; deleted::Bool; end
"Conversations API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct ConversationFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Conversations API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct ConversationCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

_parse_conversation(d::AbstractDict) = ConversationObject(id=d["id"], created_at=get(d, "created_at", nothing),
    metadata=get(d, "metadata", nothing), raw=Dict{String,Any}(d))
_parse_conv_item(d::AbstractDict) = ConversationItem(id=get(d, "id", ""), type=get(d, "type", nothing), raw=Dict{String,Any}(d))

function _conv_http(method::String, url::String, service, cfg::RequestConfig, remaining::Float64; body::Union{String,Nothing}=nothing)
    headers = auth_header(service)
    isnothing(body) ? _http(method, url, headers; cfg, remaining) :
        _http(method, url, headers, body; cfg, remaining)
end

# ─── Requests ─────────────────────────────────────────────────────────────────

"""
    create_conversation(; items=nothing, metadata=nothing, service=OPENAIServiceEndpoint)

Create a conversation. `items` is an optional vector of input items (e.g. `InputMessage`).
Returns `ConversationSuccess`, `ConversationFailure`, or `ConversationCallError`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
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
            _failure(ConversationFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(ConversationCallError, e)
    end
end

"""
    retrieve_conversation(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function retrieve_conversation(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("GET", _api_base_url(service) * CONVERSATIONS_PATH * "/" * _uripart(id), service, cfg, _remaining_s(cfg, t0))
        resp.status == 200 ? ConversationSuccess(response=_parse_conversation(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(ConversationFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(ConversationCallError, e)
    end
end

"""
    update_conversation(id, metadata; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function update_conversation(id::String, metadata::AbstractDict; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("POST", _api_base_url(service) * CONVERSATIONS_PATH * "/" * _uripart(id), service, cfg, _remaining_s(cfg, t0);
            body=JSON.json(Dict{Symbol,Any}(:metadata => metadata)))
        resp.status == 200 ? ConversationSuccess(response=_parse_conversation(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(ConversationFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(ConversationCallError, e)
    end
end

"""
    delete_conversation(id; service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function delete_conversation(id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("DELETE", _api_base_url(service) * CONVERSATIONS_PATH * "/" * _uripart(id), service, cfg, _remaining_s(cfg, t0))
        if resp.status == 200
            d = JSON.parse(resp.body; dicttype=Dict{String,Any})
            ConversationDeleteSuccess(id=get(d, "id", id), deleted=_confirm_deleted(d))
        else
            _failure(ConversationFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(ConversationCallError, e)
    end
end

"""
    add_conversation_items(conversation_id, items; service=OPENAIServiceEndpoint)

Append input items to a conversation. Returns `ConversationItemListSuccess`.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function add_conversation_items(conv_id::String, items::Vector; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("POST", _api_base_url(service) * CONVERSATIONS_PATH * "/" * _uripart(conv_id) * "/items", service, cfg, _remaining_s(cfg, t0);
            body=JSON.json(Dict{Symbol,Any}(:items => items)))
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            its = ConversationItem[_parse_conv_item(i) for i in get(data, "data", [])]
            ConversationItemListSuccess(response=ConversationItemList(data=its, has_more=get(data, "has_more", false),
                first_id=get(data, "first_id", nothing), last_id=get(data, "last_id", nothing), raw=data))
        else
            _failure(ConversationFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(ConversationCallError, e)
    end
end

"""
    list_conversation_items(conversation_id; limit=nothing, order=nothing, after=nothing, service=OPENAIServiceEndpoint)

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function list_conversation_items(conv_id::String; limit::Union{Int,Nothing}=nothing,
    order::Union{String,Nothing}=nothing, after::Union{String,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        url = _api_base_url(service) * CONVERSATIONS_PATH * "/" * _uripart(conv_id) * "/items"
        params = String[]
        !isnothing(limit) && push!(params, "limit=$(_uripart(limit))")
        !isnothing(order) && push!(params, "order=$(_uripart(order))")
        !isnothing(after) && push!(params, "after=$(_uripart(after))")
        !isempty(params) && (url *= "?" * join(params, "&"))
        resp = _conv_http("GET", url, service, cfg, _remaining_s(cfg, t0))
        if resp.status == 200
            data = JSON.parse(resp.body; dicttype=Dict{String,Any})
            its = ConversationItem[_parse_conv_item(i) for i in get(data, "data", [])]
            ConversationItemListSuccess(response=ConversationItemList(data=its, has_more=get(data, "has_more", false),
                first_id=get(data, "first_id", nothing), last_id=get(data, "last_id", nothing), raw=data))
        else
            _failure(ConversationFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(ConversationCallError, e)
    end
end

"""
    delete_conversation_item(conversation_id, item_id; service=OPENAIServiceEndpoint)

Delete one item. The service answers with the updated conversation, so a success is a
`ConversationSuccess` wrapping that [`ConversationObject`](@ref).

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function delete_conversation_item(conv_id::String, item_id::String; service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :conversations, "Conversations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _conv_http("DELETE", _api_base_url(service) * CONVERSATIONS_PATH * "/" * _uripart(conv_id) * "/items/" * _uripart(item_id), service, cfg, _remaining_s(cfg, t0))
        resp.status == 200 ? ConversationSuccess(response=_parse_conversation(JSON.parse(resp.body; dicttype=Dict{String,Any}))) :
            _failure(ConversationFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(ConversationCallError, e)
    end
end
