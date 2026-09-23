# ============================================================================
# OpenAI Moderations API — free policy classification of text and images.
# ============================================================================

"""
    ModerationResult

One classification result: `flagged` (any policy violation), `categories`
(per-category booleans), and `category_scores` (per-category confidence);
`raw` holds the unparsed JSON result.
"""
@kwdef struct ModerationResult
    flagged::Bool
    categories::Dict{String,Any} = Dict{String,Any}()
    category_scores::Dict{String,Any} = Dict{String,Any}()
    raw::Dict{String,Any} = Dict{String,Any}()
end

"""
    ModerationResponse

A moderation response: `results`, one [`ModerationResult`](@ref) per input, and
the `model` used; `raw` holds the unparsed JSON response.
"""
@kwdef struct ModerationResponse
    results::Vector{ModerationResult}
    model::String = ""
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful [`moderate`](@ref) result wrapping a [`ModerationResponse`](@ref)."
@kwdef struct ModerationSuccess <: LLMRequestResponse; response::ModerationResponse; end
"Moderations API error result: HTTP `status`, the raw `response` body, and the `request_id` the service sent (`x-request-id`/`request-id` header), if any."
@kwdef struct ModerationFailure <: LLMRequestResponse; response::String; status::Int; request_id::Union{String,Nothing} = nothing; end
"Moderations API call that produced no usable reply (transport failure, timeout, or a 200 that could not be decoded); `cause` is the underlying exception — a [`UniLMTimeout`](@ref) for a timeout."
@kwdef struct ModerationCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; cause::Union{Nothing,Exception} = nothing; end

"""
    is_flagged(r) -> Bool

True if any moderation result is flagged. Works on `ModerationResult`/`ModerationResponse`/`ModerationSuccess`.

A call that did not succeed has no verdict, so [`ModerationFailure`](@ref) and
[`ModerationCallError`](@ref) **throw** an `ArgumentError` rather than answering
`false`: in the usual `is_flagged(moderate(text)) && reject()` shape, a `false`
from a failed call would wave unmoderated content straight through. Check
`issuccess` first, or handle the throw, and decide explicitly what an
unavailable verdict should mean.
"""
is_flagged(m::ModerationResult) = m.flagged
is_flagged(r::ModerationResponse) = any(is_flagged, r.results)
is_flagged(r::ModerationSuccess) = is_flagged(r.response)
is_flagged(r::ModerationFailure) = throw(ArgumentError(
    "moderation call failed with HTTP $(r.status) — no verdict; inspect the ModerationFailure result"))
is_flagged(r::ModerationCallError) = throw(ArgumentError(
    "moderation call did not complete ($(r.error)) — no verdict; inspect the ModerationCallError result"))

"""The `flagged` verdict of one result row. A row without it is a malformed
response, not a clean verdict, so it fails the call instead of defaulting to
`false` — `moderate` turns the throw into a `ModerationCallError`."""
_moderation_verdict(row::AbstractDict)::Bool = haskey(row, "flagged") ? row["flagged"] :
    throw(ArgumentError("moderation result row has no \"flagged\" field"))

# One verdict row per submitted input: an array of strings is that many inputs, while
# a single string or an array of multi-modal parts (text + image) is one.
_moderation_inputs(input)::Int =
    input isa AbstractVector && all(x -> x isa AbstractString, input) ? length(input) : 1

"""The verdict rows of a 200. Fewer or more rows than inputs — or none at all —
cannot be matched to what was submitted, so it fails the call instead of letting a
missing row read as `flagged == false`."""
function _moderation_rows(d::AbstractDict, inputs::Int)::AbstractVector
    rows = get(d, "results", nothing)
    rows isa AbstractVector || throw(ArgumentError("moderation response has no \"results\" array"))
    length(rows) == inputs || throw(ArgumentError(
        "moderation response has $(length(rows)) result rows for $inputs input(s)"))
    rows
end

"""
    moderate(input; model="omni-moderation-latest", service=OPENAIServiceEndpoint)

Classify `input` (a `String`, a vector of strings, or a vector of multi-modal content
parts) for policy violations (free). Returns `ModerationSuccess`, `ModerationFailure`, or
`ModerationCallError`. A 200 whose `results` array does not hold exactly one row per input
(one row for a string or a parts vector, one per string of a string vector) is a
`ModerationCallError`: a verdict that cannot be matched to its input is no verdict.

Pass `config::Union{Nothing,RequestConfig}` to override the timeout budget for this call (a single bounded attempt; `max_attempts` does not apply).
"""
function moderate(input; model::String="omni-moderation-latest", service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing)
    validate_capability(service, :moderation, "Moderations API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        body = JSON.json(Dict{Symbol,Any}(:model => model, :input => input))
        resp = _http("POST", _api_base_url(service) * MODERATIONS_PATH, auth_header(service),
            body; cfg, remaining=_remaining_s(cfg, t0))
        if resp.status == 200
            d = JSON.parse(resp.body; dicttype=Dict{String,Any})
            results = ModerationResult[
                ModerationResult(flagged=_moderation_verdict(r),
                    categories=Dict{String,Any}(get(r, "categories", Dict{String,Any}())),
                    category_scores=Dict{String,Any}(get(r, "category_scores", Dict{String,Any}())),
                    raw=Dict{String,Any}(r))
                for r in _moderation_rows(d, _moderation_inputs(input))]
            ModerationSuccess(response=ModerationResponse(results=results, model=get(d, "model", model), raw=d))
        else
            _failure(ModerationFailure, resp)
        end
    catch e
        e isa InterruptException && rethrow()
        _callerr(ModerationCallError, e)
    end
end
