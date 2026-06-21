# ============================================================================
# OpenAI Moderations API — free policy classification of text and images.
# ============================================================================

@kwdef struct ModerationResult
    flagged::Bool
    categories::Dict{String,Any} = Dict{String,Any}()
    category_scores::Dict{String,Any} = Dict{String,Any}()
    raw::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct ModerationResponse
    results::Vector{ModerationResult}
    model::String = ""
    raw::Dict{String,Any} = Dict{String,Any}()
end

@kwdef struct ModerationSuccess <: LLMRequestResponse; response::ModerationResponse; end
@kwdef struct ModerationFailure <: LLMRequestResponse; response::String; status::Int; end
@kwdef struct ModerationCallError <: LLMRequestResponse; error::String; status::Union{Int,Nothing} = nothing; end

"""
    is_flagged(r) -> Bool

True if any moderation result is flagged. Works on `ModerationResult`/`ModerationResponse`/`ModerationSuccess`.
"""
is_flagged(m::ModerationResult) = m.flagged
is_flagged(r::ModerationResponse) = any(is_flagged, r.results)
is_flagged(r::ModerationSuccess) = is_flagged(r.response)
is_flagged(::ModerationFailure) = false
is_flagged(::ModerationCallError) = false

"""
    moderate(input; model="omni-moderation-latest", service=OPENAIServiceEndpoint)

Classify `input` (a `String`, or a vector of content parts) for policy violations (free).
Returns `ModerationSuccess`, `ModerationFailure`, or `ModerationCallError`.
"""
function moderate(input; model::String="omni-moderation-latest", service::ServiceEndpointSpec=OPENAIServiceEndpoint)
    validate_capability(service, :moderation, "Moderations API")
    try
        body = JSON.json(Dict{Symbol,Any}(:model => model, :input => input))
        resp = HTTP.post(_api_base_url(service) * MODERATIONS_PATH, body=body, headers=auth_header(service); status_exception=false)
        if resp.status == 200
            d = JSON.parse(resp.body; dicttype=Dict{String,Any})
            results = ModerationResult[
                ModerationResult(flagged=get(r, "flagged", false),
                    categories=Dict{String,Any}(get(r, "categories", Dict{String,Any}())),
                    category_scores=Dict{String,Any}(get(r, "category_scores", Dict{String,Any}())),
                    raw=Dict{String,Any}(r))
                for r in get(d, "results", [])]
            ModerationSuccess(response=ModerationResponse(results=results, model=get(d, "model", model), raw=d))
        else
            ModerationFailure(response=String(resp.body), status=resp.status)
        end
    catch e
        ModerationCallError(error=string(e), status=(hasproperty(e, :status) ? e.status : nothing))
    end
end
