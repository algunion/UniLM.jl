# ============================================================================
# OpenAI graders — run or validate a grader outside a fine-tuning job, under
# /v1/fine_tuning/alpha/graders (https://developers.openai.com/api/docs/guides/graders).
# Part of the Fine-tuning API: the `:fine_tuning` capability, its failure types.
# ============================================================================

const _GRADERS_PATH = "/v1/fine_tuning/alpha/graders"

"""
    Grader

A grader UniLM types: [`StringCheckGrader`](@ref), [`TextSimilarityGrader`](@ref) or
[`PythonGrader`](@ref). The `score_model`, `label_model` and `multi` graders, whose inputs are
messages or other graders, are `Dict`s in OpenAI's wire form, as in a reinforcement `method`;
a typed grader inside such a `Dict` is sent in its wire form too.
"""
abstract type Grader end

const _STRING_CHECK_OPERATIONS = ("eq", "ne", "like", "ilike")
const _SIMILARITY_METRICS = ("cosine", "fuzzy_match", "bleu", "gleu", "meteor", "rouge_1", "rouge_2", "rouge_3",
                             "rouge_4", "rouge_5", "rouge_l")

# An `ArgumentError` naming the grader, its field and the values OpenAI's reference takes.
_check_grader_value(grader::String, field::String, v::String, values::Tuple) =
    v in values || throw(ArgumentError("$grader takes $field $(join(repr.(values), ", ", " or ")) (got $(repr(v)))"))

"""
    StringCheckGrader(; name, input, reference, operation)

Compares `input` with `reference`, both of which may hold template strings such as
`"{{sample.output_text}}"` and `"{{item.answer}}"`, by `operation`: `"eq"`, `"ne"`, `"like"` or `"ilike"`
(another is an `ArgumentError`).
"""
@kwdef struct StringCheckGrader <: Grader
    name::String
    input::String
    reference::String
    operation::String
    function StringCheckGrader(name, input, reference, operation)
        _check_grader_value("StringCheckGrader", "operation", operation, _STRING_CHECK_OPERATIONS)
        new(name, input, reference, operation)
    end
end

"""
    TextSimilarityGrader(; name, input, reference, evaluation_metric)

Grades `input` by its similarity to `reference` under `evaluation_metric`: `"cosine"`,
`"fuzzy_match"`, `"bleu"`, `"gleu"`, `"meteor"`, `"rouge_1"` to `"rouge_5"` or `"rouge_l"` (another is
an `ArgumentError`).
"""
@kwdef struct TextSimilarityGrader <: Grader
    name::String
    input::String
    reference::String
    evaluation_metric::String
    function TextSimilarityGrader(name, input, reference, evaluation_metric)
        _check_grader_value("TextSimilarityGrader", "evaluation_metric", evaluation_metric, _SIMILARITY_METRICS)
        new(name, input, reference, evaluation_metric)
    end
end

"""
    PythonGrader(; name, source, image_tag=nothing)

Runs the Python `source`, which defines `grade(sample, item)`, in the image `image_tag` when given.
"""
@kwdef struct PythonGrader <: Grader
    name::String
    source::String
    image_tag::Union{String,Nothing} = nothing
end

_grader_type(::StringCheckGrader) = "string_check"
_grader_type(::TextSimilarityGrader) = "text_similarity"
_grader_type(::PythonGrader) = "python"
JSON.lower(g::Grader) = Dict{Symbol,Any}(:type => _grader_type(g),
    (f => getfield(g, f) for f in fieldnames(typeof(g)) if !isnothing(getfield(g, f)))...)

"""
    GraderRun

The result of [`run_grader`](@ref): the `reward` the grader gave the sample; its `metadata` (the
grader's `name` and `type`, its `errors`, `execution_time`, `scores` and the model grader's
`token_usage`); `sub_rewards`, the reward of each grader of a `multi` grader; `raw`, the reply,
with `model_grader_token_usage_per_model`.
"""
@kwdef struct GraderRun
    reward::Float64
    metadata::Dict{String,Any}
    sub_rewards::Dict{String,Any}
    raw::Dict{String,Any} = Dict{String,Any}()
end

"Successful [`run_grader`](@ref) result wrapping a [`GraderRun`](@ref)."
@kwdef struct GraderRunSuccess <: LLMRequestResponse; response::GraderRun; end
"Successful [`validate_grader`](@ref) result: `grader` is the grader as OpenAI read it."
@kwdef struct GraderValidateSuccess <: LLMRequestResponse; grader::Dict{String,Any}; end

function _grader_post(decode::Function, route::String, body::AbstractDict, service, config, cancel)
    validate_capability(service, :fine_tuning, "Fine-tuning API")
    cfg = _resolve_config(config); t0 = time_ns()
    try
        resp = _http("POST", _api_base_url(service) * _GRADERS_PATH * route, auth_header(service), JSON.json(body);
                     cfg, remaining=_remaining_s(cfg, t0), cancel=_resolve_cancel(cancel))
        resp.status == 200 ? decode(JSON.parse(resp.body; dicttype=Dict{String,Any})) : _failure(FineTuningFailure, resp)
    catch e
        e isa InterruptException && rethrow()
        _callerr(FineTuningCallError, e)
    end
end

"""
    run_grader(grader, model_sample; item=nothing, service=OPENAIServiceEndpoint)

Grade `model_sample`, the text the `sample` namespace of the grader's templates reads (its
`output_json` too, when the text is JSON), with `grader`: a [`Grader`](@ref) or a grader `Dict`.
`item`, a `Dict`, fills the `item` namespace. Returns a [`GraderRunSuccess`](@ref); a grader OpenAI
rejects is its 400, a `FineTuningFailure`. A model grader samples its model, and OpenAI reports the
tokens it used in the result.

Takes `config` and `cancel` as [`cancel_fine_tuning_job`](@ref) does: one bounded attempt, and a
cancelled token returns a `FineTuningCallError` whose `cause` is a [`UniLMCancelled`](@ref).
"""
function run_grader(grader::Union{Grader,AbstractDict}, model_sample::String; item::Union{AbstractDict,Nothing}=nothing,
    service::ServiceEndpointSpec=OPENAIServiceEndpoint, config::Union{Nothing,RequestConfig}=nothing,
    cancel::Union{Nothing,CancelToken}=nothing)
    body = Dict{Symbol,Any}(:grader => grader, :model_sample => model_sample)
    isnothing(item) || (body[:item] = item)
    _grader_post("/run", body, service, config, cancel) do d
        GraderRunSuccess(response=GraderRun(reward=Float64(_reply_required(d, "reward", Real, "Graders API")),
            metadata=_reply_required(d, "metadata", Dict{String,Any}, "Graders API"),
            sub_rewards=_reply_required(d, "sub_rewards", Dict{String,Any}, "Graders API"), raw=d))
    end
end

"""
    validate_grader(grader; service=OPENAIServiceEndpoint)

Check `grader`, a [`Grader`](@ref) or a grader `Dict`, without running it: a
[`GraderValidateSuccess`](@ref) holding the grader as OpenAI read it, or OpenAI's 400 for a grader it
rejects, a `FineTuningFailure`. Takes `config` and `cancel` as [`run_grader`](@ref) does.
"""
validate_grader(grader::Union{Grader,AbstractDict}; service::ServiceEndpointSpec=OPENAIServiceEndpoint,
    config::Union{Nothing,RequestConfig}=nothing, cancel::Union{Nothing,CancelToken}=nothing) =
    _grader_post("/validate", Dict(:grader => grader), service, config, cancel) do d
        GraderValidateSuccess(grader=_reply_required(d, "grader", Dict{String,Any}, "Graders API"))
    end
