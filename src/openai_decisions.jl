# ============================================================================
# OpenAI's Decisions API — the OpenAI wrapper of the System One surface.
#
# POST /v1/decisions answers predicate, choice and score questions about an
# `input` (https://developers.openai.com/api/docs/guides/decisions; a public beta
# whose only model is gpt-6-luna). Its wire is not TypeSafe's: questions are an
# array of typed objects whose `name` each answer echoes, answers come back as an
# array in question order, any question may be refused, and probabilities and
# confidences are rounded to two decimals. This file maps the generic surface
# onto that wire and refuses, before any I/O, what the wire cannot carry.
# ============================================================================

"""
    OpenAIDecisionsProtocol <: SystemOneProtocol

OpenAI's Decisions API (a public beta) as a System One backend, and the protocol
of [`OPENAIServiceEndpoint`](@ref): `ask(state, questions...; service =
OPENAIServiceEndpoint)`, a [`with_system_one`](@ref)`(OPENAIServiceEndpoint)`
scope and the natural-language verbs send `POST https://api.openai.com/v1/decisions`
with `OPENAI_API_KEY`, naming `"gpt-6-luna"` — the only model the API serves —
when the call names none.

The request:

- `state` is the `input`: a string as it is; a JSON value (object, array, tuple,
  `NamedTuple`) as its JSON text; an [`ImageAttachment`](@ref), or a `Vector` of
  strings and images, as one user message of `input_text` and `input_image`
  parts (base64 data URLs), in order. Images go there and nowhere else, at most
  128 of them; the input is at most 10,485,760 characters.
- each question is one entry of `questions`, in order, carrying its name: a
  [`NoulQuestion`](@ref) a `predicate`; a [`ChoiceQuestion`](@ref) a `choice` of
  2 to 255 options, each option's name its `value` and a string description its
  `description` (none for `nothing`); a [`ScoreQuestion`](@ref) a `score` of 2
  to 10 levels, each level its `label`; a [`RawQuestion`](@ref) verbatim, with
  its name added as `"name"` when it has none. At most 200 questions.
- provider options are keywords of [`ask`](@ref): `prompt_cache_key`, a `String`
  of at most 64 characters, and `safety_identifier`, at most 128.

What the wire has no form for is an `ArgumentError` before any I/O, naming
OpenAI's rule and the fix: a question without string `instructions`; a Noul's
`yes`/`no` criteria; a description or level that is not a string; a
[`BoxQuestion`](@ref); an image outside the state; any other option.

The reply: answers are matched to the questions by name and must come in the
order asked. A `predicate` reads as a [`NoulAnswer`](@ref), a `choice` as a
[`ChoiceAnswer`](@ref), a `score` as a [`ScoreAnswer`](@ref) whose `legend` maps
each level to its label, and a `refusal` — the API may decline any question — as
a [`RefusalAnswer`](@ref). Probabilities and confidences arrive rounded to two
decimals, so a distribution over `n` outcomes may sum to 1 within `0.005n`.
`usage` counts input, cached and output tokens; `request_id` is the
`x-request-id` header; a non-2xx reply is a [`SystemOneFailure`](@ref) whose
`error_type` is the error's `code`, else its `type`. Recordings are filed under
`/openai-decisions/v1/decisions`. [`estimated_cost`](@ref) bills what OpenAI
bills: every input token, cached ones included, at the model's input rate, twice
that for a request of more than 272K input tokens, and no output.

Public extension API (not exported).
"""
struct OpenAIDecisionsProtocol <: SystemOneProtocol end

system_one_protocol(::Type{OPENAIServiceEndpoint}) = OpenAIDecisionsProtocol()

const _DECISIONS_PATH = "/v1/decisions"
const _DECISIONS_MODEL = "gpt-6-luna"

_system_one_default_model(::OpenAIDecisionsProtocol, service)::String = _DECISIONS_MODEL
_system_one_namespace(::OpenAIDecisionsProtocol)::String = "openai-decisions"
_system_one_request_id(::OpenAIDecisionsProtocol, resp::HTTP.Response)::Union{Nothing,String} =
    _decisions_request_id(resp)
# Every probability arrives rounded to two decimals, off by at most 0.005 from the value
# the model computed: `n` of them sum to 1 within 0.005n, and the chosen option — the
# most probable before rounding — may read up to 0.01 below another (one rounded down by
# 0.005, the other up). A score, rounded too, lies within 0.005 of its range. One value
# serves the three checks: 0.005n, never below the 0.01 of a swap, and 1e-9 for the
# binary error of summing decimals.
_probability_tolerance(::OpenAIDecisionsProtocol, n::Int)::Float64 = max(0.005n, 0.01) + 1e-9
_allows_refusal(::OpenAIDecisionsProtocol)::Bool = true
_allows_raw(::OpenAIDecisionsProtocol)::Bool = true

# ─── Encoding ────────────────────────────────────────────────────────────────

# The limits of the published request schema (`DecisionRequest` in OpenAI's OpenAPI
# description); lengths are in characters, as JSON Schema counts them.
const _DECISIONS_MAX_QUESTIONS = 200
const _DECISIONS_CHOICES = 2:255
const _DECISIONS_LEVELS = 2:10
const _DECISIONS_MAX_IMAGES = 128
const _DECISIONS_MAX_INPUT = 10_485_760
# The provider options, each a string of at most this many characters.
const _DECISIONS_OPTIONS = (prompt_cache_key = 64, safety_identifier = 128)

_decisions_refuse(name::String, why::String) = throw(ArgumentError(
    "OpenAI's Decisions API cannot carry the question $(repr(name)): $why"))

const _DECISIONS_IMAGE_FORMS = "an ImageAttachment, or a Vector of strings and ImageAttachments, which is sent as " *
                               "one user message of text and image parts in order"

# Every question carries instructions, as a string (the `instructions` OpenAI requires).
function _decisions_instructions(name::String, instructions)::String
    instructions isa AbstractString && return String(instructions)
    _decisions_refuse(name, isnothing(instructions) ?
        "every question needs `instructions`, a string. Give them as the first argument, as in " *
        "`choice(\"Which team should handle this?\", options)`" :
        "its `instructions` must be a string, not $(typeof(instructions)). Write them as text" *
        (instructions isa ImageAttachment ? "; an image belongs in the state ($(_DECISIONS_IMAGE_FORMS))" : ""))
end

function _decisions_head(type::String, name::String, instructions)::JSON.Object{String,Any}
    o = JSON.Object{String,Any}()
    o["type"] = type
    o["name"] = name
    o["instructions"] = _decisions_instructions(name, instructions)
    o
end

function _decisions_question(name::String, q::NoulQuestion)::JSON.Object{String,Any}
    c = q.criteria
    if !isnothing(c) && !(isnothing(c.yes) && isnothing(c.no))
        _decisions_refuse(name,
            "a predicate has no true/false criteria. Fold what `yes` and `no` describe into its instructions, " *
            "as in `noul(\"Is the customer frustrated (impatient or upset, not merely neutral)?\")`")
    end
    _decisions_head("predicate", name, q.instructions)
end

function _decisions_question(name::String, q::ChoiceQuestion)::JSON.Object{String,Any}
    o = _decisions_head("choice", name, q.instructions)
    n = length(q.criteria)
    n in _DECISIONS_CHOICES || _decisions_refuse(name,
        "a choice offers $(first(_DECISIONS_CHOICES)) to $(last(_DECISIONS_CHOICES)) options, and it has $n. " *
        (n < first(_DECISIONS_CHOICES) ? "Add one, such as \"other\"" : "Split it, or narrow the options"))
    o["choices"] = Any[_decisions_choice(name, value, description) for (value, description) in q.criteria]
    o
end

function _decisions_choice(name::String, value::String, description)::JSON.Object{String,Any}
    c = JSON.Object{String,Any}()
    c["value"] = value
    isnothing(description) && return c
    description isa AbstractString || _decisions_refuse(name,
        "the description of its option $(repr(value)) must be a string or `nothing`, not $(typeof(description)). " *
        "Write it as text")
    c["description"] = String(description)
    c
end

function _decisions_question(name::String, q::ScoreQuestion)::JSON.Object{String,Any}
    o = _decisions_head("score", name, q.instructions)
    n = length(q.criteria)
    n in _DECISIONS_LEVELS || _decisions_refuse(name,
        "a score has $(first(_DECISIONS_LEVELS)) to $(last(_DECISIONS_LEVELS)) levels, and it has $n. " *
        (n < first(_DECISIONS_LEVELS) ? "Add a level" : "Merge levels into a coarser rubric"))
    o["levels"] = Any[_decisions_level(name, i - 1, level) for (i, level) in enumerate(q.criteria)]
    o
end

function _decisions_level(name::String, index::Int, level)::JSON.Object{String,Any}
    level isa AbstractString || _decisions_refuse(name,
        "its level $index must be a string, the level's label, not $(typeof(level)). Write it as text")
    l = JSON.Object{String,Any}()
    l["label"] = String(level)
    l
end

_decisions_question(name::String, ::BoxQuestion) = _decisions_refuse(name,
    "it is a Box question, and the API answers predicate, choice and score questions only. Send it to a " *
    "backend that answers Box questions")

# Sent as it is, named: every answer is matched to its question by the name it echoes.
function _decisions_question(name::String, q::RawQuestion)::JSON.Object{String,Any}
    isempty(_collect_images!(ImageAttachment[], q.fields)) || _decisions_refuse(name,
        "it holds an image, and images go in the state only: $(_DECISIONS_IMAGE_FORMS)")
    given = get(q.fields, "name", nothing)
    isnothing(given) || given == name || _decisions_refuse(name,
        "its \"name\" field is $(repr(given)), and the answer is matched to the question by the name it echoes. " *
        "Drop the field, or name the question $(repr(given))")
    o = JSON.Object{String,Any}()
    for (k, v) in q.fields
        o[k] = v
    end
    o["name"] = name
    o
end

_decisions_question(name::String, q::SystemOneQuestion) = _decisions_refuse(name,
    "it is a $(nameof(typeof(q))), and the API answers predicate, choice and score questions; a RawQuestion " *
    "is sent verbatim")

# The `input`: text, or one user message of text and image parts. The API takes text or
# messages, so a JSON state is sent as its JSON text, which is what the model reads.
_decisions_input(state::AbstractString) = _decisions_text(String(state), "the state")
_decisions_input(state::ImageAttachment) = _decisions_message(Any[state])
_decisions_input(state::AbstractVector) =
    all(x -> x isa Union{AbstractString,ImageAttachment}, state) && any(x -> x isa ImageAttachment, state) ?
        _decisions_message(state) : _decisions_json_input(state)
_decisions_input(state) = _decisions_json_input(state)

function _decisions_json_input(state)::String
    isempty(_collect_images!(ImageAttachment[], state)) || throw(ArgumentError(
        "OpenAI's Decisions API takes images in the state only as $(_DECISIONS_IMAGE_FORMS); this " *
        "$(nameof(typeof(state))) holds one nested in JSON. Pass the text and the images as such a Vector"))
    _decisions_text(_system_one_body(state), "the state's JSON text")
end

function _decisions_text(text::String, what::String)::String
    length(text) <= _DECISIONS_MAX_INPUT || throw(ArgumentError(
        "OpenAI's Decisions API takes text of at most $(_DECISIONS_MAX_INPUT) characters; $what has " *
        "$(length(text)). Shorten it"))
    text
end

function _decisions_message(items::AbstractVector)::Vector{Any}
    n = count(x -> x isa ImageAttachment, items)
    n <= _DECISIONS_MAX_IMAGES || throw(ArgumentError(
        "OpenAI's Decisions API takes at most $(_DECISIONS_MAX_IMAGES) images per request; the state has $n. " *
        "Split it into several requests"))
    parts = Any[]
    for x in items
        part = JSON.Object{String,Any}()
        if x isa ImageAttachment
            part["type"] = "input_image"
            part["image_url"] = string("data:", x.mime, ";base64,", base64encode(x.data))
        else
            part["type"] = "input_text"
            part["text"] = _decisions_text(String(x), "a text part of the state")
        end
        push!(parts, part)
    end
    message = JSON.Object{String,Any}()
    message["role"] = "user"
    message["content"] = parts
    Any[message]
end

# The options as the body writes them: in one order whatever the caller's, so the same
# request always has the same bytes, and the same recording key.
function _decisions_options(options::NamedTuple)::Vector{Pair{String,String}}
    accepted() = join(("$k (a String of at most $n characters)" for (k, n) in pairs(_DECISIONS_OPTIONS)), " and ")
    for (k, v) in pairs(options)
        haskey(_DECISIONS_OPTIONS, k) || throw(ArgumentError(
            "OpenAI's Decisions API accepts the options $(accepted()); got `$k`"))
        v isa AbstractString && length(v) <= _DECISIONS_OPTIONS[k] || throw(ArgumentError(
            "OpenAI's Decisions API takes `$k` as a String of at most $(_DECISIONS_OPTIONS[k]) characters; got " *
            (v isa AbstractString ? "$(length(v)) characters" : "a $(typeof(v))")))
    end
    Pair{String,String}[String(k) => String(options[k]) for k in keys(_DECISIONS_OPTIONS) if haskey(options, k)]
end

function encode_system_one(p::OpenAIDecisionsProtocol, service, request::SystemOneRequest)::SystemOneWire
    _check_request(p, service, request)
    n = length(request.questions)
    n <= _DECISIONS_MAX_QUESTIONS || throw(ArgumentError(
        "OpenAI's Decisions API takes at most $(_DECISIONS_MAX_QUESTIONS) questions per request; this one has $n. " *
        "Split it into several requests"))
    options = _decisions_options(request.options)
    questions = Any[_decisions_question(name, q) for (name, q) in request.questions]
    body = JSON.Object{String,Any}()
    body["model"] = _system_one_model(p, service, request)
    body["input"] = _decisions_input(request.state)
    body["questions"] = questions
    for (k, v) in options
        body[k] = v
    end
    # The options are fields of the body, so the body alone keys a recording.
    SystemOneWire("POST", _api_base_url(service) * _DECISIONS_PATH, _DECISIONS_PATH, _system_one_body(body))
end

# ─── Decoding ────────────────────────────────────────────────────────────────
# As permissive on read as TypeSafe's decoder (unknown fields ignored, an unknown answer
# type kept), as strict about a required field that is missing or of the wrong JSON type.

_decisions_request_id(resp::HTTP.Response)::Union{Nothing,String} =
    (v = HTTP.header(resp, "x-request-id", ""); isempty(v) ? nothing : String(v))

function _decisions_list(d::AbstractDict, key::String, name::String)::AbstractVector
    v = _answer_field(d, key, name)
    v isa AbstractVector || throw(ArgumentError(
        "System One answer $(repr(name)) field \"$key\" is not an array (got $(typeof(v)))"))
    v
end

function _decisions_entry(e, name::String)::AbstractDict
    e isa AbstractDict || throw(ArgumentError(
        "System One answer $(repr(name)) has a probability entry that is not an object (got $(typeof(e)))"))
    e
end

# A choice value as the request named the option. The API types a boolean value apart
# from the string with the same text; only strings are sent, and a boolean reads as
# "true" or "false".
_decisions_value(v::AbstractString, ::String)::String = String(v)
_decisions_value(v::Bool, ::String)::String = string(v)
_decisions_value(v, name::String) = throw(ArgumentError(
    "System One answer $(repr(name)) has a choice value that is neither a string nor a boolean (got $(typeof(v)))"))

function _decisions_choices(d::AbstractDict, name::String)::Dict{String,Float64}
    out = Dict{String,Float64}()
    for e in _decisions_list(d, "probabilities", name)
        entry = _decisions_entry(e, name)
        value = _decisions_value(_answer_field(entry, "value", name), name)
        haskey(out, value) && throw(ArgumentError(
            "System One answer $(repr(name)) gives the option $(repr(value)) two probabilities"))
        out[value] = _probability(_answer_field(entry, "probability", name), name)
    end
    out
end

# A score's probabilities by level, and the level labels the reply names them by.
function _decisions_levels(d::AbstractDict, name::String)::Tuple{Dict{Int,Any},Dict{Int,Float64}}
    legend = Dict{Int,Any}()
    probabilities = Dict{Int,Float64}()
    for e in _decisions_list(d, "probabilities", name)
        entry = _decisions_entry(e, name)
        v = _answer_field(entry, "value", name)
        v isa Integer && !(v isa Bool) && 0 <= v <= typemax(Int) || throw(ArgumentError(
            "System One score answer $(repr(name)) has the level $(repr(v)); levels are 0-based integers"))
        level = Int(v)
        haskey(probabilities, level) && throw(ArgumentError(
            "System One score answer $(repr(name)) gives level $(level) twice"))
        legend[level] = _answer_string(entry, "label", name)
        probabilities[level] = _probability(_answer_field(entry, "probability", name), name)
    end
    legend, probabilities
end

function _decode_decision(name::String, d::AbstractDict)::SystemOneAnswer
    raw = Dict{String,Any}(string(k) => v for (k, v) in d)
    kind = _answer_string(d, "type", name)
    if kind == "predicate"
        NoulAnswer(_answer_number(d, "probability", name), raw)
    elseif kind == "choice"
        ChoiceAnswer(_decisions_value(_answer_field(d, "choice", name), name), _answer_number(d, "confidence", name),
                     _decisions_choices(d, name), raw)
    elseif kind == "score"
        legend, probabilities = _decisions_levels(d, name)
        ScoreAnswer(_answer_number(d, "score", name), _answer_number(d, "confidence", name), legend, probabilities, raw)
    elseif kind == "refusal"
        RefusalAnswer(raw)
    else
        UnknownAnswer(kind, raw)
    end
end

# Usage is accounting: a count that is missing reads as 0, one that is present but no
# token count reads as 0 with one warning, and neither discards answers that were paid for.
function _decisions_usage(d::AbstractDict)::TokenUsage
    haskey(d, "usage") || return TokenUsage()
    u = d["usage"]
    at(path) = _token_count(foldl((v, k) -> v isa AbstractDict ? get(v, k, 0) : v, path; init=u))
    counts = (at(("input_tokens",)), at(("output_tokens",)), at(("input_tokens_details", "cached_tokens")),
              at(("output_tokens_details", "reasoning_tokens")))
    any(isnothing, counts) &&
        @warn "System One usage holds a value that is not a non-negative whole token count; it reads as 0" usage = u
    i, o, cached, reasoning = map(c -> something(c, 0), counts)
    TokenUsage(prompt_tokens=i, completion_tokens=o, total_tokens=i + o, cached_tokens=cached,
               reasoning_tokens=reasoning)
end

# The decoded reply and the order its answers came in.
function _decode_decisions(body::AbstractString, request_id::Union{Nothing,String},
                           service::Union{Nothing,ServiceEndpointSpec})::Tuple{SystemOneResponse,Vector{String}}
    parsed = JSON.parse(body; dicttype=Dict{String,Any})
    parsed isa AbstractDict || throw(ArgumentError("the Decisions response body is not a JSON object"))
    raw = Dict{String,Any}(string(k) => v for (k, v) in parsed)
    list = get(raw, "answers", nothing)
    list isa AbstractVector || throw(ArgumentError("the Decisions response carries no \"answers\" array"))
    answers = Dict{String,SystemOneAnswer}()
    order = String[]
    for (i, a) in enumerate(list)
        a isa AbstractDict || throw(ArgumentError("Decisions answer $i is not a JSON object"))
        name = get(a, "name", nothing)
        name isa AbstractString || throw(ArgumentError(
            "Decisions answer $i has the name $(repr(name)), not the name of a question asked"))
        haskey(answers, name) && throw(ArgumentError("the Decisions response answers $(repr(name)) twice"))
        answers[String(name)] = _decode_decision(String(name), a)
        push!(order, String(name))
    end
    model = get(raw, "model", nothing)
    model isa AbstractString || throw(ArgumentError(
        "the Decisions response carries no \"model\" string naming the model that answered"))
    SystemOneResponse(String(model), answers, _decisions_usage(raw), request_id, raw, service), order
end

function decode_system_one(p::OpenAIDecisionsProtocol, service, request::SystemOneRequest,
                           response::HTTP.Response)::Union{SystemOneSuccess,SystemOneFailure}
    response.status == 200 || return _decisions_failure(response)
    decoded, order = _decode_decisions(String(response.body), _decisions_request_id(response), service)
    _check_answers(p, request, decoded)
    # Every name matched; the API also answers in question order, and one out of place is
    # a reply that does not follow its own contract.
    asked = String[name for (name, _) in request.questions]
    order == asked || throw(ArgumentError(
        "the Decisions response answers in the order $(join(map(repr, order), ", ")), not in the order asked"))
    SystemOneSuccess(decoded)
end

# An error body is `{"error": {"message", "type", "param", "code"}}`; `code` is the finer
# kind (`model_not_found`, `array_above_max_length`) and `type` the coarser one.
function _decisions_failure(resp::HTTP.Response)::SystemOneFailure
    body = String(resp.body)
    parsed = _typesafe_parse_body(body)
    err = parsed isa AbstractDict ? get(parsed, "error", nothing) : nothing
    field(key) = (v = err isa AbstractDict ? get(err, key, nothing) : nothing; v isa AbstractString ? String(v) : nothing)
    SystemOneFailure(response=body, status=resp.status, request_id=_decisions_request_id(resp),
                     error_type=something(field("code"), field("type"), Some(nothing)),
                     message=something(field("message"), _truncate_body(body)),
                     retry_after=_typesafe_retry_after(resp))
end

# ─── Cost ────────────────────────────────────────────────────────────────────

# A Decisions call is billed for its input tokens only — the guide's "Pricing and
# availability": "no cache-read, cache-write, or output-token charges" — so a cached
# token costs the input rate, and a request of more than 272K input tokens costs twice
# that rate for all of them (gpt-6-luna's long-context pricing). A regional-processing
# premium is not visible in a response, and is not added.
const _DECISIONS_LONG_CONTEXT = 272_000

function _system_one_cost(::OpenAIDecisionsProtocol, result::SystemOneSuccess, model::Union{Nothing,String},
                          pricing::AbstractDict{String,PriceRow})::Float64
    rates = _price_row(pricing, something(model, result.response.model))
    isnothing(rates) && return 0.0
    n = token_usage(result).prompt_tokens
    n * rates.input * (n > _DECISIONS_LONG_CONTEXT ? 2 : 1)
end
