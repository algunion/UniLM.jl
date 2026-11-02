# ============================================================================
# OpenAI's Decisions API — the OpenAI wrapper of the System One surface.
#
# POST /v1/decisions answers predicate, choice and score questions about an
# `input` (https://developers.openai.com/api/docs/guides/decisions; generally
# available, and its only model is gpt-6-luna). Its wire is not TypeSafe's: questions are an
# array of typed objects whose `name` each answer echoes, answers come back as an
# array in question order, any question may be refused, and probabilities and
# confidences are rounded to two decimals. This file maps the generic surface
# onto that wire and refuses, before any I/O, what the wire cannot carry.
# ============================================================================

"""
    OpenAIDecisionsProtocol <: SystemOneProtocol

OpenAI's Decisions API, generally available, as a System One backend, and the protocol
of [`OPENAIServiceEndpoint`](@ref): `ask(state, questions...; service =
OPENAIServiceEndpoint)`, a [`with_system_one`](@ref)`(OPENAIServiceEndpoint)`
scope and the natural-language verbs send `POST https://api.openai.com/v1/decisions`
with `OPENAI_API_KEY`, naming `"gpt-6-luna"` — the only model the API serves —
when the call names none.

The request:

- `state` is the `input`: a string as it is; a JSON value (object, array, tuple,
  `NamedTuple`) as its JSON text; an [`ImageAttachment`](@ref), or a vector or
  tuple of strings and images, as one user message of `input_text` and `input_image`
  parts (base64 data URLs, each with its `detail` when the image sets one), in
  order. Each text — the input, or each text part of the message — is at most
  10,485,760 characters.
- each question is one entry of `questions`, in order, carrying its name: a
  [`NoulQuestion`](@ref) a `predicate`; a [`ChoiceQuestion`](@ref) a `choice` of
  2 to 255 options, each option's name its `value` and a string description its
  `description` (none for `nothing`); a [`ScoreQuestion`](@ref) a `score` of 2
  to 10 levels, each level its `label`; a [`RawQuestion`](@ref) verbatim, with
  its name added as `"name"` when it has none. At most 400 questions; each
  question's name, string instructions, option names and descriptions and level
  labels, and the model's name, at most 1,048,576 characters.
- a question's `instructions` are a string as it is, or else the 1 to 16,384 parts
  that stand in its place: an image an `input_image` part, a JSON object (a
  `NamedTuple` too) an `input_json` part, and a vector or tuple the parts of its
  items, in order — a string an `input_text` of at most 10,485,760 characters, an
  image an `input_image`, an object or an array of objects an `input_json`. An
  `input_json` part holds no image and at most 32 levels of objects and arrays,
  10,000 values and 256 KiB of compact JSON.
- images go in the state and in the questions' instructions, at most 128 of them
  in all.
- provider options are keywords of [`ask`](@ref): `prompt_cache_key`, a `String`
  of at most 64 characters; `safety_identifier`, at most 128; and
  `prompt_cache_options`, a [`PromptCacheOptions`](@ref) of `mode` and `ttl`, `"30m"` or
  `"12h"` (OpenAI's prompt-caching guide names `"12h"` for the Responses API; the
  Decisions API answered it with HTTP 200). With `mode="explicit"` the request marks no
  cache breakpoint, so OpenAI caches none of it.

What the wire has no form for is an `ArgumentError` before any I/O, naming
OpenAI's rule and the fix: a question without `instructions`, or with
instructions beyond those forms and limits; a Noul's
`yes`/`no` criteria; a description or level that is not a string; a
[`BoxQuestion`](@ref); an image anywhere but the state and the instructions; a [`DocumentAttachment`](@ref)
anywhere (the API takes no files), as on every System One backend; a
`prompt_cache_options` with another `ttl`, `prewarm` or
`comparison_response_id`; any other option.

The reply: answers are matched to the questions by name and must come in the
order asked. A `predicate` reads as a [`NoulAnswer`](@ref), a `choice` as a
[`ChoiceAnswer`](@ref), a `score` as a [`ScoreAnswer`](@ref) whose `legend` maps
each level to its label, and a `refusal` — the API may decline any question — as
a [`RefusalAnswer`](@ref). Probabilities and confidences arrive rounded to two
decimals, so a distribution over `n` outcomes may sum to 1 within `0.005n`.
`usage` counts input, cached, cache-write — the 12-hour ones apart, as on the
Responses API — and output tokens, and holds the service tier when the reply names
one; `request_id` is the
`x-request-id` header; a reply with a status other than 200 is a
[`SystemOneFailure`](@ref) whose `error_type` is the error's `code`, else its
`type`. Recordings are filed under `/openai-decisions/v1/decisions`, keyed
without the three options, which change how a request is served but never what it
is answered, and with each image as the SHA-256 of its base64 text, never its
bytes. [`estimated_cost`](@ref) bills what OpenAI
bills: the input tokens alone, and no output — uncached ones at the model's input
rate, cache reads at its cached-input rate, and cache writes and a request of more
than 272K input tokens at the surcharges OpenAI lists for the model (`gpt-6-luna`:
a write at 1.25 times the input rate, 2.5 times for a 12-hour lifetime, and every
rate twice above 272K).

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
# 0.005, the other up). A score, rounded too, lies within 0.005 of its range. The sum
# allowance grows with `n`; the slack for the choice and the score is one swap, 0.01,
# however many options there are; 1e-9 covers the binary error of summing decimals.
_probability_tolerance(::OpenAIDecisionsProtocol, n::Int)::Float64 = 0.005n + 1e-9
_probability_slack(::OpenAIDecisionsProtocol)::Float64 = 0.01 + 1e-9
_allows_refusal(::OpenAIDecisionsProtocol)::Bool = true
_allows_raw(::OpenAIDecisionsProtocol)::Bool = true
# An `input_image` part takes `detail` — low, high, auto or original, "using the selected
# model's image profile" (DecisionInputImage in OpenAI's OpenAPI description) — sent as set.
_image_detail_refusal(::OpenAIDecisionsProtocol)::Nothing = nothing
# Transport: a hosted service like TypeSafe's — the same bounds, and the same batch width
# for `ask_each` (its rate limits were 30,000 requests a minute for the measured account).
_system_one_default_config(::OpenAIDecisionsProtocol, service)::Union{Nothing,RequestConfig} = _HOSTED_SYSTEM_ONE_CONFIG
_system_one_concurrency(::OpenAIDecisionsProtocol, service)::Int = 16

# ─── Encoding ────────────────────────────────────────────────────────────────

# The limits of the published request schema (`DecisionRequest` in OpenAI's OpenAPI
# description); lengths are in characters, as JSON Schema counts them. The image cap is not in
# the schema: it is the API's own, which answers more images with HTTP 400 ("at most 128 input
# images"), and OpenAI's guide counts a question's images with the input's ("Question images
# share the request's image limits with images in `input`").
const _DECISIONS_MAX_QUESTIONS = 400       # "Ask 1–400 questions about the input"
const _DECISIONS_CHOICES = 2:255
const _DECISIONS_LEVELS = 2:10
const _DECISIONS_MAX_IMAGES = 128
const _DECISIONS_MAX_INPUT = 10_485_760
# An `input_image` part's `image_url`, the image's data URL (DecisionInputImage).
const _DECISIONS_MAX_IMAGE_URL = 1_073_741_824
# The model, and each question's name, string instructions, choice value and description and level label.
const _DECISIONS_MAX_FIELD = 1_048_576
# Instructions as parts (DecisionQuestionInputPart), and an `input_json` part's data: "Limited to 32
# container levels, 10,000 values, and 256 KiB of compact UTF-8 JSON" (DecisionInputJson).
const _DECISIONS_PARTS = 1:16_384
const _DECISIONS_JSON_LEVELS = 32
const _DECISIONS_JSON_VALUES = 10_000
const _DECISIONS_JSON_BYTES = 262_144
# The string options, each of at most this many characters; `prompt_cache_options` follows them.
const _DECISIONS_OPTIONS = (prompt_cache_key = 64, safety_identifier = 128)
const _DECISIONS_ACCEPTED = join(("$k (a String of at most $n characters)" for (k, n) in pairs(_DECISIONS_OPTIONS)), ", ") *
    " and prompt_cache_options (a PromptCacheOptions of mode and ttl \"30m\" or \"12h\")"

_decisions_refuse(name::String, why::String) = throw(ArgumentError(
    "OpenAI's Decisions API cannot carry the question $(repr(name)): $why"))

# A string the schema caps at `_DECISIONS_MAX_FIELD` characters. The refusal names the field, `what`,
# and its length, never the text itself.
function _decisions_field(text::AbstractString, what::String)::String
    n = length(text)
    n <= _DECISIONS_MAX_FIELD || throw(ArgumentError(
        "OpenAI's Decisions API takes at most $(_DECISIONS_MAX_FIELD) characters in $what; it has $n. Shorten it"))
    String(text)
end

const _DECISIONS_IMAGE_FORMS = "an ImageAttachment, or a Vector or Tuple of strings and ImageAttachments, which is " *
                               "sent as one user message of text and image parts in order"

# The fix for guidance that is not text: write it as text, and move an image to the state or the instructions.
_decisions_as_text(x)::String = "Write it as text" *
    (x isa ImageAttachment ? "; an image belongs in the state, as $(_DECISIONS_IMAGE_FORMS), or in the question's " *
                             "instructions" : "")

# Every question carries the `instructions` OpenAI requires: a string as it is, any other entry as the parts that
# stand in its place (DecisionQuestionInputPart) — a vector or tuple the parts of its items, in order, an image or a
# JSON object one part. Each image is checked against `limit` before it is encoded.
function _decisions_instructions(name::String, instructions;
                                 limit::Integer=_DECISIONS_MAX_IMAGE_URL)::Union{String,Vector{Any}}
    instructions isa AbstractString && return _decisions_field(instructions, "the instructions of $(repr(name))")
    isnothing(instructions) && _decisions_refuse(name,
        "every question needs `instructions`, a string or parts of text, images and JSON. Give them as the first " *
        "argument, as in `choice(\"Which team should handle this?\", options)`")
    parts = instructions isa Union{AbstractVector,Tuple} ? instructions : (instructions,)
    n = length(parts)
    n in _DECISIONS_PARTS || _decisions_refuse(name,
        "its instructions are $n parts, and a question takes $(first(_DECISIONS_PARTS)) to $(last(_DECISIONS_PARTS)). " *
        (n < first(_DECISIONS_PARTS) ? "Give it one, or a string" : "Merge them, or split the question"))
    Any[_decisions_part(name, i, x; limit) for (i, x) in enumerate(parts)]
end

# One part of a question's instructions: a string an `input_text`, an image an `input_image`, and a JSON object,
# or an array of objects, an `input_json`.
_decisions_part(name::String, i::Int, x::AbstractString; limit::Integer)::JSON.Object{String,Any} =
    _decisions_text_part(x, "part $i of the instructions of $(repr(name))")
_decisions_part(::String, ::Int, x::ImageAttachment; limit::Integer)::JSON.Object{String,Any} =
    _decisions_image_part(x; limit)
_decisions_part(name::String, i::Int, x::Union{AbstractDict,NamedTuple}; limit::Integer)::JSON.Object{String,Any} =
    _decisions_json_part(name, i, x)
_decisions_part(name::String, i::Int, x::Union{AbstractVector,Tuple}; limit::Integer)::JSON.Object{String,Any} =
    all(v -> v isa Union{AbstractDict,NamedTuple}, x) ? _decisions_json_part(name, i, x) : _decisions_no_part(name, i, x)
_decisions_part(name::String, i::Int, x; limit::Integer) = _decisions_no_part(name, i, x)

_decisions_no_part(name::String, i::Int, x) = _decisions_refuse(name,
    "part $i of its instructions is a $(typeof(x)), and a part is a string, an ImageAttachment, or a JSON object or " *
    "array of objects. Write it as text, or put it in an object")

# An `input_json` part. The levels are counted first, so data that holds itself is refused rather than walked
# forever; then an image in the data, which the API would read as text, never as an image; then the bytes of the
# compact JSON, and the values the data holds — itself, and every object, array and scalar inside it.
function _decisions_json_part(name::String, i::Int, data)::JSON.Object{String,Any}
    what = "part $i of its instructions"
    _nests_deeper(data, _DECISIONS_JSON_LEVELS) && _decisions_refuse(name,
        "$what nests objects and arrays more than $(_DECISIONS_JSON_LEVELS) levels deep, the most an `input_json` " *
        "part takes. Flatten it, or send it as text")
    isempty(_collect_images!(ImageAttachment[], data)) || _decisions_refuse(name,
        "$what holds an image inside JSON, which the API reads as text and fetches no image from. Make the image " *
        "a part of its own, as in `noul([\"Does the photo match the record?\", record, image])`")
    text = _system_one_body(data)
    n = sizeof(text)
    n <= _DECISIONS_JSON_BYTES || _decisions_refuse(name,
        "$what is $n bytes of compact JSON, and an `input_json` part takes at most $(_DECISIONS_JSON_BYTES) " *
        "(256 KiB). Trim it, or send it as text")
    held = _json_values(JSON.parse(text))
    held <= _DECISIONS_JSON_VALUES || _decisions_refuse(name,
        "$what holds $held JSON values, and an `input_json` part takes at most $(_DECISIONS_JSON_VALUES). Trim it, " *
        "or send it as text")
    JSON.Object{String,Any}("type" => "input_json", "data" => data)
end

_json_values(x::AbstractDict)::Int = 1 + sum(_json_values, values(x); init=0)
_json_values(x::AbstractVector)::Int = 1 + sum(_json_values, x; init=0)
_json_values(_)::Int = 1

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
    c["value"] = _decisions_field(value, "an option's name in $(repr(name))")
    isnothing(description) && return c
    description isa AbstractString || _decisions_refuse(name,
        "the description of its option $(repr(value)) must be a string or `nothing`, not $(typeof(description)). " *
        _decisions_as_text(description))
    c["description"] = _decisions_field(description, "the description of the option $(repr(value)) in $(repr(name))")
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
        "its level $index must be a string, the level's label, not $(typeof(level)). $(_decisions_as_text(level))")
    l = JSON.Object{String,Any}()
    l["label"] = _decisions_field(level, "the label of level $index of $(repr(name))")
    l
end

_decisions_question(name::String, ::BoxQuestion) = _decisions_refuse(name,
    "it is a Box question, and the API answers predicate, choice and score questions only. Send it to a " *
    "backend that answers Box questions")

# Sent as it is, named: every answer is matched to its question by the name it echoes.
function _decisions_question(name::String, q::RawQuestion)::JSON.Object{String,Any}
    isempty(_collect_images!(ImageAttachment[], q.fields)) || _decisions_refuse(name,
        "it holds an image, which a RawQuestion, sent as written, cannot carry: an image goes in the state, as " *
        "$(_DECISIONS_IMAGE_FORMS), or in the instructions of a `choice`, `score` or `noul`")
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
# A tuple is taken as a vector is, as every System One backend takes it.
_decisions_input(state::Union{AbstractVector,Tuple}) =
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

# The length of the data URL of `nbytes` bytes of a `mime` image: "data:" (5 characters), the mime type,
# ";base64," (8) and the padded base64 of the bytes, 4 characters for every 3 bytes begun. It follows from
# the byte count, so an image too large for the schema is refused without being encoded.
_decisions_image_url_length(mime::AbstractString, nbytes::Integer)::Int = 13 + ncodeunits(mime) + 4cld(nbytes, 3)

function _check_decisions_image_url(mime::AbstractString, nbytes::Integer;
                                    limit::Integer=_DECISIONS_MAX_IMAGE_URL)::Nothing
    n = _decisions_image_url_length(mime, nbytes)
    n <= limit || throw(ArgumentError(
        "OpenAI's Decisions API takes an image as a data URL of at most $limit characters; this $mime " *
        "image's would have $n. Downscale it, or re-encode it in a more compact format"))
    nothing
end

# One user message of the state's text and image parts, in order. Each image is checked against `limit`
# before it is encoded.
function _decisions_message(items::Union{AbstractVector,Tuple};
                            limit::Integer=_DECISIONS_MAX_IMAGE_URL)::Vector{Any}
    parts = Any[x isa ImageAttachment ? _decisions_image_part(x; limit) : _decisions_text_part(x, "a text part of the state")
                for x in items]
    Any[JSON.Object{String,Any}("role" => "user", "content" => parts)]
end

_decisions_text_part(text::AbstractString, what::String)::JSON.Object{String,Any} =
    JSON.Object{String,Any}("type" => "input_text", "text" => _decisions_text(String(text), what))

# An `input_image` part: the image's data URL, checked against `limit` before it is encoded, and its `detail`
# when it sets one.
function _decisions_image_part(x::ImageAttachment; limit::Integer=_DECISIONS_MAX_IMAGE_URL)::JSON.Object{String,Any}
    _check_decisions_image_url(x.mime, length(x.data); limit)
    part = JSON.Object{String,Any}("type" => "input_image", "image_url" => _data_url(x))
    isnothing(x.detail) || (part["detail"] = x.detail)
    part
end

# The images a request sends, counted before any is encoded: an image, or the images among the items of a vector or
# tuple — in the state, and in each question's instructions.
_decisions_images(::ImageAttachment)::Int = 1
_decisions_images(x::Union{AbstractVector,Tuple})::Int = count(i -> i isa ImageAttachment, x)
_decisions_images(_)::Int = 0
_decisions_question_images(q::Union{ChoiceQuestion,ScoreQuestion,NoulQuestion})::Int = _decisions_images(q.instructions)
_decisions_question_images(::SystemOneQuestion)::Int = 0

# The options as the body writes them: in one order whatever the caller's, so the same
# request always has the same bytes, and the same recording key.
function _decisions_options(options::NamedTuple)::Vector{Pair{String,Any}}
    for (k, v) in pairs(options)
        k === :prompt_cache_options && continue
        haskey(_DECISIONS_OPTIONS, k) || throw(ArgumentError(
            "OpenAI's Decisions API accepts the options $(_DECISIONS_ACCEPTED); got `$k`"))
        v isa AbstractString && length(v) <= _DECISIONS_OPTIONS[k] || throw(ArgumentError(
            "OpenAI's Decisions API takes `$k` as a String of at most $(_DECISIONS_OPTIONS[k]) characters; got " *
            (v isa AbstractString ? "$(length(v)) characters" : "a $(typeof(v))")))
    end
    written = Pair{String,Any}[String(k) => String(options[k]) for k in keys(_DECISIONS_OPTIONS) if haskey(options, k)]
    haskey(options, :prompt_cache_options) &&
        push!(written, "prompt_cache_options" => _decisions_cache_options(options.prompt_cache_options))
    written
end

# The prompt-cache options as the body writes them, `mode` then `ttl`: the Decisions API's
# `prompt_cache_options` object has these two fields only.
function _decisions_cache_options(o)::JSON.Object{String,Any}
    o isa PromptCacheOptions || throw(ArgumentError(
        "OpenAI's Decisions API takes `prompt_cache_options` as a PromptCacheOptions; got a $(typeof(o))"))
    isnothing(o.prewarm) && isnothing(o.comparison_response_id) || throw(ArgumentError(
        "OpenAI's Decisions API takes prompt_cache_options mode and ttl only; prewarm and comparison_response_id " *
        "are fields of the Responses API (Respond)"))
    _check_openai_cache_ttl(o, _OPENAI_CACHE_TTLS.decisions)
    written = JSON.Object{String,Any}()
    isnothing(o.mode) || (written["mode"] = o.mode)
    isnothing(o.ttl) || (written["ttl"] = o.ttl)
    written
end

function encode_system_one(p::OpenAIDecisionsProtocol, service, request::SystemOneRequest)::SystemOneWire
    _check_request(p, service, request)
    n = length(request.questions)
    n <= _DECISIONS_MAX_QUESTIONS || throw(ArgumentError(
        "OpenAI's Decisions API takes at most $(_DECISIONS_MAX_QUESTIONS) questions per request; this one has $n. " *
        "Split it into several requests"))
    options = _decisions_options(request.options)
    for (name, _) in request.questions
        _decisions_field(name, "a question's name")
    end
    images = _decisions_images(request.state) + sum(((_, q),) -> _decisions_question_images(q), request.questions; init=0)
    images <= _DECISIONS_MAX_IMAGES || throw(ArgumentError(
        "OpenAI's Decisions API takes at most $(_DECISIONS_MAX_IMAGES) images per request, the state's and the " *
        "questions' instructions' together; this one has $images. Split it into several requests"))
    questions = Any[_decisions_question(name, q) for (name, q) in request.questions]
    body = JSON.Object{String,Any}()
    body["model"] = _decisions_field(_system_one_model(p, service, request), "the model's name")
    body["input"] = _decisions_input(request.state)
    body["questions"] = questions
    url, keyed = _api_base_url(service) * _DECISIONS_PATH, _system_one_body(body)
    isempty(options) && return SystemOneWire("POST", url, _DECISIONS_PATH, keyed)
    # The options choose a prompt cache, how long it keeps the prompt, and an identity for
    # abuse monitoring: they change how a request is served, never what it is answered, so
    # they are sent as fields of the body and left out of what a recording is keyed on.
    for (k, v) in options
        body[k] = v
    end
    SystemOneWire("POST", url, _DECISIONS_PATH, _system_one_body(body); record_body=keyed)
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
# Cache writes are read as on the Responses API (`input_tokens_details.cache_write_tokens`,
# the 12-hour ones in `cache_write_tokens_details.ttl_12h_tokens`), and the billed tier
# from beside the usage, where a reply names one.
function _decisions_usage(d::AbstractDict)::TokenUsage
    tier = _wire_string(d, "service_tier")
    haskey(d, "usage") || return TokenUsage(service_tier=tier)
    u = d["usage"]
    counts = (_decisions_count(u, ("input_tokens",)), _decisions_count(u, ("output_tokens",)),
              _decisions_count(u, ("input_tokens_details", "cached_tokens")),
              _decisions_count(u, ("output_tokens_details", "reasoning_tokens")),
              _decisions_count(u, ("input_tokens_details", "cache_write_tokens")),
              _decisions_count(u, ("input_tokens_details", "cache_write_tokens_details", "ttl_12h_tokens")))
    any(isnothing, counts) &&
        @warn "System One usage holds a value that is not a non-negative whole token count; it reads as 0" usage = u
    i, o, cached, reasoning, writes, extended = map(c -> something(c, 0), counts)
    TokenUsage(prompt_tokens=i, completion_tokens=o, total_tokens=i + o, cached_tokens=cached,
               reasoning_tokens=reasoning, cache_write_tokens=writes, cache_write_extended_tokens=extended,
               service_tier=tier)
end

# The count at `path` in a usage object: 0 when a step is missing, `nothing` when a step is
# not the object the schema gives it (`input_tokens_details` is an object holding
# `cached_tokens`) or the count is no token count.
function _decisions_count(u, path::Tuple{Vararg{String}})::Union{Nothing,Int}
    v = u
    for k in path
        v isa AbstractDict || return nothing
        haskey(v, k) || return 0
        v = v[k]
    end
    _token_count(v)
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
                     retry_after=_retry_after(resp))
end

# ─── Cost ────────────────────────────────────────────────────────────────────

# A Decisions call is billed for its input tokens only — the guide's "Pricing and
# availability": "Prompt cache hits receive discounted input pricing; cache writes are
# charged. Cache-read and cache-write rates match the standard `gpt-6-luna` prompt caching
# rates. Output tokens are not charged", and "These rates apply to /v1/decisions" rather
# than the processing-tier prices of other gpt-6-luna requests — so OpenAI's billing
# (`_billed_cost`) is given the input tokens with their cache reads and writes, with no
# output and no tier: it bills cache reads at the row's cached rate, writes at 1.25× the
# input rate (2.5× for 12 hours), and a request over 272K input tokens at twice each rate
# ("long-context input pricing multipliers apply"). A regional-processing premium is not
# visible in a response, and is not added.

function _system_one_cost(::OpenAIDecisionsProtocol, result::SystemOneSuccess, model::Union{Nothing,String},
                          pricing::AbstractDict{String,PriceRow})::Float64
    priced = something(model, result.response.model)
    rates = _price_row(pricing, priced)
    isnothing(rates) && return 0.0
    u = token_usage(result)
    billed = TokenUsage(prompt_tokens=u.prompt_tokens, total_tokens=u.prompt_tokens, cached_tokens=u.cached_tokens,
                        cache_write_tokens=u.cache_write_tokens, cache_write_extended_tokens=u.cache_write_extended_tokens)
    _billed_cost(OPENAIServiceEndpoint, priced, billed, rates)
end
