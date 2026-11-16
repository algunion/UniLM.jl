# ============================================================================
# Gemini's Decisions API — the Gemini wrapper of the System One surface.
#
# POST {GEMINI_NATIVE_BASE}/decisions answers noul, choice and score questions about an
# `input` (https://ai.google.dev/gemini-api/docs/decisions-api; Beta, on v1beta only, and its
# only model is gemini-decision-1). Its questions are a map keyed by name and its answers come
# back under the same keys, as on TypeSafe's wire, but the input is text or Interactions
# content blocks, a question's instructions are text only, a Noul's criteria are
# `true_description` and `false_description`, the reply names neither the model nor a request
# id, and a blocked input is reported once for the whole request. This file maps the generic
# surface onto that wire and refuses, before any I/O, what the wire cannot carry.
# ============================================================================

"""
    GeminiDecisionsProtocol <: SystemOneProtocol

Gemini's Decisions API (Beta) as a System One backend, and the protocol of
[`GEMINIServiceEndpoint`](@ref): `ask(state, questions...; service =
GEMINIServiceEndpoint)`, a [`with_system_one`](@ref)`(GEMINIServiceEndpoint)` scope and
the natural-language verbs send `POST
https://generativelanguage.googleapis.com/v1beta/decisions` with `GEMINI_API_KEY`, naming
`"gemini-decision-1"` — the only model the API serves — when the call names none; a
model the call names is sent as written (`"models/gemini-decision-1"` was answered too).
[`GEMINIOpenAIServiceEndpoint`](@ref) has no System One protocol.

The request:

- `state` is the `input`: a string as it is, unless it is blank; a JSON value (object,
  array, tuple, `NamedTuple`) as its JSON text; an [`ImageAttachment`](@ref), or a vector or
  tuple of strings and images, as a list of `text` and `image` content blocks in order (each
  image its base64 `data` and `mime_type`, and its `detail` as the block's `resolution`:
  `"low"` or `"high"`, none for `"auto"`). The images hold at most 100 MB (read as MiB)
  in all.
- each question is the entry of `questions` under its name, which is 1 to 64 ASCII
  letters, digits, `_` and `-`: a [`NoulQuestion`](@ref) a `noul`, its `yes` and `no` the
  `true_description` and `false_description` of its `criteria`; a [`ChoiceQuestion`](@ref)
  a `choice` of 2 to 255 options, each option's name a key of `criteria` — 1 to 128
  characters and at most 512 bytes, with no leading or trailing whitespace and no control,
  format or non-ASCII space characters — and its description the value (`""` for
  `nothing`); a [`ScoreQuestion`](@ref) a `score` of 2 to 255 levels, each a non-empty
  string; a [`RawQuestion`](@ref) its fields as written, nesting objects and arrays at most
  98 levels, as the body takes 100 and holds a question at the third. At most 1,000
  questions.
- every question needs `instructions`: a non-empty string, or a vector or tuple of
  non-empty strings sent as `text` blocks, which the API joins with no separator.

What the wire has no form for is an `ArgumentError` before any I/O, naming Gemini's rule
and the fix: a blank state, an empty vector state or text block, a [`Message`](@ref) in
the state (the API takes no conversation: pass its text); an image nested in JSON or
anywhere in a question; `detail="original"`; an [`AudioAttachment`](@ref) or a
[`DocumentAttachment`](@ref) anywhere (UniLM's System One state holds text and images);
instructions that are missing, empty, or not text — an object, an image, a vector holding
anything but strings; an option, a description, a level or a criterion outside the rules
above; a [`BoxQuestion`](@ref); a RawQuestion holding an image; any provider option (the
API takes none).

The reply: answers are matched to the questions by name. A `noul` reads as a
[`NoulAnswer`](@ref), a `choice` as a [`ChoiceAnswer`](@ref), a `score` as a
[`ScoreAnswer`](@ref) whose `legend` maps each level to its text, and an answer of another
type, to a RawQuestion, as an [`UnknownAnswer`](@ref). Probabilities arrive unrounded, so a
distribution sums to 1 within the floating-point tolerance every unrounded backend has. The
`confidence` of a Choice and a Score is recomputed from the probabilities with TypeSafe's
published formulas, as on Ollama, so a threshold means the same on every backend; Gemini's
own — the top probability — stays in each answer's `raw`. An input the API's filters block
(a 200 whose `prompt_feedback.block_reason` is set) reads as a [`RefusalAnswer`](@ref) for
every question, its `raw` the `block_reason` and `block_reason_message`; the
natural-language verbs take it as a decline. The reply names no model, so the response's
`model` is the one requested without one leading `models/`, the SDK's form of the same id;
`request_id` is `nothing`, as no header carries one; `usage`
counts the input (its audio apart) and the output, always 0. A reply with a status other
than 200 is a [`SystemOneFailure`](@ref) whose `error_type` is the error's `code` when it is
a string (`"invalid_request"`, `"not_found"`, and for a blocked input `"safety"`,
`"prohibited_content"`, `"blocklist"`, `"image_safety"` or `"content_blocked"`), else its
`status`. Recordings are filed under `/gemini-decisions/v1beta/decisions`, each image keyed
and written as the SHA-256 of its base64 text, never its bytes. [`estimated_cost`](@ref)
bills the input tokens at the input rate of the model, named without one leading `models/`,
the same for every modality, and no output.

Public extension API (not exported).
"""
struct GeminiDecisionsProtocol <: SystemOneProtocol end

system_one_protocol(::Type{GEMINIServiceEndpoint}) = GeminiDecisionsProtocol()

const _GEMINI_DECISIONS_PATH = "/v1beta/decisions"
const _GEMINI_DECISIONS_MODEL = "gemini-decision-1"

_system_one_default_model(::GeminiDecisionsProtocol, service)::String = _GEMINI_DECISIONS_MODEL
_system_one_namespace(::GeminiDecisionsProtocol)::String = "gemini-decisions"
# No per-question refusal exists; a blocked input is one for every question.
_allows_refusal(::GeminiDecisionsProtocol)::Bool = true
_allows_raw(::GeminiDecisionsProtocol)::Bool = true
# An image block takes `resolution` (low, medium, high, ultra_high), which `_gemini_detail_level` maps an
# image's detail level to; "original" has none and is refused there.
_image_detail_refusal(::GeminiDecisionsProtocol)::Nothing = nothing
# Transport: a hosted service answering in about 0.3 to 0.7 s (measured on 2026-11-16), so the hosted bounds;
# `ask_each` keeps the fallback 8 requests in flight, under the 25 a second of the API's first paid tier.
_system_one_default_config(::GeminiDecisionsProtocol, service)::Union{Nothing,RequestConfig} = _HOSTED_SYSTEM_ONE_CONFIG

# The API's URL, the native base's `/decisions`; an endpoint that speaks this wire elsewhere (a gateway, a
# test server) gives its base through `_resolve_base_url`, which the path follows.
_gemini_decisions_url(::Type{GEMINIServiceEndpoint})::String = GEMINI_NATIVE_BASE * "/decisions"
_gemini_decisions_url(service::ServiceEndpointSpec)::String = _api_base_url(service) * _GEMINI_DECISIONS_PATH

# ─── Encoding ────────────────────────────────────────────────────────────────

# The Limits of the guide, each answered with HTTP 400 naming the rule when broken (measured on 2026-11-16).
# The 262,144-token context is the server's to count. Inline data: "up to 100 MB per request", read as
# MiB, the larger reading, so nothing the API takes is refused here.
const _GEMINI_DECISIONS_MAX_QUESTIONS = 1_000
const _GEMINI_DECISIONS_CHOICES = 2:255
const _GEMINI_DECISIONS_LEVELS = 2:255
const _GEMINI_DECISIONS_NAME = r"^[A-Za-z0-9_-]{1,64}$"
const _GEMINI_DECISIONS_OPTION_CHARS = 128
const _GEMINI_DECISIONS_OPTION_BYTES = 512
const _GEMINI_DECISIONS_INLINE_BYTES = 100 * 1024^2
# "JSON nesting depth: 100 levels" of the request body, which holds a question at the third level (the body
# object, then `questions`), so a question's fields nest at most 98 more: the API answered a body 101 levels deep
# with HTTP 400 "exceeds maximum nesting depth of 100" and one of 100 with its next rule (measured on 2026-11-16).
# A JSON state is sent as its text, a string, which the depth does not count: a state's text 101 levels deep was
# answered.
const _GEMINI_DECISIONS_BODY_DEPTH = 100
const _GEMINI_DECISIONS_QUESTION_DEPTH = _GEMINI_DECISIONS_BODY_DEPTH - 2

_gemini_decisions_refuse(name::String, why::String) = throw(ArgumentError(
    "Gemini's Decisions API cannot carry the question $(repr(name)): $why"))

const _GEMINI_DECISIONS_IMAGE_FORMS = "an ImageAttachment, or a Vector or Tuple of strings and ImageAttachments, " *
                                      "sent as content blocks in order"

function _gemini_decisions_name(name::String)::Nothing
    occursin(_GEMINI_DECISIONS_NAME, name) || throw(ArgumentError(
        "Gemini's Decisions API takes a question's name as 1 to 64 ASCII letters, digits, `_` and `-`; " *
        "$(repr(name)) is not one. Rename the question, as in `\"is_damaged\" => noul(…)`"))
    nothing
end

# A text block of the input or of the instructions. The API answers an empty one with HTTP 400 "Missing text
# in content of type text."; a block of whitespace alone separates others ("Include your own spaces or
# newlines (`\n`) between blocks"), and is sent.
function _gemini_decisions_text(text::AbstractString, what::String)::JSON.Object{String,Any}
    isempty(text) && throw(ArgumentError(
        "Gemini's Decisions API takes no empty text block, and $what is empty. Drop it, or write the text"))
    JSON.Object{String,Any}("type" => "text", "text" => String(text))
end

# Instructions are text only ("Question instructions accept text only"): a string, or text blocks for a
# vector or tuple of strings. An object is not turned into its JSON text: that would send words the caller
# did not write as instructions.
function _gemini_decisions_instructions(name::String, instructions)::Union{String,Vector{Any}}
    isnothing(instructions) && _gemini_decisions_refuse(name,
        "every question needs `instructions`, which the API requires. Give them as the first argument, as in " *
        "`noul(\"Is the item damaged?\")`")
    if instructions isa AbstractString
        isempty(instructions) && _gemini_decisions_refuse(name,
            "its instructions are empty, and the API requires non-empty instructions. Write the question")
        return String(instructions)
    end
    texts = instructions isa Union{AbstractVector,Tuple} && all(x -> x isa AbstractString, instructions)
    texts || _gemini_decisions_refuse(name,
        "its instructions are a $(typeof(instructions)), and the API takes instructions as text only: a string, or " *
        "a Vector or Tuple of strings. " * (isempty(_collect_images!(ImageAttachment[], instructions)) ?
            "Write them as text (for JSON data, its text: `JSON.json(data)`)" :
            "An image belongs in the state, as $(_GEMINI_DECISIONS_IMAGE_FORMS)"))
    isempty(instructions) && _gemini_decisions_refuse(name,
        "its instructions hold no text, and the API requires at least one block. Give them a string")
    Any[_gemini_decisions_text(s, "part $i of the instructions of $(repr(name))") for (i, s) in enumerate(instructions)]
end

_gemini_decisions_head(type::String, name::String, instructions)::JSON.Object{String,Any} =
    JSON.Object{String,Any}("type" => type, "instructions" => _gemini_decisions_instructions(name, instructions))

# What `guidance` is when it is not a string: the fix names the state for an image, text for the rest.
_gemini_decisions_as_text(x)::String = x isa ImageAttachment ?
    "An image belongs in the state, as $(_GEMINI_DECISIONS_IMAGE_FORMS)" : "Write it as text"

function _gemini_decisions_question(name::String, q::NoulQuestion)::JSON.Object{String,Any}
    o = _gemini_decisions_head("noul", name, q.instructions)
    c = q.criteria
    isnothing(c) && return o
    criteria = JSON.Object{String,Any}()
    for (side, key, v) in (("yes", "true_description", c.yes), ("no", "false_description", c.no))
        isnothing(v) && continue
        v isa AbstractString || _gemini_decisions_refuse(name,
            "its `$side` criterion must be a string, its $key, not $(typeof(v)). $(_gemini_decisions_as_text(v))")
        criteria[key] = String(v)
    end
    isempty(criteria) || (o["criteria"] = criteria)
    o
end

# A control (Cc), format (Cf) or separator character other than the ASCII space: "ids must not contain
# newlines, tabs, non-ASCII spaces, or other control or format characters" (the API's 400 on 2026-11-16).
_gemini_decisions_unsafe(c::AbstractChar)::Bool =
    (k = Base.Unicode.category_abbrev(c); k in ("Cc", "Cf", "Zl", "Zp") || k == "Zs" && c != ' ')

function _gemini_decisions_option(name::String, option::String)::Nothing
    n, bytes = length(option), ncodeunits(option)
    shown = repr(n > 40 ? first(option, 40) * "…" : option)
    n <= _GEMINI_DECISIONS_OPTION_CHARS && bytes <= _GEMINI_DECISIONS_OPTION_BYTES || _gemini_decisions_refuse(name,
        "its option $shown is $n characters ($bytes bytes), and an option's name takes at most " *
        "$(_GEMINI_DECISIONS_OPTION_CHARS) characters and $(_GEMINI_DECISIONS_OPTION_BYTES) bytes. Give the option " *
        "a short name and the sentence as its description, as in `choice(\"…\", [\"refund\" => \"the customer " *
        "wants a refund\"])`")
    (isspace(first(option)) || isspace(last(option))) && _gemini_decisions_refuse(name,
        "its option $shown begins or ends with whitespace, which an option's name may not. Trim it")
    any(_gemini_decisions_unsafe, option) && _gemini_decisions_refuse(name,
        "its option $shown holds a control or format character, or a space other than the ASCII one, which an " *
        "option's name may not. Remove it")
    nothing
end

function _gemini_decisions_question(name::String, q::ChoiceQuestion)::JSON.Object{String,Any}
    o = _gemini_decisions_head("choice", name, q.instructions)
    n = length(q.criteria)
    n in _GEMINI_DECISIONS_CHOICES || _gemini_decisions_refuse(name,
        "a choice offers $(first(_GEMINI_DECISIONS_CHOICES)) to $(last(_GEMINI_DECISIONS_CHOICES)) options, and it " *
        "has $n. " * (n < first(_GEMINI_DECISIONS_CHOICES) ? "Add one, such as \"other\"" : "Split it, or narrow the options"))
    criteria = JSON.Object{String,Any}()
    for (option, description) in q.criteria
        _gemini_decisions_option(name, option)
        isnothing(description) || description isa AbstractString || _gemini_decisions_refuse(name,
            "the description of its option $(repr(option)) must be a string or `nothing`, not $(typeof(description)). " *
            _gemini_decisions_as_text(description))
        criteria[option] = isnothing(description) ? "" : String(description)
    end
    o["criteria"] = criteria
    o
end

function _gemini_decisions_question(name::String, q::ScoreQuestion)::JSON.Object{String,Any}
    o = _gemini_decisions_head("score", name, q.instructions)
    n = length(q.criteria)
    n in _GEMINI_DECISIONS_LEVELS || _gemini_decisions_refuse(name,
        "a score has $(first(_GEMINI_DECISIONS_LEVELS)) to $(last(_GEMINI_DECISIONS_LEVELS)) levels, and it has $n. " *
        (n < first(_GEMINI_DECISIONS_LEVELS) ? "Add a level" : "Merge levels into a coarser rubric"))
    for (i, level) in enumerate(q.criteria)
        level isa AbstractString && !isempty(level) || _gemini_decisions_refuse(name,
            "its level $(i - 1) must be a non-empty string, its description, not " *
            (level isa AbstractString ? "an empty one. Describe the level" :
                                        "$(typeof(level)). $(_gemini_decisions_as_text(level))"))
    end
    o["criteria"] = Any[String(level) for level in q.criteria]
    o
end

_gemini_decisions_question(name::String, ::BoxQuestion) = _gemini_decisions_refuse(name,
    "it is a Box question, and the API answers noul, choice and score questions only. Send it to a backend " *
    "that answers Box questions")

function _gemini_decisions_question(name::String, q::RawQuestion)::JSON.Object{String,Any}
    isempty(_collect_images!(ImageAttachment[], q.fields)) || _gemini_decisions_refuse(name,
        "it holds an image, and a question's instructions are text only: an image goes in the state, as " *
        _GEMINI_DECISIONS_IMAGE_FORMS)
    _nests_deeper(q.fields, _GEMINI_DECISIONS_QUESTION_DEPTH) && _gemini_decisions_refuse(name,
        "it nests objects and arrays more than $(_GEMINI_DECISIONS_QUESTION_DEPTH) levels deep, and the request body " *
        "takes a JSON nesting depth of $(_GEMINI_DECISIONS_BODY_DEPTH), holding the question at its third level. " *
        "Flatten it")
    q.fields
end

_gemini_decisions_question(name::String, q::SystemOneQuestion) = _gemini_decisions_refuse(name,
    "it is a $(nameof(typeof(q))), and the API answers noul, choice and score questions; a RawQuestion is sent " *
    "verbatim")

# The `input`: text, or content blocks of text and images. A JSON state is sent as its JSON text ("Serialize
# JSON objects and arrays to a string and pass them as text").
function _gemini_decisions_input(state::AbstractString)::String
    isempty(strip(state)) && throw(ArgumentError(
        "Gemini's Decisions API takes no empty input (it answers a blank one with HTTP 400 \"input must not be " *
        "empty\"), and this state is $(repr(state)). Pass the text the questions are about"))
    String(state)
end
_gemini_decisions_input(state::ImageAttachment)::Vector{Any} = Any[_gemini_decisions_image(state)]
# A tuple is taken as a vector is, as every System One backend takes it.
function _gemini_decisions_input(state::Union{AbstractVector,Tuple})
    isempty(state) && throw(ArgumentError(
        "Gemini's Decisions API takes no empty input, and this state is an empty $(nameof(typeof(state))). Pass " *
        "the text the questions are about"))
    any(x -> x isa Message, state) && throw(ArgumentError(
        "Gemini's Decisions API takes no conversation as its input (\"Serialize multi-turn context as text\"), " *
        "and this state holds a Message. Pass the conversation's text"))
    all(x -> x isa Union{AbstractString,ImageAttachment}, state) && any(x -> x isa ImageAttachment, state) ?
        Any[x isa ImageAttachment ? _gemini_decisions_image(x) : _gemini_decisions_text(x, "text block $i of the state")
            for (i, x) in enumerate(state)] :
        _gemini_decisions_json(state)
end
_gemini_decisions_input(state)::String = _gemini_decisions_json(state)

function _gemini_decisions_json(state)::String
    isempty(_collect_images!(ImageAttachment[], state)) || throw(ArgumentError(
        "Gemini's Decisions API takes images in the state only as $(_GEMINI_DECISIONS_IMAGE_FORMS); this " *
        "$(nameof(typeof(state))) holds one nested in JSON, which is sent as its text. Pass the text and the " *
        "images as such a Vector"))
    _system_one_body(state)
end

# An image block (ImageContent): its base64 bytes, its MIME type, and its detail level as `resolution`.
function _gemini_decisions_image(x::ImageAttachment)::JSON.Object{String,Any}
    o = JSON.Object{String,Any}("type" => "image", "mime_type" => x.mime, "data" => base64encode(x.data))
    level = _gemini_detail_level(x.detail)
    isnothing(level) || (o["resolution"] = level)
    o
end

# A block as a recording keys and writes it: an image's bytes as their digest, `data:<mime>;sha256,<hex>`.
function _gemini_decisions_keyed(block::JSON.Object{String,Any})::JSON.Object{String,Any}
    block["type"] == "image" || return block
    keyed = JSON.Object{String,Any}()
    for (k, v) in block
        keyed[k] = k == "data" ? _media_digest(block["mime_type"], v) : v
    end
    keyed
end

function _check_gemini_inline(nbytes::Integer; limit::Integer=_GEMINI_DECISIONS_INLINE_BYTES)::Nothing
    nbytes <= limit || throw(ArgumentError(
        "Gemini's Decisions API takes at most $limit bytes of inline data per request; this request's images " *
        "hold $nbytes. Downscale them, or split them across requests"))
    nothing
end

_validate_system_one_options(::GeminiDecisionsProtocol, options::NamedTuple)::Nothing = isempty(options) ? nothing :
    throw(ArgumentError("Gemini's Decisions API takes no provider options: its request holds the model, the input " *
                        "and the questions alone; got $(join(("`$k`" for k in keys(options)), ", ")). Drop them"))

# A clip has an audio block on this wire, but a System One state carries text and images: refused, never
# written as bytes the API would read as text.
_check_medium(::GeminiDecisionsProtocol, service, a::AudioAttachment, place::String) = throw(ArgumentError(
    "Gemini's Decisions API is sent a System One request's text and images, so $(a) in $(place) cannot be " *
    "sent: UniLM does not send audio in a decision's input. Put the clip's transcript in the request"))

function encode_system_one(p::GeminiDecisionsProtocol, service, request::SystemOneRequest)::SystemOneWire
    _check_request(p, service, request)
    n = length(request.questions)
    n <= _GEMINI_DECISIONS_MAX_QUESTIONS || throw(ArgumentError(
        "Gemini's Decisions API takes at most $(_GEMINI_DECISIONS_MAX_QUESTIONS) questions per request; this one " *
        "has $n. Split it into several requests"))
    _validate_system_one_options(p, request.options)
    questions = JSON.Object{String,Any}()
    for (name, q) in request.questions
        _gemini_decisions_name(name)
        questions[name] = _gemini_decisions_question(name, q)
    end
    _check_gemini_inline(sum(x -> length(x.data), _collect_images!(ImageAttachment[], request.state); init=0))
    input = _gemini_decisions_input(request.state)
    body = JSON.Object{String,Any}("model" => _system_one_model(p, service, request), "input" => input,
                                   "questions" => questions)
    url, sent = _gemini_decisions_url(service), _system_one_body(body)
    input isa String && return SystemOneWire("POST", url, _GEMINI_DECISIONS_PATH, sent)
    body["input"] = Any[_gemini_decisions_keyed(b) for b in input]
    SystemOneWire("POST", url, _GEMINI_DECISIONS_PATH, sent; record_body=_system_one_body(body))
end

# ─── Decoding ────────────────────────────────────────────────────────────────
# The answers are TypeSafe's shape (`type`, `noul`; `choice`, `probabilities`; `score`, `legend`), read by
# its decoder; what differs is the blocked input, the model the reply does not name, and the confidence.

# `prompt_feedback.block_reason` set: "the prompt (`input` and question text) was blocked by the input
# policy, no decisions were evaluated, and `answers` is empty" (the API reference's CreateDecisionResponse).
function _gemini_decisions_block(raw::Dict{String,Any})::Union{Nothing,Dict{String,Any}}
    feedback = get(raw, "prompt_feedback", nothing)
    isnothing(feedback) && return nothing
    feedback isa AbstractDict || throw(ArgumentError(
        "the Gemini Decisions response's prompt_feedback is not an object (got $(typeof(feedback)))"))
    reason = get(feedback, "block_reason", nothing)
    isnothing(reason) && return nothing
    reason isa AbstractString || throw(ArgumentError(
        "the Gemini Decisions response's prompt_feedback.block_reason is not a string (got $(typeof(reason)))"))
    answers = get(raw, "answers", nothing)
    isnothing(answers) || answers isa Union{AbstractDict,AbstractVector} && isempty(answers) || throw(ArgumentError(
        "the Gemini Decisions response blocks its input (block_reason $(repr(reason))) and still carries answers"))
    Dict{String,Any}(string(k) => v for (k, v) in feedback)
end

function _gemini_decisions_answers(raw::Dict{String,Any})::Dict{String,SystemOneAnswer}
    list = get(raw, "answers", nothing)
    list isa AbstractDict || throw(ArgumentError("the Gemini Decisions response carries no \"answers\" object"))
    out = Dict{String,SystemOneAnswer}()
    for (k, v) in list
        name = string(k)
        v isa AbstractDict || throw(ArgumentError("Gemini Decisions answer $(repr(name)) is not a JSON object"))
        out[name] = _decode_answer(name, v)
    end
    out
end

# Usage is the Interactions usage object: `total_input_tokens` (every modality; video frames and PDF pages
# count as image tokens), `total_output_tokens` (always 0) and `total_cached_tokens`, with the audio among
# the input in `input_tokens_by_modality`. As on OpenAI's Decisions API, a count that is missing reads as 0,
# and one that is present but no token count as 0 with one warning.
function _gemini_decisions_usage(d::AbstractDict)::TokenUsage
    haskey(d, "usage") || return TokenUsage()
    u = d["usage"]
    counts = (_decisions_count(u, ("total_input_tokens",)), _decisions_count(u, ("total_output_tokens",)),
              _decisions_count(u, ("total_cached_tokens",)))
    any(isnothing, counts) &&
        @warn "System One usage holds a value that is not a non-negative whole token count; it reads as 0" usage = u
    i, o, cached = map(c -> something(c, 0), counts)
    audio = u isa AbstractDict ? _modality_tokens(get(u, "input_tokens_by_modality", nothing), "audio", "tokens") : 0
    TokenUsage(prompt_tokens=i, completion_tokens=o, total_tokens=i + o, cached_tokens=cached, prompt_audio_tokens=audio)
end

function decode_system_one(p::GeminiDecisionsProtocol, service, request::SystemOneRequest,
                           response::HTTP.Response)::Union{SystemOneSuccess,SystemOneFailure}
    response.status == 200 || return _gemini_decisions_failure(response)
    parsed = JSON.parse(String(response.body); dicttype=Dict{String,Any})
    parsed isa AbstractDict || throw(ArgumentError("the Gemini Decisions response body is not a JSON object"))
    raw = Dict{String,Any}(string(k) => v for (k, v) in parsed)
    blocked = _gemini_decisions_block(raw)
    answers = isnothing(blocked) ? _gemini_decisions_answers(raw) :
              Dict{String,SystemOneAnswer}(name => RefusalAnswer(copy(blocked)) for (name, _) in request.questions)
    # The reply names no model: the one requested answered, as the API serves one.
    model, usage = _gemini_decisions_model(_system_one_model(p, service, request)), _gemini_decisions_usage(raw)
    _check_answers(p, request, SystemOneResponse(model, answers, usage, nothing, raw, service))
    surfaced = Dict{String,SystemOneAnswer}(name => _surface_confidence(name, a) for (name, a) in answers)
    SystemOneSuccess(SystemOneResponse(model, surfaced, usage, nothing, raw, service))
end

# A model's id without one leading "models/": the SDK documents CreateDecisionRequest.model as `models/{model}`
# and its guide names the bare id, and the API answered both (measured on 2026-11-16), so the two are one model,
# named and priced alike. The request still sends the id as given.
_gemini_decisions_model(model::AbstractString)::String = String(chopprefix(model, "models/"))

# The default price of an evaluation, for the model named without that prefix: a `model` given to
# `estimated_cost` as well as the response's.
_system_one_cost(::GeminiDecisionsProtocol, result::SystemOneSuccess, model::Union{Nothing,String},
                 pricing::AbstractDict{String,PriceRow})::Float64 =
    _system_one_cost(nothing, result, _gemini_decisions_model(something(model, result.response.model)), pricing)

# An error body is `{"error": {"message", "code"}}` with a string `code` (`invalid_request`, `not_found`, and
# for a blocked input `safety`, `prohibited_content`, `blocklist`, `image_safety` or `content_blocked`), or
# Google's `{"error": {"code": 400, "message", "status": "INVALID_ARGUMENT", "details"}}` for a key the API
# refuses (both measured on 2026-11-16).
function _gemini_decisions_failure(resp::HTTP.Response)::SystemOneFailure
    body = String(resp.body)
    parsed = _typesafe_parse_body(body)
    err = parsed isa AbstractDict ? get(parsed, "error", nothing) : nothing
    field(key) = (v = err isa AbstractDict ? get(err, key, nothing) : nothing; v isa AbstractString ? String(v) : nothing)
    SystemOneFailure(response=body, status=resp.status, error_type=something(field("code"), field("status"), Some(nothing)),
                     message=something(field("message"), _truncate_body(body)), retry_after=_retry_after(resp))
end
