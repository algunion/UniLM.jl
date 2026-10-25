# [Write your own decision provider](@id decisions_custom)

```@setup custom
using UniLM, JSON
```

A decision provider is an endpoint that declares `:system_one` among its
[`provider_capabilities`](@ref): [`ask`](@ref), [`ask_each`](@ref), [`nl_classify`](@ref), [`nl_dispatch`](@ref),
[`@branch`](@ref) and [`with_system_one`](@ref) accept it, and reach it through its protocol, a Holy trait of the
endpoint that says how a request is sent and how its reply is read. A service that speaks TypeSafe's wire needs no
protocol of its own; a service with a wire of its own needs one.

| I want to… | Section |
| :--- | :--- |
| send decisions to a gateway, a proxy or a mock that speaks TypeSafe's wire | [A gateway on TypeSafe's wire](@ref decisions_custom_gateway) |
| reach a decision service whose wire is its own | [A protocol of your own](@ref decisions_custom_protocol) |
| change what UniLM's shared machinery assumes of a provider | [Internal hooks](@ref decisions_custom_hooks) |

## [A gateway on TypeSafe's wire](@id decisions_custom_gateway)

A server that speaks TypeSafe's wire needs no code at all to receive [`TYPESAFEServiceEndpoint`](@ref)'s requests:
set `TYPESAFE_BASE_URL` to its root, and every request goes there, read when each is sent
([Setup](@ref jev_setup__next)). To give such a server an endpoint of its own, beside TypeSafe's, declare
`:system_one`, its root and its credentials; an endpoint without a
[`system_one_protocol`](@ref UniLM.system_one_protocol) method speaks
[`TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol), TypeSafe's wire. `_api_base_url` is an internal hook
([Internal hooks](@ref decisions_custom_hooks)):

```@example custom
struct JevGateway <: UniLM.ServiceEndpoint end
UniLM.provider_capabilities(::Type{JevGateway}) = Set([:system_one])
UniLM._api_base_url(::Type{JevGateway}) = "https://jev-gateway.example.com"   # internal hook
UniLM.auth_header(::Type{JevGateway}) =
    ["Authorization" => "Bearer $(ENV["JEV_GATEWAY_KEY"])", "Content-Type" => "application/json"]

request = SystemOneRequest("My parcel never arrived.", ["team" => choice("Which team?", ["billing", "shipping"])])
wire = UniLM.encode_system_one(UniLM.system_one_protocol(JevGateway), JevGateway, request)
(UniLM.system_one_protocol(JevGateway), wire.url, wire.path)
```

`ask(state, questions...; service = JevGateway)` sends that request there, with the key read from
`JEV_GATEWAY_KEY` only when it goes out. Its requests are TypeSafe's byte for byte, with TypeSafe's limits, default
model, timeouts and retries, and its recordings are keyed as TypeSafe's — on the request's path and body, never its
host — so an answer recorded through one replays for the other
([Testing and recordings](@ref jev_testing)).

## [A protocol of your own](@id decisions_custom_protocol)

A protocol is a singleton type, a subtype of [`UniLM.SystemOneProtocol`](@ref UniLM.SystemOneProtocol), and three
methods make it a provider. They are UniLM's public extension API, not exported:

| Method | What it does |
| :--- | :--- |
| [`UniLM.system_one_protocol`](@ref UniLM.system_one_protocol)`(service)` | returns your protocol for your endpoint: a method on `::Type{MyEndpoint}` for an endpoint with no fields, on the instance for one with fields |
| [`UniLM.encode_system_one`](@ref UniLM.encode_system_one)`(p, service, request)` | returns the [`UniLM.SystemOneWire`](@ref UniLM.SystemOneWire) the request is sent as, and what a recording is keyed on; throws, before any I/O, an `ArgumentError` naming your provider, the rule and the fix for what your wire cannot carry; sends the default model when `request.model` is `nothing` |
| [`UniLM.decode_system_one`](@ref UniLM.decode_system_one)`(p, service, request, response)` | returns the typed result of the reply: a [`SystemOneSuccess`](@ref), or a [`SystemOneFailure`](@ref) for a status other than 200; throws for a 200 that is no usable answer, which [`ask`](@ref) returns as a [`SystemOneCallError`](@ref) |

Three protocols ship with UniLM: [`UniLM.TypeSafeProtocol`](@ref UniLM.TypeSafeProtocol) (TypeSafe's
`POST /v1/systemone`), [`UniLM.OpenAIDecisionsProtocol`](@ref UniLM.OpenAIDecisionsProtocol) (OpenAI's
`POST /v1/decisions`, for `OPENAIServiceEndpoint`) and [`UniLM.OllamaSystemOneProtocol`](@ref UniLM.OllamaSystemOneProtocol)
(Ollama's `POST /v1/systemone`, for `OllamaEndpoint`). The endpoint itself declares `:system_one` in
[`provider_capabilities`](@ref) and gives its credentials in
`auth_header`, which UniLM reads only when a request goes out, so a replay needs no key. The rest is shared by every
provider: the provider, model and options a call resolves ([`with_system_one`](@ref)), recording and replay,
retries and cancellation, the connection pool and the typed results. A labelling service whose wire takes a text
and its labels and answers one label with a score for each:

```@example custom
const LABEL_API = "https://labels.example.com"

struct LabelAPI <: UniLM.ServiceEndpoint end
UniLM.provider_capabilities(::Type{LabelAPI}) = Set([:system_one])
UniLM.auth_header(::Type{LabelAPI}) =
    ["Authorization" => "Bearer $(ENV["LABEL_API_KEY"])", "Content-Type" => "application/json"]

struct LabelProtocol <: UniLM.SystemOneProtocol end
UniLM.system_one_protocol(::Type{LabelAPI}) = LabelProtocol()

function UniLM.encode_system_one(p::LabelProtocol, service, r::SystemOneRequest)
    UniLM._check_request(p, service, r)       # internal: the refusals every provider shares
    length(r.questions) == 1 && last(only(r.questions)) isa ChoiceQuestion && r.state isa AbstractString ||
        throw(ArgumentError("LabelAPI answers one Choice about a text: send one `choice` and a string state"))
    labels = collect(keys(last(only(r.questions)).criteria))
    body = JSON.json((model = something(r.model, "labeller-1"), text = r.state, labels = labels))
    UniLM.SystemOneWire("POST", LABEL_API * "/v1/label", "/v1/label", body)
end

function UniLM.decode_system_one(p::LabelProtocol, service, r::SystemOneRequest, resp::UniLM.HTTP.Response)
    body = String(resp.body)
    resp.status == 200 || return SystemOneFailure(response = body, status = resp.status, message = body)
    d = JSON.parse(body)
    scores = Dict{String,Float64}(k => v for (k, v) in d["scores"])
    n = length(scores)
    confidence = n == 1 ? 1.0 : clamp((n * maximum(values(scores)) - 1) / (n - 1), 0.0, 1.0)
    answer = ChoiceAnswer(d["label"], confidence, scores, Dict{String,Any}(d))
    response = SystemOneResponse(d["model"], Dict{String,SystemOneAnswer}(first(only(r.questions)) => answer),
                                 TokenUsage(), nothing, Dict{String,Any}(d), service)
    UniLM._check_answers(p, r, response)      # internal: the checks every provider's answers pass
    SystemOneSuccess(response)
end
nothing # hide
```

`decode_system_one` types its reply as an `HTTP.Response`, as UniLM's own method does: left untyped, a call would
match both methods equally and Julia would refuse it as ambiguous. The encoder and the decoder run on their own,
with no service:

```@example custom
request = SystemOneRequest("I was charged twice for order #4471.",
                           ["team" => choice("Which team should handle this message?", ["billing", "shipping", "other"])])
wire = UniLM.encode_system_one(LabelProtocol(), LabelAPI, request)
println(wire.url, "\n", wire.body)

reply = UniLM.HTTP.Response(200, ["Content-Type" => "application/json"], Vector{UInt8}(
    JSON.json((model = "labeller-1", label = "billing", scores = (billing = 0.9, shipping = 0.06, other = 0.04)))))
r = UniLM.decode_system_one(LabelProtocol(), LabelAPI, request, reply)
(r["team"].choice, round(r["team"].confidence; digits = 2))
```

With these, `ask(state, "team" => choice(…); service = LabelAPI)`, a `with_system_one(LabelAPI)` scope and the
natural-language verbs reach the service, and [`with_recorded_answers`](@ref) records and replays its exchanges, filed
under `/LabelProtocol/v1/label`: the protocol's name in front of the wire's path, so they never replay as another
provider's. The encoder refuses what the wire cannot carry rather than adapt it, as every built-in provider does:
a request changed to fit would ask a question you did not write.

## [Internal hooks](@id decisions_custom_hooks)

The shared machinery asks a protocol a few more things through functions whose names start with `_`. They are
internal: not part of UniLM's public API, they can change in any release, and a protocol that defines them must be
checked again on each upgrade. Each has a fallback, which a protocol that defines nothing gets:

| Internal hook | Fallback | The built-in providers |
| :--- | :--- | :--- |
| `_system_one_default_model(p, service)` | an `ArgumentError`: a call must name its model | TypeSafe: `TYPESAFE_DEFAULT_MODEL`, else `jev-latest`; OpenAI: `gpt-6-luna`; Ollama: none |
| `_system_one_namespace(p)` | the protocol's type name, in front of a recording's path | TypeSafe: none; OpenAI: `openai-decisions`; Ollama: `ollama` |
| `_system_one_request_id(p, response)` | `nothing` | TypeSafe: `x-typesafe-request-id`; OpenAI: `x-request-id` |
| `_probability_tolerance(p, n)`, `_probability_slack(p)` | `1e-6` each: how far `n` probabilities may sum from 1, and the chosen option fall below the most probable | TypeSafe: `0.005n + 1e-6`, `0.01 + 1e-6`; OpenAI: `0.005n + 1e-9`, `0.01 + 1e-9`; Ollama: `n·0.5e-4 + 1e-6`, `1e-4 + 1e-6` |
| `_allows_refusal(p)` | `false`: a declined question makes the call a `SystemOneCallError` | OpenAI: `true`, a [`RefusalAnswer`](@ref) |
| `_allows_raw(p)` | `false`: a [`RawQuestion`](@ref) is refused before sending | all three: `true` |
| `_system_one_models(p, service, config, cancel)` | an `ArgumentError`: [`list_models`](@ref) has no listing | TypeSafe: `GET /v1/models` |
| `_system_one_default_config(p, service)` | `nothing`: the [`RequestConfig`](@ref) field defaults, 600 s per attempt and 900 s in all | TypeSafe and OpenAI: 60 s per attempt, 150 s in all |
| `_system_one_retry_header(p)` | `nothing`: retries carry no count | TypeSafe: `X-TypeSafe-Retry-Count` |
| `_system_one_concurrency(p, service)` | 8: the requests [`ask_each`](@ref) keeps in flight when its `ntasks` names none | TypeSafe and OpenAI: 16; Ollama: 1 |
| `_system_one_cost(p, result, model, pricing)` | input, cached input and output tokens at the model's row of the price table | OpenAI: every input token at the input rate, and no output |
| `_api_base_url(service)` | none for an endpoint of your own: define it, or write the URL into the wire | where the built-in encoders start a request's URL |
| `_check_request(p, service, request)` | — | called first by each built-in encoder: a blank or repeated question name, criteria that are not valid, a `RawQuestion` the protocol does not allow, a `DocumentAttachment` or an image `detail` |
| `_check_answers(p, request, response)` | — | called by each built-in decoder: every question answered once, of its kind, within the protocol's tolerance |

[`UniLM.decode_system_one`](@ref UniLM.decode_system_one) promises answers that have passed the checks every
provider's answers pass, and only `_check_answers` runs them: a protocol of your own that does not call it returns
whatever its decoder built. The built-in protocols are in the API reference, with the seam:
[Provider protocol](@ref system_one_protocol_api).

See also: [`UniLM.SystemOneProtocol`](@ref UniLM.SystemOneProtocol),
[`UniLM.system_one_protocol`](@ref UniLM.system_one_protocol), [`UniLM.encode_system_one`](@ref UniLM.encode_system_one),
[`UniLM.decode_system_one`](@ref UniLM.decode_system_one) and [`UniLM.SystemOneWire`](@ref UniLM.SystemOneWire) in
the API reference, and [Write your own backend](@ref custom_backends_guide) for a language-model provider.
