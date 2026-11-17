# test/platform_seam_fixtures.jl
using UniLM

# Test rows, not prices: the suite's mock servers reply with these model ids, and a Chat success
# prices the model its reply names, so without a row each id printed the unpriced-model warning.
# A zero rate keeps the suite's output to the warnings a test asserts; the tests of that warning
# use ids of their own.
for id in ("mock", "mock-model")
    DEFAULT_PRICING[id] = UniLM._price(0, 0, 0)
end

# A capability-complete endpoint at a dead port (127.0.0.1:1). With a real config it is
# never reached — a total_deadline of 1e-300 makes _remaining_s resolve to 0.0, so the
# seam throws UniLMTimeout(:deadline) before any connect. If a verb ignores its config,
# the request instead hits the dead port and fails with a connection error (no "timeout"
# text) — that is the discriminator the assertion below relies on.
struct SeamProbe <: UniLM.ServiceEndpoint end
UniLM._api_base_url(::Type{SeamProbe}) = "http://127.0.0.1:1"
UniLM.auth_header(::Type{SeamProbe}) = ["Authorization" => "Bearer test", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{SeamProbe}) = Set([:files, :images, :image_edits, :audio,
    :uploads, :conversations, :vector_stores, :batch, :containers, :moderation,
    :realtime, :speech, :fine_tuning, :system_one, :models, :token_count, :file_registration,
    :webhooks, :safety, :content_provenance, :skills, :context_caching, :file_search_stores,
    :stored_completions, :voices])
UniLM.default_image_model(::Type{SeamProbe}) = "seam-probe-image"

# total_deadline=1e-300 is > 0 (passes RequestConfig validation) yet so small that
# max(total_deadline - elapsed, 0.0) == 0.0 for any realistic elapsed → deterministic
# :deadline breach with no network I/O.
const _TINY_DEADLINE = UniLM.RequestConfig(total_deadline = 1e-300)

# Every platform *CallError carries `error::String`. Short-circuits on the type check, so a
# non-CallError result (Success/Failure without `.error`) fails the @test without erroring.
_reached_seam(r, ::Type{T}) where {T} = (r isa T) && occursin("timeout", lowercase(r.error))

# ...and keeps the typed timeout itself, not only its rendering: `cause` is the
# UniLMTimeout, so a caller can dispatch on phase/elapsed/limit.
_seam_timeout(r, ::Type{T}) where {T} =
    _reached_seam(r, T) && hasproperty(r, :cause) && r.cause isa UniLMTimeout

# ── Scripted local endpoint ──────────────────────────────────────────────────
# A live local endpoint whose base URL is chosen after the listener binds. The
# recorded `target` is the raw origin-form request target, so an id that leaks its
# own `/`, `?` or `#` shows up as extra path segments / a query / a fragment rather
# than as one encoded segment.
using Sockets

struct URLProbe <: UniLM.ServiceEndpoint end
const _url_probe_base = Ref("")
UniLM._resolve_base_url(::Type{URLProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{URLProbe}) =
    ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{URLProbe}) = UniLM.provider_capabilities(SeamProbe)

"A JSON reply, with any extra headers."
_json(status::Int, body::AbstractString; headers::Vector{Pair{String,String}}=Pair{String,String}[]) =
    HTTP.Response(status, ["Content-Type" => "application/json", headers...], Vector{UInt8}(body))

# A request with no body arrives as a sentinel that supports no byte access.
_body_text(req)::String = applicable(copy, req.body) ? String(copy(req.body)) : ""

const _SeenRequest = @NamedTuple{method::String, target::String, body::String}

"""
Run `f()` against URLProbe, answering the n-th request with `respond(n, req)`, and
return `(f(), requests)`: each request as `(method, target, body)`, in arrival order.
Handlers can overlap (a client that gave up on a slow reply may already be sending
the next request), so recording is serialized.
"""
function _with_scripted(f::Function, respond::Function)
    seen = _SeenRequest[]
    guard = ReentrantLock()
    # Held behind an abstractly typed Ref so every scripted server shares ONE handler
    # type: capturing each `respond` closure directly would specialize the server
    # stack anew per test, and that compile time would land inside the timed exchange.
    script = Ref{Function}(respond)
    handler = function (req)
        n = @lock guard begin
            push!(seen, (method=String(req.method), target=String(req.target), body=_body_text(req)))
            length(seen)
        end
        script[](n, req)
    end
    # Probe an ephemeral port, then serve on it; the close-then-rebind window can race.
    for _ in 1:5
        tcp = Sockets.listen(Sockets.localhost, 0)
        port = Int(Sockets.getsockname(tcp)[2])
        close(tcp)
        server = try
            HTTP.serve!(handler, "127.0.0.1", port; verbose=false)
        catch e
            e isa Base.IOError || rethrow()
            continue
        end
        _url_probe_base[] = "http://127.0.0.1:$port"
        try
            return f(), seen
        finally
            close(server)
        end
    end
    error("could not bind an ephemeral port for the scripted endpoint")
end

# An empty JSON object is enough: every platform verb funnels a parse failure into
# its own typed *CallError, and these tests assert on the recorded target only.
"Run `f()` against a local recorder; returns the raw request targets it saw, in order."
_recorded_targets(f::Function) =
    String[r.target for r in last(_with_scripted(f, (_, _) -> _json(200, "{}")))]

"The result of `call()` against an endpoint that answers every request with `status` and `headers`."
_answered(call::Function, status::Int; body::AbstractString="{}",
          headers::Vector{Pair{String,String}}=Pair{String,String}[]) =
    first(_with_scripted(call, (_, _) -> _json(status, body; headers)))

# One id that carries every separator a URL template is built from. Escaped it is a
# single path segment / query value; unescaped it adds segments, a query and a fragment.
const _HOSTILE_ID = "a b/../c?x=1#f"
const _HOSTILE_ENC = "a%20b%2F..%2Fc%3Fx%3D1%23f"

# ── Provider-wire probes at the scripted endpoint ────────────────────────────
# Each speaks one provider's wire for a verb family, with its paths under `_url_probe_base`.

# Files wire `W`.
struct FilesProbe{W} <: UniLM.ServiceEndpoint end
UniLM._files_wire(::Type{FilesProbe{W}}) where {W} = W()
UniLM._files_url(::Type{<:FilesProbe}) = _url_probe_base[] * "/v1/files"
UniLM.auth_header(::Type{<:FilesProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{<:FilesProbe}) = Set([:files])

# Gemini's Files API, its resumable uploads and its registration.
struct GeminiFilesProbe <: UniLM.ServiceEndpoint end
UniLM._files_wire(::Type{GeminiFilesProbe}) = UniLM._GeminiFiles()
UniLM._files_url(::Type{GeminiFilesProbe}) = _url_probe_base[] * "/v1beta/files"
UniLM._files_upload_url(::Type{GeminiFilesProbe}) = _url_probe_base[] * "/upload/v1beta/files"
UniLM.auth_header(::Type{GeminiFilesProbe}) = ["x-goog-api-key" => "t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{GeminiFilesProbe}) = Set([:files, :file_registration])

# Model listing (and lookup) wire `W`.
struct ListingProbe{W} <: UniLM.ServiceEndpoint end
UniLM._model_listing(::Type{ListingProbe{W}}) where {W} = W()
UniLM._models_url(::Type{<:ListingProbe}) = _url_probe_base[] * "/v1/models"
UniLM.auth_header(::Type{<:ListingProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{<:ListingProbe}) = Set([:models, :fine_tuning])

# An endpoint that counts tokens as `service` counts (its wire, its encoder, its headers).
struct _CountAt <: UniLM.ServiceEndpoint
    service::UniLM.ServiceEndpointSpec
end
UniLM.provider_capabilities(s::_CountAt) = UniLM.provider_capabilities(s.service)
UniLM._token_counting(s::_CountAt, request) = UniLM._token_counting(s.service, request)
UniLM.encode_request(s::_CountAt, chat::Chat) = UniLM.encode_request(s.service, chat)
UniLM.encode_agentic(s::_CountAt, r::Respond) = UniLM.encode_agentic(s.service, r)
UniLM._count_headers(w::UniLM._TokenCounting, s::_CountAt) = UniLM._count_headers(w, s.service)
UniLM._count_url(::UniLM._AnthropicCount, ::_CountAt, ::Chat) = _url_probe_base[] * "/count"
UniLM._count_url(::UniLM._GeminiCount, ::_CountAt, ::Chat) = _url_probe_base[] * "/count"
UniLM._count_url(::UniLM._ResponsesCount, ::_CountAt, ::Respond) = _url_probe_base[] * "/count"

# Batch wire `W`: `_batch_probe(W, E)` makes its Chats encode and its replies decode as endpoint `E`'s do.
struct BatchProbe{W} <: UniLM.ServiceEndpoint end
UniLM._batch_wire(::Type{BatchProbe{W}}) where {W} = W()
UniLM._batch_root(::Type{<:BatchProbe}) = _url_probe_base[]
UniLM._resolve_base_url(::Type{<:BatchProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{<:BatchProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{<:BatchProbe}) = Set([:chat, :batch, :files])
UniLM.default_model(::Type{<:BatchProbe}) = "probe-default"
_batch_probe(W, E) = (@eval UniLM.encode_request(::Type{BatchProbe{$W}}, c::Chat) = UniLM.encode_request($E, c);
                      @eval UniLM.decode_response(::Type{BatchProbe{$W}}, r::HTTP.Response) = UniLM.decode_response($E, r);
                      BatchProbe{W})
# The probe of each provider's batch wire, its Chats encoded as that provider's are.
const _OAB = _batch_probe(UniLM._OpenAIBatches, OPENAIServiceEndpoint)
const _ANB = _batch_probe(UniLM._AnthropicBatches, ANTHROPICServiceEndpoint)
const _GMB = _batch_probe(UniLM._GeminiBatches, GEMINIServiceEndpoint)

# A batch endpoint speaking wire W that embeds too, its paths under the scripted server.
struct EmbedBatchProbe{W} <: UniLM.ServiceEndpoint end
UniLM._batch_wire(::Type{EmbedBatchProbe{W}}) where {W} = W()
UniLM._batch_root(::Type{<:EmbedBatchProbe}) = _url_probe_base[]
UniLM._resolve_base_url(::Type{<:EmbedBatchProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{<:EmbedBatchProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{<:EmbedBatchProbe}) = Set([:batch, :embeddings, :files])

# Gemini's embeddings wire at the scripted endpoint.
struct GeminiEmbedProbe <: UniLM.ServiceEndpoint end
UniLM._embeddings_wire(::Type{GeminiEmbedProbe}) = UniLM._GeminiEmbeddings()
UniLM.get_url(::Type{GeminiEmbedProbe}, e::Embeddings) = UniLM._gemini_embed_url(_url_probe_base[] * "/v1beta", e)
UniLM.auth_header(::Type{GeminiEmbedProbe}) = ["x-goog-api-key" => "t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{GeminiEmbedProbe}) = Set([:embeddings])

# The stored-completions wire at the scripted endpoint.
struct StoreProbe <: UniLM.ServiceEndpoint end
UniLM._resolve_base_url(::Type{StoreProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{StoreProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{StoreProbe}) = Set([:stored_completions])

# Skills wire `W` at the scripted endpoint.
struct SkillsProbe{W} <: UniLM.ServiceEndpoint end
UniLM._skills_wire(::Type{SkillsProbe{W}}) where {W} = W()
UniLM._skills_url(::Type{<:SkillsProbe}) = _url_probe_base[] * "/v1/skills"
UniLM.auth_header(::Type{<:SkillsProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{<:SkillsProbe}) = Set([:skills])

# A Gemini endpoint whose cache requests `_with_scripted` serves.
struct CacheProbe <: UniLM.ServiceEndpoint end
UniLM._gemini_api_root(::Type{CacheProbe}) = _url_probe_base[] * "/v1beta"
UniLM.auth_header(::Type{CacheProbe}) = ["x-goog-api-key" => "t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{CacheProbe}) = Set([:context_caching])

# A Gemini endpoint whose File Search requests `_with_scripted` serves.
struct FileSearchProbe <: UniLM.ServiceEndpoint end
UniLM._gemini_api_root(::Type{FileSearchProbe}) = _url_probe_base[] * "/v1beta"
UniLM._gemini_upload_root(::Type{FileSearchProbe}) = _url_probe_base[] * "/upload/v1beta"
UniLM.auth_header(::Type{FileSearchProbe}) = ["x-goog-api-key" => "t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{FileSearchProbe}) = Set([:file_search_stores])

# The voices wire `W` at the scripted endpoint.
struct VoicesProbe{W} <: UniLM.ServiceEndpoint end
UniLM._voices_wire(::Type{VoicesProbe{W}}) where {W} = W()
UniLM._voices_url(::Type{<:VoicesProbe}) = _url_probe_base[] * "/voices"
UniLM._voice_consents_url(::Type{<:VoicesProbe}) = _url_probe_base[] * "/voice_consents"
UniLM.auth_header(::Type{<:VoicesProbe}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{<:VoicesProbe}) = Set([:voices])

# Gemini's transcription served by `_with_scripted`, held to Gemini's rules and default.
struct GeminiTranscribeProbe <: UniLM.ServiceEndpoint end
UniLM._transcription_wire(::Type{GeminiTranscribeProbe}) = UniLM._GeminiTranscriptionWire()
UniLM._gemini_speech_base(::Type{GeminiTranscribeProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{GeminiTranscribeProbe}) = ["x-goog-api-key" => "test-key", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{GeminiTranscribeProbe}) = Set([:transcription])
UniLM._check_transcription(::Type{GeminiTranscribeProbe}, t::UniLM.TranscriptionRequest) =
    UniLM._check_transcription(GEMINIServiceEndpoint, t)
UniLM._default_transcription_model(::Type{GeminiTranscribeProbe}) = UniLM._default_transcription_model(GEMINIServiceEndpoint)
