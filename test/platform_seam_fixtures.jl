# test/platform_seam_fixtures.jl
using UniLM

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
    :realtime, :fine_tuning, :system_one, :models])
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
