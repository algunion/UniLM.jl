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
    :uploads, :video, :conversations, :vector_stores, :batch, :containers, :moderation,
    :realtime, :fine_tuning])
UniLM.default_image_model(::Type{SeamProbe}) = "seam-probe-image"

# total_deadline=1e-300 is > 0 (passes RequestConfig validation) yet so small that
# max(total_deadline - elapsed, 0.0) == 0.0 for any realistic elapsed → deterministic
# :deadline breach with no network I/O.
const _TINY_DEADLINE = UniLM.RequestConfig(total_deadline = 1e-300)

# Every platform *CallError carries `error::String`. Short-circuits on the type check, so a
# non-CallError result (Success/Failure without `.error`) fails the @test without erroring.
_reached_seam(r, ::Type{T}) where {T} = (r isa T) && occursin("timeout", lowercase(r.error))

# ── Raw request-target recorder ──────────────────────────────────────────────
# A live local endpoint whose base URL is chosen after the listener binds, used to
# read back the exact request target UniLM put on the wire. `req.target` is the
# unparsed origin-form target on both supported HTTP majors, so an id that leaks
# its own `/`, `?` or `#` shows up here as extra path segments / a query / a
# fragment rather than as one encoded segment.
using Sockets

struct URLProbe <: UniLM.ServiceEndpoint end
const _url_probe_base = Ref("")
UniLM._resolve_base_url(::Type{URLProbe}) = _url_probe_base[]
UniLM.auth_header(::Type{URLProbe}) =
    ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{URLProbe}) = UniLM.provider_capabilities(SeamProbe)

# Probe an ephemeral port, then serve on it; the close-then-rebind window can race.
# An empty JSON object is enough: every platform verb funnels a parse failure into
# its own typed *CallError, and these tests assert on the recorded target only.
"Run `f()` against a local recorder; returns the raw request targets it saw, in order."
function _recorded_targets(f::Function)
    seen = String[]
    handler = req -> (push!(seen, req.target);
                      HTTP.Response(200, ["Content-Type" => "application/json"],
                                    Vector{UInt8}("{}")))
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
            f()
        finally
            close(server)
        end
        return seen
    end
    error("could not bind an ephemeral port for the request-target recorder")
end

# One id that carries every separator a URL template is built from. Escaped it is a
# single path segment / query value; unescaped it adds segments, a query and a fragment.
const _HOSTILE_ID = "a b/../c?x=1#f"
const _HOSTILE_ENC = "a%20b%2F..%2Fc%3Fx%3D1%23f"
