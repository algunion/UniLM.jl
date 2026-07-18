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
