# ─── Hosted-provider live suites, in one run ─────────────────────────────────
# Every live suite for a hosted provider (OpenAI, Anthropic, Gemini), each file in its own
# testset, against the real APIs: the calls are billed. Needs UNILM_LIVE=1,
# UNILM_RUN_IMAGE_TESTS=true, OPENAI_API_KEY, ANTHROPIC_API_KEY and GEMINI_API_KEY:
#
#     julia --startup-file=no --project=test -e 'include("test/live_hosted.jl")'
#
# A suite whose guard on these variables is not met skips itself and records no test, so the
# run fails for every listed file that recorded none, and for one that does not exist. Not
# listed: the TypeSafe and DeepSeek suites (the release gate passes only the three keys above)
# and the Ollama and MCP suites, which need local servers.

using Test, HTTP, JSON, UniLM   # the suites call HTTP and JSON, as under test/runtests.jl

const _HOSTED_LIVE_FILES = [
    "integration.jl", "integration_tool_loop.jl", "integration_provider_compat.jl",
    "integration_openai_decisions.jl", "integration_openai_audio.jl", "integration_cost.jl",
    "integration_documents.jl", "integration_audio.jl", "integration_images.jl",
    "integration_anthropic.jl", "integration_gemini.jl", "integration_interactions.jl",
    "integration_provider_drift.jl"]

# The results a finished testset recorded, its nested testsets' included.
function _recorded(ts::Test.DefaultTestSet)::Int
    c = Test.get_test_counts(ts)
    c.passes + c.fails + c.errors + c.broken +
        c.cumulative_passes + c.cumulative_fails + c.cumulative_errors + c.cumulative_broken
end

@testset "hosted live suites" begin
    skipped = String[]
    for file in _HOSTED_LIVE_FILES
        path = joinpath(@__DIR__, file)
        isfile(path) || (push!(skipped, "$file: file not found"); continue)
        # Two attempts per call: a transient failure is retried once.
        ts = @testset "$file" begin
            with_request_config(max_attempts=2) do
                include(path)
            end
        end
        _recorded(ts) > 0 || push!(skipped, "$file: no test recorded")
    end
    @test isempty(skipped)
end
