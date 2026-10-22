# ─── Hosted-provider live suites, in one run ─────────────────────────────────
# Every live suite for a hosted provider (OpenAI, Anthropic, Gemini), each file in its own
# testset, against the real APIs: the calls are billed. Needs UNILM_LIVE=1,
# UNILM_RUN_IMAGE_TESTS=true, OPENAI_API_KEY, ANTHROPIC_API_KEY and GEMINI_API_KEY:
#
#     julia --startup-file=no --project=test -e 'include("test/live_hosted.jl")'
#
# The run fails, naming what is wrong, when a flag above does not hold its value or a key is
# unset or empty (a key's value is never printed), when a listed file does not exist, and when
# a file ran no test: a suite whose guard on these variables is not met skips itself. A suite
# with sections for several providers can skip one section and still run tests, which the
# variable check catches. Not listed: the TypeSafe and DeepSeek suites (the release gate passes
# only the three keys above) and the Ollama and MCP suites, which need local servers.

using Test, HTTP, JSON, UniLM   # the suites call HTTP and JSON, as under test/runtests.jl

const _HOSTED_LIVE_FILES = [
    "integration.jl", "integration_tool_loop.jl", "integration_provider_compat.jl",
    "integration_openai_decisions.jl", "integration_openai_audio.jl", "integration_cost.jl",
    "integration_documents.jl", "integration_audio.jl", "integration_images.jl",
    "integration_anthropic.jl", "integration_gemini.jl", "integration_interactions.jl",
    "integration_provider_drift.jl"]

# What each variable the suites' guards read must hold.
const _HOSTED_LIVE_ENV = ("UNILM_LIVE" => ==("1"), "UNILM_RUN_IMAGE_TESTS" => ==("true"),
    "OPENAI_API_KEY" => !isempty, "ANTHROPIC_API_KEY" => !isempty, "GEMINI_API_KEY" => !isempty)

# The tests a finished testset ran (passed, failed or errored), its nested testsets' included.
# A broken or skipped test (@test_broken, @test_skip) does not count.
function _ran(ts::Test.DefaultTestSet)::Int
    c = Test.get_test_counts(ts)
    c.passes + c.fails + c.errors + c.cumulative_passes + c.cumulative_fails + c.cumulative_errors
end

@testset verbose=true "hosted live suites" begin
    # Names only: a key's value never enters a test expression.
    unmet = [name for (name, holds) in _HOSTED_LIVE_ENV if !holds(get(ENV, name, ""))]
    @test isempty(unmet)
    skipped = String[]
    for file in _HOSTED_LIVE_FILES
        path = joinpath(@__DIR__, file)
        isfile(path) || (push!(skipped, "$file: file not found"); continue)
        # Two attempts per call unless a suite passes its own config: a transient failure is
        # retried once.
        ts = @testset "$file" begin
            with_request_config(max_attempts=2) do
                include(path)
            end
        end
        _ran(ts) > 0 || push!(skipped, "$file: no test ran")
    end
    @test isempty(skipped)
end
