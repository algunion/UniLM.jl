# Native Gemini translation — deterministic, zero-spend unit tests.
using UniLM
using UniLM: encode_request, decode_response, decode_stream_chunk, StreamState,
             _build_stream_message, GEMINIServiceEndpoint, GPTFunction, GPTToolChoice,
             GEMINI_NATIVE_BASE, RoleSystem, RoleUser, RoleAssistant, RoleTool,
             TOOL_CALLS, STOP, CONTENT_FILTER
using Test, HTTP, JSON

@testset "routing — model in URL, stream branches on method" begin
    chat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash")
    @test UniLM.get_url(chat) == "$(GEMINI_NATIVE_BASE)/models/gemini-3.5-flash:generateContent"
    schat = Chat(service=GEMINIServiceEndpoint, model="gemini-3.5-flash", stream=true)
    @test UniLM.get_url(schat) == "$(GEMINI_NATIVE_BASE)/models/gemini-3.5-flash:streamGenerateContent?alt=sse"
end

@testset "auth — x-goog-api-key" begin
    withenv("GEMINI_API_KEY" => "test-key") do
        h = Dict(UniLM.auth_header(GEMINIServiceEndpoint))
        @test h["x-goog-api-key"] == "test-key"
        @test !haskey(h, "Authorization")            # NOT Bearer
    end
end

@testset "capabilities & default" begin
    @test UniLM.provider_capabilities(GEMINIServiceEndpoint) == Set([:chat, :tools, :streaming])
    @test UniLM.default_model(GEMINIServiceEndpoint) == "gemini-3.5-flash"
    @test_throws ArgumentError UniLM._api_base_url(GEMINIServiceEndpoint)
end
