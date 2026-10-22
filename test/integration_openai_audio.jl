# ─── OpenAI Chat audio replies — live ────────────────────────────────────────
# Requires UNILM_LIVE=1 and OPENAI_API_KEY (both, or the suite skips). Four calls on
# gpt-audio-1.5 ($2.50 / $10 per 1M text tokens, $32 / $64 per 1M audio tokens): a
# one-word spoken reply, streamed and not, each capped at 100 completion tokens; a text
# follow-up that sends the first reply back by its id; and a follow-up naming an id
# OpenAI never issued, which it refuses with HTTP 400. Under two cents at the caps.

using Logging

const _AUDIO_LIVE_CFG = RequestConfig(max_attempts=1, total_deadline=120.0)   # one attempt: the spend stays bounded
const _AUDIO_LIVE_MODEL = "gpt-audio-1.5"
const _AUDIO_LIVE_ASK = "Which word did you just say? Answer in text."

_audio_live_chat(; kwargs...) = Chat(; model=_AUDIO_LIVE_MODEL, max_completion_tokens=100,
    messages=[Message(Val(:system), "Reply with exactly one word."), Message(Val(:user), "Say hi.")], kwargs...)

if !haskey(ENV, "OPENAI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping OpenAI Chat audio integration tests (set UNILM_LIVE=1 and OPENAI_API_KEY to run live)"
else

HTTP.close_idle_connections!()

@testset "OpenAI Chat audio — a spoken reply is captured, and the next turn sends it back by id" begin
    chat = _audio_live_chat(modalities=["text", "audio"], audio=Dict("voice" => "alloy", "format" => "wav"))
    # UniLM prices text tokens only: an audio model has no price row, which each billed call says.
    r = @test_logs (:warn, r"no price row") min_level=Logging.Warn chatrequest!(chat; config=_AUDIO_LIVE_CFG)
    @test r isa LLMSuccess
    a = r.message.audio_output
    @test a isa AudioOutput && startswith(a.id, "audio_") && a.expires_at > time()
    @test a.format == "wav" && length(a.data) > 44 && a.data[1:4] == b"RIFF" && a.data[9:12] == b"WAVE"
    @test !isempty(strip(a.transcript)) && r.message.content == a.transcript
    @info "Chat audio live reply" id = a.id bytes = length(a.data) transcript = a.transcript usage = r.usage

    # The audio turn goes back as its transcript and its id; the follow-up is answered in text.
    follow = Chat(model=_AUDIO_LIVE_MODEL, max_completion_tokens=20, messages=copy(r.self.messages))
    push!(follow, Message(Val(:user), _AUDIO_LIVE_ASK))
    sent = JSON.parse(UniLM.encode_request(OPENAIServiceEndpoint, follow))
    @test sent["messages"][3] == Dict("role" => "assistant", "content" => a.transcript, "audio" => Dict("id" => a.id))
    f = @test_logs (:warn, r"no price row") min_level=Logging.Warn chatrequest!(follow; config=_AUDIO_LIVE_CFG)
    @test f isa LLMSuccess
    @test !isempty(strip(something(f.message.content, ""))) && isnothing(f.message.audio_output)
    @info "Chat audio live follow-up" text = f.message.content usage = f.usage

    # The server resolves the id it is sent: one it never issued is a 400.
    unknown = AudioOutput(id="audio_unilm_never_issued", data=UInt8[], expires_at=0, transcript=a.transcript)
    bogus = Chat(model=_AUDIO_LIVE_MODEL, max_completion_tokens=20, messages=[chat.messages[1:2]...,
        Message(role=RoleAssistant, audio_output=unknown), Message(Val(:user), _AUDIO_LIVE_ASK)])
    b = chatrequest!(bogus; config=_AUDIO_LIVE_CFG)
    @test b isa LLMFailure && b.status == 400
    @info "Chat audio live unknown id" status = (b isa LLMFailure ? b.status : nothing) body = (b isa LLMFailure ? b.response : b)
end

@testset "OpenAI Chat audio — a streamed spoken reply assembles from its chunks" begin
    chat = _audio_live_chat(stream=true, modalities=["text", "audio"], audio=Dict("voice" => "alloy", "format" => "pcm16"))
    heard = String[]
    r = @test_logs (:warn, r"no price row") min_level=Logging.Warn fetch(chatrequest!(chat; config=_AUDIO_LIVE_CFG,
        callback=(c, _) -> c isa String && push!(heard, c)))
    @test r isa LLMSuccess
    a = r.message.audio_output
    @test a isa AudioOutput && startswith(a.id, "audio_") && a.expires_at > time()
    @test isnothing(a.format) && !isempty(a.data) && iseven(length(a.data))      # pcm16: raw samples, no header
    @test !isempty(strip(a.transcript)) && r.message.content == a.transcript == join(heard)
    @test r.usage isa TokenUsage && r.usage.completion_tokens > 0
    @info "Chat audio live stream" id = a.id bytes = length(a.data) transcript = a.transcript usage = r.usage
end

end
