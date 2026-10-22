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

# What OpenAI bills a gpt-audio-1.5 call for its usage (https://developers.openai.com/api/docs/pricing,
# USD per 1M tokens): text $2.50 input and $10 output, audio $32 input and $64 output, no cached-input rate.
_audio_live_bill(u::TokenUsage) = ((u.prompt_tokens - u.prompt_audio_tokens) * 2.5 + u.prompt_audio_tokens * 32.0 +
    (u.completion_tokens - u.completion_audio_tokens) * 10.0 + u.completion_audio_tokens * 64.0) / 1e6

if !haskey(ENV, "OPENAI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping OpenAI Chat audio integration tests (set UNILM_LIVE=1 and OPENAI_API_KEY to run live)"
else

HTTP.close_idle_connections!()

@testset "OpenAI Chat audio — a spoken reply is captured, and the next turn sends it back by id" begin
    chat = _audio_live_chat(modalities=["text", "audio"], audio=Dict("voice" => "alloy", "format" => "wav"))
    # The audio model is priced: a billed call logs no warning, and its estimate is OpenAI's bill.
    r = @test_logs min_level=Logging.Warn chatrequest!(chat; config=_AUDIO_LIVE_CFG)
    @test r isa LLMSuccess
    a = r.message.audio_output
    @test a isa AudioOutput && startswith(a.id, "audio_") && a.expires_at > time()
    @test a.format == "wav" && length(a.data) > 44 && a.data[1:4] == b"RIFF" && a.data[9:12] == b"WAVE"
    @test !isempty(strip(a.transcript)) && r.message.content == a.transcript
    @info "Chat audio live reply" id = a.id bytes = length(a.data) transcript = a.transcript usage = r.usage
    # The spoken reply is audio output, so part of its completion is billed at the audio rate.
    @test 0 < r.usage.completion_audio_tokens <= r.usage.completion_tokens
    @test estimated_cost(r) > 0 && isapprox(estimated_cost(r), _audio_live_bill(r.usage); rtol=1e-12)
    @test cumulative_cost(chat) == estimated_cost(r)

    # The audio turn goes back as its transcript and its id; the follow-up is answered in text.
    follow = Chat(model=_AUDIO_LIVE_MODEL, max_completion_tokens=20, messages=copy(r.self.messages))
    push!(follow, Message(Val(:user), _AUDIO_LIVE_ASK))
    sent = JSON.parse(UniLM.encode_request(OPENAIServiceEndpoint, follow))
    @test sent["messages"][3] == Dict("role" => "assistant", "content" => a.transcript, "audio" => Dict("id" => a.id))
    f = @test_logs min_level=Logging.Warn chatrequest!(follow; config=_AUDIO_LIVE_CFG)
    @test f isa LLMSuccess
    @test !isempty(strip(something(f.message.content, ""))) && isnothing(f.message.audio_output)
    @info "Chat audio live follow-up" text = f.message.content usage = f.usage
    # The answer names the word the spoken reply said.
    said = [m.match for m in eachmatch(r"\p{L}+", a.transcript)]
    @test !isempty(said) && all(w -> occursin(Regex("\\b$w\\b", "i"), something(f.message.content, "")), said)
    # The audio sent back by its id is audio input; the text answer has no audio output.
    @test 0 < f.usage.prompt_audio_tokens <= f.usage.prompt_tokens && f.usage.completion_audio_tokens == 0
    @test estimated_cost(f) > 0 && isapprox(estimated_cost(f), _audio_live_bill(f.usage); rtol=1e-12)

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
    r = @test_logs min_level=Logging.Warn fetch(chatrequest!(chat; config=_AUDIO_LIVE_CFG,
        callback=(c, _) -> c isa String && push!(heard, c)))
    @test r isa LLMSuccess
    a = r.message.audio_output
    @test a isa AudioOutput && startswith(a.id, "audio_") && a.expires_at > time()
    @test isnothing(a.format) && !isempty(a.data) && iseven(length(a.data))      # pcm16: raw samples, no header
    @test !isempty(strip(a.transcript)) && r.message.content == a.transcript == join(heard)
    @test r.usage isa TokenUsage && r.usage.completion_tokens > 0
    @info "Chat audio live stream" id = a.id bytes = length(a.data) transcript = a.transcript usage = r.usage
    # The streamed usage counts the audio as the reply does unstreamed, and is billed the same way.
    @test 0 < r.usage.completion_audio_tokens <= r.usage.completion_tokens
    @test estimated_cost(r) > 0 && isapprox(estimated_cost(r), _audio_live_bill(r.usage); rtol=1e-12)
    @test cumulative_cost(chat) == estimated_cost(r)
end

end
