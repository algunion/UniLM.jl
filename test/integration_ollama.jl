# ─── Ollama + Gemma 4 live witnesses ─────────────────────────────────────────
# Requires UNILM_LIVE_OLLAMA=1 and a reachable Ollama server (OLLAMA_HOST, else
# 127.0.0.1:11434) with the chat model UNILM_OLLAMA_MODEL (default gemma4:e4b) and the
# embedding model UNILM_OLLAMA_EMBED_MODEL (default embeddinggemma); the FIM witness also
# needs UNILM_OLLAMA_FIM_MODEL (default qwen2.5-coder:1.5b-base) and skips without it.
# Local models: no spend. Each witness names the outcome that would refute it. Sampling
# witnesses run at temperature 0 (greedy, so repeatable on one machine) unless they
# measure the model's own default behaviour.

using Sockets

const _OLLAMA_MODEL = get(ENV, "UNILM_OLLAMA_MODEL", "gemma4:e4b")
const _OLLAMA_EMBED = get(ENV, "UNILM_OLLAMA_EMBED_MODEL", "embeddinggemma")
const _OLLAMA_FIM = get(ENV, "UNILM_OLLAMA_FIM_MODEL", "qwen2.5-coder:1.5b-base")
const _OLLAMA_MEDIA = joinpath(@__DIR__, "fixtures", "ollama")

# The server's installed tags, or `nothing` when it does not answer.
function _ollama_installed(ep::OllamaEndpoint)
    try
        r = HTTP.get(ep.base_url * "/api/tags"; readtimeout=5, connect_timeout=5, retry=false)
        Set(String(m["name"]) for m in JSON.parse(r.body)["models"])
    catch
        nothing
    end
end
_has(tags, m) = m in tags || (m * ":latest") in tags

# The server's loaded models, from /api/ps: name => context length.
_ollama_ps(ep::OllamaEndpoint) =
    Dict(String(m["name"]) => get(m, "context_length", nothing)
         for m in JSON.parse(HTTP.get(ep.base_url * "/api/ps"; retry=false).body)["models"])

const _OLLAMA_EP = OllamaEndpoint()
const _OLLAMA_TAGS = get(ENV, "UNILM_LIVE_OLLAMA", "") == "1" ? _ollama_installed(_OLLAMA_EP) : nothing

if isnothing(_OLLAMA_TAGS) || !_has(_OLLAMA_TAGS, _OLLAMA_MODEL) || !_has(_OLLAMA_TAGS, _OLLAMA_EMBED)
    @info "Skipping Ollama live witnesses (set UNILM_LIVE_OLLAMA=1, run an Ollama server and pull " *
          "$_OLLAMA_MODEL and $_OLLAMA_EMBED)"
else

_lchat(; ep=_OLLAMA_EP, system="You are concise.", kw...) =
    (c = Chat(; service=ep, model=_OLLAMA_MODEL, kw...); push!(c, Message(Val(:system), system)); c)
_ask(c::Chat, q; kw...) = (push!(c, Message(Val(:user), q)); chatrequest!(c; kw...))

const _WEATHER_TOOL = UniLM.CallableTool(
    Tool(func=FunctionSignature(name="get_weather", description="Current weather in a city",
        parameters=Dict("type" => "object", "properties" => Dict("city" => Dict("type" => "string")),
                        "required" => ["city"]))),
    (name, args) -> "18°C and sunny in $(args["city"])")

@testset "Ollama live — $_OLLAMA_MODEL" begin

@testset "text, history and cost" begin
    # Refuted by: a non-success, a missing usage, a price, or a missing-price warning.
    c = _lchat(temperature=0.0)
    r = @test_logs _ask(c, "Reply with the single word: pong")
    @test r isa LLMSuccess
    @test occursin("pong", lowercase(text(r)))
    @test r.usage.prompt_tokens > 0 && r.usage.completion_tokens > 0
    @test estimated_cost(r) == 0.0 && cumulative_cost(c) == 0.0
    # The conversation carries across turns.
    _ask(c, "Remember this number: 4817. Reply with OK.")
    r = _ask(c, "Which number did I ask you to remember? Digits only.")
    @test occursin("4817", text(r))
end

@testset "streaming: deltas, one turn, usage" begin
    # Refuted by: fewer than two deltas, deltas that do not add up to the text, no usage.
    c = _lchat(stream=true, temperature=0.0, reasoning_effort="none")
    push!(c, Message(Val(:user), "Count from 1 to 8, comma separated."))
    deltas = String[]
    r = fetch(chatrequest!(c; callback=(x, _) -> x isa String && push!(deltas, x)))
    @test r isa LLMSuccess
    @test length(deltas) >= 2
    @test join(deltas) == text(r)
    @test occursin("1, 2, 3", text(r))
    @test !isnothing(r.usage) && r.usage.completion_tokens > 0
    @test r.message.finish_reason == "stop"
end

@testset "thinking: visible when on, absent when off, sent back with its turn" begin
    q = "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. " *
        "How much does the ball cost? Answer with the number only."
    # Refuted by: no reasoning text at temperature 0 on a prompt this model thinks about.
    c = _lchat(reasoning_effort="low", temperature=0.0)
    r = _ask(c, q)
    @test r isa LLMSuccess
    @test !isnothing(reasoning_text(r)) && length(reasoning_text(r)) > 20
    @test occursin("0.05", text(r))
    # Streamed, the thinking is captured the same way.
    s = fetch(chatrequest!(Chat(service=_OLLAMA_EP, model=_OLLAMA_MODEL, reasoning_effort="low",
                                temperature=0.0, stream=true, messages=c.messages[1:2])))
    @test s isa LLMSuccess && !isnothing(reasoning_text(s))
    # The next turn goes out with the captured thinking and still answers.
    r2 = _ask(c, "And the bat? Number only.")
    @test r2 isa LLMSuccess && occursin("1.05", text(r2))
    # Refuted by: reasoning text with thinking off.
    off = _ask(_lchat(reasoning_effort="none", temperature=0.0), q)
    @test off isa LLMSuccess && isnothing(reasoning_text(off))
end

@testset "tools: one call, a full tool loop, streamed calls" begin
    # Refuted by: no call, the wrong call or arguments, or a loop that does not finish.
    c = _lchat(tools=[_WEATHER_TOOL], temperature=0.0)
    r = _ask(c, "What's the weather in Paris? Use the tool.")
    @test r isa LLMSuccess
    call = only(r.message.tool_calls)
    @test call.func.name == "get_weather" && occursin("Paris", call.func.arguments["city"])
    @test startswith(call.id, "call_")                       # Ollama names its calls
    @test r.message.finish_reason == "tool_calls"
    lc = _lchat(tools=[_WEATHER_TOOL], temperature=0.0)
    push!(lc, Message(Val(:user), "What's the weather in Lyon? Use the tool, then answer in one sentence."))
    lr = tool_loop!(lc; tools=[_WEATHER_TOOL])
    @test lr.completed && lr.turns_used == 2
    @test only(lr.tool_calls).success && occursin("Lyon", only(lr.tool_calls).arguments["city"])
    @test occursin("18", text(lr.response))
    fired = ToolCall[]
    sc = _lchat(tools=[_WEATHER_TOOL], temperature=0.0, stream=true)
    push!(sc, Message(Val(:user), "What's the weather in Rome? Use the tool."))
    s = fetch(chatrequest!(sc; on_tool_call=tc -> push!(fired, tc)))
    @test s isa LLMSuccess && length(fired) == 1
    @test occursin("Rome", only(fired).func.arguments["city"]) && only(fired).id == only(s.message.tool_calls).id
end

@testset "structured output is enforced (thinking off with a format)" begin
    schema = Dict("type" => "object", "additionalProperties" => false, "required" => ["name", "year", "city"],
                  "properties" => Dict("name" => Dict("type" => "string"), "year" => Dict("type" => "integer"),
                                       "city" => Dict("type" => "string")))
    # Refuted by: any reply that is not the schema's JSON — the default (thinking-on) request
    # failed 10 of 10 on gemma4:e4b before the format turned thinking off. Default sampling.
    for _ in 1:5
        r = _ask(_lchat(response_format=UniLM.json_schema("person", "A person", schema)),
                 "Extract the person: Ada Lovelace, born 1815 in London.")
        d = JSON.parse(text(r))
        @test d["name"] == "Ada Lovelace" && d["year"] == 1815 && d["city"] == "London"
    end
    for _ in 1:3
        r = _ask(_lchat(response_format=UniLM.json_object()), "Give a JSON object with keys a and b set to 1 and 2.")
        @test JSON.parse(text(r)) isa AbstractDict
    end
end

@testset "vision and audio attachments" begin
    img = ImageAttachment(joinpath(_OLLAMA_MEDIA, "red_circle.png"))
    wav = AudioAttachment(joinpath(_OLLAMA_MEDIA, "secret_word.wav"))
    # Refuted by: an answer that does not name the colour or the spoken word.
    for _ in 1:3
        c = _lchat(temperature=0.0)
        push!(c, Message(Val(:user), "What colour is the shape? One word.", img))
        @test occursin("red", lowercase(text(chatrequest!(c))))
        c = _lchat(temperature=0.0)
        push!(c, Message(Val(:user), "What is the secret word in this recording? One word.", wav))
        @test occursin("banana", lowercase(text(chatrequest!(c))))
    end
end

@testset "context window: num_ctx loads the model with it; a prompt that does not fit fails" begin
    # Refuted by: a runner context other than the one asked for, or a 200 for an over-long prompt.
    small = OllamaEndpoint(num_ctx=4096)
    @test _ask(_lchat(ep=small, temperature=0.0, reasoning_effort="none"), "Say OK.") isa LLMSuccess
    @test _ollama_ps(small)[_OLLAMA_MODEL] == 4096
    long = repeat("The logistics report covered warehouses, fleets and suppliers in detail. ", 450)  # > 4096 tokens
    f = _ask(_lchat(ep=small, reasoning_effort="none"), long * "\nWhat was the report about?";
             config=RequestConfig(max_attempts=1))
    @test f isa LLMFailure && f.status == 400 && occursin("exceeds the available context size", f.response)
    # truncate=true is Ollama's own behaviour: the prompt is shortened and the call succeeds.
    lossy = OllamaEndpoint(num_ctx=4096, truncate=true)
    @test _ask(_lchat(ep=lossy, reasoning_effort="none"), long * "\nWhat was the report about?") isa LLMSuccess
end

@testset "keep_alive: 0 unloads the model after the call" begin
    # Refuted by: the model still loaded 10 s after a keep_alive=0 request.
    r = _ask(_lchat(ep=OllamaEndpoint(keep_alive=0), reasoning_effort="none"), "Say OK.")
    @test r isa LLMSuccess
    @test timedwait(() -> !haskey(_ollama_ps(_OLLAMA_EP), _OLLAMA_MODEL), 10.0) == :ok
end

@testset "embeddings: native vectors, over-long input fails" begin
    # Refuted by: wrong dimension, similar texts not closer than unrelated ones, or silent truncation.
    e = embeddingrequest!(Embeddings(["cats purr", "kittens meow", "stock markets fell"];
                                     service=_OLLAMA_EP, model=_OLLAMA_EMBED))
    @test e isa EmbeddingSuccess
    v = embedding_vectors(e)
    cosine(a, b) = sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))
    @test length(v) == 3 && all(x -> length(x) == length(v[1]) > 100, v)
    @test cosine(v[1], v[2]) > cosine(v[1], v[3])
    @test estimated_cost(e) == 0.0
    f = embeddingrequest!(Embeddings(repeat("word ", 5000); service=_OLLAMA_EP, model=_OLLAMA_EMBED);
                          config=RequestConfig(max_attempts=1))
    @test f isa EmbeddingFailure && f.status == 400
    ok = embeddingrequest!(Embeddings(repeat("word ", 5000); service=OllamaEndpoint(truncate=true), model=_OLLAMA_EMBED))
    @test ok isa EmbeddingSuccess
end

@testset "respond over /v1/responses" begin
    # Refuted by: no text, or an image the model does not see.
    r = respond(Respond(service=_OLLAMA_EP, model=_OLLAMA_MODEL, input="Say hello in French, one word.",
                        reasoning=Reasoning(effort="none"), temperature=0.0))
    @test r isa ResponseSuccess && occursin("bonjour", lowercase(output_text(r)))
    png = UniLM.Base64.base64encode(read(joinpath(_OLLAMA_MEDIA, "red_circle.png")))
    ri = respond(Respond(service=_OLLAMA_EP, model=_OLLAMA_MODEL, reasoning=Reasoning(effort="none"), temperature=0.0,
                         input=[InputMessage(role="user", content=[input_text("What colour is the shape? One word."),
                                                                  input_image("data:image/png;base64," * png)])]))
    @test ri isa ResponseSuccess && occursin("red", lowercase(output_text(ri)))
end

@testset "fill-in-the-middle with a code model" begin
    if !_has(_OLLAMA_TAGS, _OLLAMA_FIM)
        @info "Skipping the Ollama FIM witness: $_OLLAMA_FIM is not installed"
    else
        # Refuted by: a non-success or an empty completion.
        r = fim_complete(FIMCompletion(service=_OLLAMA_EP, model=_OLLAMA_FIM, prompt="def add(a, b):\n    return ",
                                       suffix="\n\nprint(add(1, 2))\n", max_tokens=16, temperature=0.0))
        @test r isa FIMSuccess && !isempty(strip(fim_text(r)))
    end
end

@testset "failures are typed and say what to do" begin
    # Refuted by: a success for a model that is not installed, or an unreachable server without the hint.
    r = _ask(Chat(service=_OLLAMA_EP, model="gemma4:no-such-tag") |> c -> (push!(c, Message(Val(:system), "s")); c), "hi")
    @test r isa LLMFailure && r.status == 404 && occursin("not found", r.response)
    tcp = Sockets.listen(Sockets.localhost, 0); port = Int(Sockets.getsockname(tcp)[2]); close(tcp)
    down = _ask(_lchat(ep=OllamaEndpoint(base_url="http://127.0.0.1:$port")), "hi"; config=RequestConfig(max_attempts=1))
    @test down isa LLMCallError && occursin("ollama serve", down.error)
end

@testset "concurrent calls all complete" begin
    # Refuted by: any non-success among concurrent chats and streams (Ollama serializes them).
    base = _lchat(temperature=0.0, reasoning_effort="none")
    push!(base, Message(Val(:user), "Say OK."))
    chats = [Threads.@spawn(chatrequest!(fork(base))) for _ in 1:6]
    streams = [chatrequest!(Chat(service=_OLLAMA_EP, model=_OLLAMA_MODEL, stream=true, temperature=0.0,
                                 reasoning_effort="none", messages=copy(base.messages))) for _ in 1:3]
    @test all(t -> fetch(t) isa LLMSuccess, chats)
    @test all(t -> fetch(t) isa LLMSuccess, streams)
end

@testset "record once, replay offline" begin
    # Refuted by: a replay that differs from the recorded answer, or one that needs the server.
    dir = mktempdir()
    c = _lchat(temperature=0.0, reasoning_effort="low")
    recorded = with_recorded_answers(dir; mode=:record) do
        _ask(c, "Is 391 prime? Answer yes or no.")
    end
    offline = OllamaEndpoint(base_url="http://127.0.0.1:1")
    replayed = with_recorded_answers(dir; mode=:replay) do
        rc = Chat(service=offline, model=_OLLAMA_MODEL, temperature=0.0, reasoning_effort="low",
                  messages=c.messages[1:2])
        chatrequest!(rc)
    end
    @test text(replayed) == text(recorded)
    @test reasoning_text(replayed) == reasoning_text(recorded)
end

end # Ollama live
end # guard
