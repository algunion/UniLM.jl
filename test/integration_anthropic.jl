# ─── Anthropic Integration Tests (live) ──────────────────────────────────────
# Requires UNILM_LIVE=1 and ANTHROPIC_API_KEY. Minimal tokens on claude-haiku-5-5, the cheapest
# current Claude model ($0.10 input / $0.50 output per 1M tokens); one short call on
# claude-sonnet-5-5 for the rule that is its own, a thinking tool round-trip on
# claude-sonnet-5, which thinks before the tool call it chooses, and a 16-token fast-mode call
# on claude-opus-5-5, the model that serves it.

if !haskey(ENV, "ANTHROPIC_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping Anthropic integration tests (set UNILM_LIVE=1 and ANTHROPIC_API_KEY to run live)"
else

const _LIVE_CLAUDE = "claude-haiku-5-5"
const _LIVE_WEATHER = Tool(func=FunctionSignature(name="get_current_weather",
    description="Get the current weather for a location",
    parameters=Dict("type" => "object",
        "properties" => Dict("location" => Dict("type" => "string", "description" => "City name")),
        "required" => ["location"])))
function _live_claude(user; system="You are a helpful assistant.", kw...)
    chat = Chat(; service=ANTHROPICServiceEndpoint, model=_LIVE_CLAUDE, kw...)
    push!(chat, Message(Val(:system), system))
    push!(chat, Message(Val(:user), user))
    chat
end
_block_types(m::Message) = [b["type"] for b in m.provider_content.blocks]
# One log line per billed call: the served model, tokens in and out, the estimated cost.
_billed(r) = (r isa LLMSuccess && @info "billed" r.model r.usage.prompt_tokens r.usage.completion_tokens cost = estimated_cost(r); r)

@testset "Anthropic Chat — a basic call reports its served model and is priced at its row" begin
    # The region is pinned, so the usage reports "global" and the 1x rates whatever the workspace default.
    chat = _live_claude("Reply with exactly: hello"; service=ANTHROPICServiceEndpoint(inference_geo="global"),
                        max_tokens=64, reasoning_effort="none")
    @test JSON.parse(UniLM.encode_request(chat.service, chat))["inference_geo"] == "global"
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    @test startswith(something(r.model, ""), _LIVE_CLAUDE)
    u = r.usage
    @test u.prompt_tokens > 0 && u.completion_tokens > 0
    @test (u.cached_tokens, u.cache_write_tokens, u.service_tier, u.inference_geo) == (0, 0, "standard", "global")
    # claude-haiku-5-5: $0.10 per 1M input tokens, $0.50 per 1M output tokens.
    @test isapprox(estimated_cost(r), (u.prompt_tokens * 0.10 + u.completion_tokens * 0.50) / 1e6; rtol=1e-12)
    @test cumulative_cost(chat) == estimated_cost(r)
end

@testset "Anthropic Chat — a forced tool call, then its result" begin
    chat = _live_claude("What is the weather in Paris?"; system="Use the weather tool when asked about weather.",
                        max_tokens=256, tools=[_LIVE_WEATHER], tool_choice=UniLM.GPTToolChoice(func="get_current_weather"))
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    m = r.message
    @test m.finish_reason == UniLM.TOOL_CALLS
    @test [tc.func.name for tc in m.tool_calls] == ["get_current_weather"]
    @test haskey(m.tool_calls[1].func.arguments, "location")
    push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="22C and sunny"))
    @test _billed(chatrequest!(chat)) isa LLMSuccess
end

@testset "Anthropic Chat — a streamed call reports the usage of the same call unstreamed" begin
    call(stream) = (chat = _live_claude("Count from 1 to 5, separated by spaces."; max_tokens=64,
                                        reasoning_effort="none", stream);
                    _billed(stream ? fetch(chatrequest!(chat; callback=(_, _) -> nothing)) : chatrequest!(chat)))
    plain, streamed = call(false), call(true)
    @test plain isa LLMSuccess && streamed isa LLMSuccess
    @test streamed.usage.prompt_tokens == plain.usage.prompt_tokens
    @test (streamed.usage.service_tier, streamed.usage.inference_geo) == (plain.usage.service_tier, plain.usage.inference_geo)
    @test streamed.model == plain.model
    @test !isempty(streamed.message.content)
end

@testset "Anthropic Chat — tool turns go back verbatim after the tool result" begin
    # Claude Haiku 5.5 answers a forced tool_choice with the tool call and no thinking block
    # (https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide.md); the captured
    # blocks are what the follow-up sends back.
    chat = _live_claude("What is the weather in Oslo?"; system="Use the weather tool when asked about weather.",
                        max_tokens=256, reasoning_effort="low", tools=[_LIVE_WEATHER],
                        tool_choice=UniLM.GPTToolChoice(func="get_current_weather"))
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    m = r.message
    @test m.finish_reason == UniLM.TOOL_CALLS && _block_types(m) == ["tool_use"]
    push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="4C and cloudy"))
    body = JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["messages"][2]["content"] == m.provider_content.blocks
    @test _billed(chatrequest!(chat)) isa LLMSuccess
end

@testset "Anthropic Chat — signed thinking blocks survive a tool round-trip (claude-sonnet-5)" begin
    # A thinking model that chooses the tool: Claude Haiku 5.5 called it without thinking in
    # three of three tries at effort "high" (2026-10-22); Claude Sonnet 5 thinks, then calls it.
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-sonnet-5", max_tokens=1024, reasoning_effort="high",
                tools=[_LIVE_WEATHER], tool_choice="auto")
    push!(chat, Message(Val(:system), "Always call the weather tool for a weather question; think first."))
    push!(chat, Message(Val(:user), "Is it colder in Paris right now than the October average? Use the tool."))
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    m = r.message
    # Thinking first, then the tool call; the model may add text or further blocks between them.
    types = _block_types(m)
    @test m.finish_reason == UniLM.TOOL_CALLS && first(types) == "thinking" && "tool_use" in types[2:end]
    @test !isempty(m.provider_content.blocks[1]["signature"])
    push!(chat, Message(role=UniLM.RoleTool, tool_call_id=m.tool_calls[1].id, content="9C, light rain"))
    body = JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["messages"][2]["content"] == m.provider_content.blocks
    @test _billed(chatrequest!(chat)) isa LLMSuccess
end

@testset "Anthropic Chat — a turn cut at max_tokens with thinking only goes back as returned" begin
    chat = _live_claude("Is 1009 prime? Check every candidate divisor carefully before you answer.";
                        system="You are terse.", max_tokens=16, reasoning_effort="high")
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    @test r.message.finish_reason == "length" && r.message.content == "" && _block_types(r.message) == ["thinking"]
    push!(chat, Message(Val(:user), "Answer with yes or no only."))
    follow = Chat(service=ANTHROPICServiceEndpoint, model=_LIVE_CLAUDE, max_tokens=256, reasoning_effort="low",
                  messages=chat.messages)
    body = JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, follow))
    @test [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    @test _billed(chatrequest!(follow)) isa LLMSuccess
end

@testset "Anthropic Chat — a system message pushed after a reply, then a user turn" begin
    # Claude Haiku 5.5 rejects a thinking block whose prefix changed; the system message goes
    # after the next user turn, never into the top-level system prompt.
    chat = _live_claude("Is 391 prime? Think it through, then answer yes or no."; system="You are terse.",
                        max_tokens=1024, reasoning_effort="high")
    r1 = _billed(chatrequest!(chat))
    @test r1 isa LLMSuccess
    @test "thinking" in _block_types(r1.message)          # a block for the prefix check to check
    push!(chat, Message(Val(:system), "From now on, answer in French."))
    push!(chat, Message(Val(:user), "Is 397 prime?"))
    body = JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))
    @test body["system"] == "You are terse." && body["messages"][end]["role"] == "system"
    @test _billed(chatrequest!(chat)) isa LLMSuccess
end

@testset "Anthropic Chat — Claude Sonnet 5.5 turns up-front thinking off with between_tools" begin
    chat = Chat(service=ANTHROPICServiceEndpoint, model="claude-sonnet-5-5", max_tokens=16, reasoning_effort="none")
    push!(chat, Message(Val(:system), "You are terse."))
    push!(chat, Message(Val(:user), "Reply with exactly: hello"))
    @test JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))["thinking"] == Dict("type" => "between_tools")
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    @test startswith(something(r.model, ""), "claude-sonnet-5-5")
end

@testset "Anthropic Chat — refused locally, before any request" begin
    for (model, kw) in ((_LIVE_CLAUDE, (temperature=0.2,)),
                        ("claude-sonnet-5-5", (tools=[_LIVE_WEATHER], tool_choice="required")),
                        (_LIVE_CLAUDE, (service_tier="fast",)),
                        (_LIVE_CLAUDE, (prompt_cache_options=PromptCacheOptions(ttl="30m"),)))
        chat = Chat(; service=ANTHROPICServiceEndpoint, model, max_tokens=16, kw...)
        push!(chat, Message(Val(:system), "You are terse."))
        push!(chat, Message(Val(:user), "Reply with exactly: hello"))
        @test_throws ArgumentError chatrequest!(chat)
        @test length(chat) == 2 && cumulative_cost(chat) == 0.0
    end
end

const _LIVE_FIXTURES = joinpath(@__DIR__, "fixtures")
_live_media(question, attachment; kw...) = Chat(; service=ANTHROPICServiceEndpoint, model=_LIVE_CLAUDE, max_tokens=16,
    reasoning_effort="none", messages=[Message(Val(:user), question, attachment)], kw...)

@testset "Anthropic Chat — reads an image sent as a base64 image block" begin
    r = _billed(chatrequest!(_live_media("What colour is the circle? One word.",
                                         ImageAttachment(joinpath(_LIVE_FIXTURES, "ollama", "red_circle.png")))))
    @test r isa LLMSuccess
    @test occursin("red", lowercase(r.message.content))
end

@testset "Anthropic Chat — reads a PDF sent as a base64 document block" begin
    # The fixture's only text is "The secret word is PELICAN.": nothing else answers the question.
    r = _billed(chatrequest!(_live_media("What is the secret word? Answer with the word only.",
                                         DocumentAttachment(joinpath(_LIVE_FIXTURES, "documents", "secret_word.pdf")))))
    @test r isa LLMSuccess
    @test occursin("PELICAN", uppercase(r.message.content))
end

@testset "Anthropic Chat — automatic caching writes a 5-minute prefix, then reads it, at Anthropic's rates" begin
    # Claude Haiku 5.5 caches prompts of 512 tokens or more; this system prompt is longer than
    # 1,200, and the nonce in front makes it a prefix no earlier request wrote. The region is
    # pinned, so the rates checked are the 1x global ones whatever the workspace default.
    filler = join(("The quick brown fox number $i jumps over the lazy dog near the river bank." for i in 1:100), " ")
    system = "Request $(time_ns())-$(rand(UInt32)).\n$filler\nReply with the single word OK."
    chat() = _live_claude("Ready?"; system, service=ANTHROPICServiceEndpoint(inference_geo="global"), max_tokens=16,
                          reasoning_effort="none", prompt_cache_options=PromptCacheOptions(ttl="5m"))
    probe = chat()
    @test JSON.parse(UniLM.encode_request(probe.service, probe))["inference_geo"] == "global"
    call() = _billed(chatrequest!(chat()))
    first = call()
    second = call()
    @test first isa LLMSuccess && second isa LLMSuccess
    w, r = first.usage, second.usage
    @test w.prompt_tokens >= 1_200
    @test w.cache_write_tokens > 0 && w.cache_write_extended_tokens == 0 && w.cached_tokens == 0
    @test r.cached_tokens == w.cache_write_tokens && r.cache_write_tokens == 0
    @test (w.inference_geo, r.inference_geo) == ("global", "global")
    # claude-haiku-5-5: $0.10 input, $0.01 cache read, $0.50 output per 1M tokens; a 5-minute write at 1.25x input.
    cost(u) = ((u.prompt_tokens - u.cached_tokens - u.cache_write_tokens) * 0.10 + u.cached_tokens * 0.01 +
               u.cache_write_tokens * 0.10 * 1.25 + u.completion_tokens * 0.50) / 1e6
    @test isapprox(estimated_cost(first), cost(w); rtol=1e-12)
    @test isapprox(estimated_cost(second), cost(r); rtol=1e-12)
end

@testset "Anthropic Chat — an explicit reasoning_effort returns the summarized thinking" begin
    chat = _live_claude("Is 1009 prime? Check the candidate divisors, then answer yes or no."; system="You are terse.",
                        max_tokens=1024, reasoning_effort="low")
    @test JSON.parse(UniLM.encode_request(ANTHROPICServiceEndpoint, chat))["thinking"] ==
          Dict("type" => "adaptive", "display" => "summarized")
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    @test "thinking" in _block_types(r.message)
    @test !isempty(something(reasoning_text(r), ""))
end

@testset "Anthropic Chat — fast mode on claude-opus-5-5 is reported and billed at twice the rates" begin
    # The region is pinned, so the rates checked are the 1x global ones whatever the workspace default.
    chat = Chat(service=ANTHROPICServiceEndpoint(inference_geo="global"), model="claude-opus-5-5", max_tokens=16,
                service_tier="fast", messages=[Message(Val(:user), "Reply with exactly: hello")])
    body = JSON.parse(UniLM.encode_request(chat.service, chat))
    @test body["speed"] == "fast" && !haskey(body, "service_tier") && body["inference_geo"] == "global"
    r = _billed(chatrequest!(chat))
    @test r isa LLMSuccess
    u = r.usage
    @test u.service_tier == "fast" && u.inference_geo == "global" && u.cached_tokens == 0 && u.cache_write_tokens == 0
    # claude-opus-5-5: $4 input, $20 output per 1M tokens; fast mode bills every rate at 2x.
    @test isapprox(estimated_cost(r), 2 * (u.prompt_tokens * 4.0 + u.completion_tokens * 20.0) / 1e6; rtol=1e-12)
end

@testset "Anthropic Chat — an endpoint's beta, user-profile and workspace headers reach the API" begin
    ask(ep) = chatrequest!(Chat(service=ep, model=_LIVE_CLAUDE, max_tokens=8, reasoning_effort="none",
                                messages=[Message(Val(:user), "Reply with exactly: hello")]);
                           config=RequestConfig(max_attempts=1))
    # A documented beta that needs no request field is accepted...
    @test _billed(ask(ANTHROPICServiceEndpoint(betas=["thinking-binding-controls-2026-08-01"]))) isa LLMSuccess
    # ...and a value the API does not take is its HTTP 400 naming the header, before any inference:
    # proof the header was sent (2026-10-22).
    for (ep, says) in ((ANTHROPICServiceEndpoint(betas=["unilm-no-such-beta-2026-01-01"]), "`anthropic-beta` header"),
                       (ANTHROPICServiceEndpoint(user_profile_id="uprof_011UniLMWitness0000000000"), "anthropic-user-profile-id"),
                       (ANTHROPICServiceEndpoint(workspace_id="wrkspc_011UniLMWitness000000000"), "anthropic-workspace-id"))
        r = ask(ep)
        @test r isa LLMFailure && r.status == 400 && occursin(says, r.response)
    end
end

@testset "Anthropic Chat — claude-fable-5-2 at effort low is priced at its row; a forced tool choice is refused locally" begin
    # Claude Fable 5.2: $8 input, $0.20 cache hit, $40 output per 1M tokens (2026-11-16), adaptive thinking always on;
    # forced tool use is an HTTP 400 ("tool_choice: type \"tool\" and \"any\" are not supported for this model.",
    # 2026-11-16), which UniLM refuses before the request.
    chat = Chat(service=ANTHROPICServiceEndpoint(inference_geo="global"), model="claude-fable-5-2", max_tokens=128,
                reasoning_effort="low", messages=[Message(Val(:user), "Reply with the single word: ok")])
    r = chatrequest!(chat)
    @test r isa LLMSuccess
    r isa LLMSuccess && @info "billed" r.model r.usage.prompt_tokens r.usage.completion_tokens cost = estimated_cost(r)
    @test startswith(something(r.model, ""), "claude-fable-5-2")
    u = r.usage
    @test u.prompt_tokens > 0 && u.completion_tokens > 0 && u.inference_geo == "global"
    @test (u.cached_tokens, u.cache_write_tokens) == (0, 0)
    @test isapprox(estimated_cost(r), (u.prompt_tokens * 8.0 + u.completion_tokens * 40.0) / 1e6; rtol=1e-12)
    tool = Tool(func=FunctionSignature(name="f", parameters=Dict("type" => "object", "properties" => Dict())))
    forced = Chat(service=ANTHROPICServiceEndpoint, model="claude-fable-5-2", tools=[tool], tool_choice="required",
                  messages=[Message(Val(:user), "Call f.")])
    @test_throws ArgumentError chatrequest!(forced)
end

@testset "Anthropic Chat — fast mode on claude-fable-5-2 bills twice its rates" begin
    # Fast mode is served on Claude Fable 5.2 "with no beta header", at "$16 USD per MTok of input and $80 USD per MTok
    # of output" (the release notes of 2026-11-09). A tiny cap: the turn may stop at max_tokens.
    chat = Chat(service=ANTHROPICServiceEndpoint(inference_geo="global"), model="claude-fable-5-2", max_tokens=32,
                reasoning_effort="low", service_tier="fast", messages=[Message(Val(:user), "Reply with exactly: hello")])
    @test JSON.parse(UniLM.encode_request(chat.service, chat))["speed"] == "fast"
    r = chatrequest!(chat)
    @test r isa LLMSuccess
    r isa LLMSuccess && @info "billed" r.model r.usage.prompt_tokens r.usage.completion_tokens cost = estimated_cost(r)
    u = r.usage
    @test u.service_tier == "fast" && u.inference_geo == "global" && (u.cached_tokens, u.cache_write_tokens) == (0, 0)
    @test isapprox(estimated_cost(r), 2 * (u.prompt_tokens * 8.0 + u.completion_tokens * 40.0) / 1e6; rtol=1e-12)
end

@testset "Anthropic Files — upload with an expiry, list, retrieve, the refused download, delete" begin
    # Free: Files API operations bill no tokens. A 22-byte text file, deleted at the end.
    path = joinpath(mktempdir(), "unilm-live.txt")
    write(path, "UniLM live Files test\n")
    a = ANTHROPICServiceEndpoint
    up = upload_file(path; service=a, expires_after=3600)
    @test up isa FileSuccess
    if up isa FileSuccess
        f = up.response
        try
            @test (f.filename, f.bytes, f.purpose, f.status) == ("unilm-live.txt", 22, nothing, nothing)
            @test f.raw["mime_type"] == "text/plain" && f.raw["downloadable"] === false
            @test abs(f.created_at - time()) < 600
            expires = UniLM.Dates.datetime2unix(something(UniLM._rfc3339_utc(f.raw["expires_at"])))
            @test isapprox(expires - f.created_at, 3600; atol=1)
            got = retrieve_file(f.id; service=a)
            @test got isa FileSuccess && got.response.id == f.id && got.response.created_at == f.created_at
            ready = poll_file(f.id; service=a)   # an Anthropic file needs no wait: one read
            @test ready isa FileSuccess && ready.response.id == f.id
            listed = list_files(; limit=1000, service=a)
            @test listed isa FileListSuccess && any(x -> x.id == f.id, listed.response.data)
            one = list_files(; limit=1, service=a)
            @test one isa FileListSuccess && length(one.response.data) == 1
            if one isa FileListSuccess && one.response.has_more
                @test list_files(; limit=1, after=one.response.raw["next_page"], service=a) isa FileListSuccess
            end
            # An uploaded file is not downloadable: Anthropic's HTTP 400, as it came.
            dl = file_content(f.id; service=a)
            @test dl isa FileFailure && dl.status == 400 && occursin("not downloadable", dl.response)
        finally
            del = delete_file(f.id; service=a)
            @test del isa FileDeleteSuccess && (del.id, del.object) == (f.id, "file_deleted")
        end
        @test retrieve_file(f.id; service=a) isa FileFailure
    end
end

# Token counting "is free to use" (https://platform.claude.com/docs/en/build-with-claude/token-counting), and so
# are the Models API's listing and lookup.
@testset "Anthropic count_tokens — a Chat's input tokens, on the GA route and with betas on the beta route" begin
    chat = _live_claude("Reply with exactly: hello"; max_tokens=64, stop=["END"], user="u-1")
    r = count_tokens(chat)
    @test r isa TokenCountSuccess && r.input_tokens > 0 && r.raw == Dict{String,Any}("input_tokens" => r.input_tokens)
    beta = _live_claude("Reply with exactly: hello"; service=ANTHROPICServiceEndpoint(betas=["fast-mode-2026-02-01"]),
                        max_tokens=64)
    @test count_tokens(beta) isa TokenCountSuccess && count_tokens(beta).input_tokens == r.input_tokens
    tools = _live_claude("What is the weather in Paris?"; tools=[_LIVE_WEATHER])
    @test count_tokens(tools).input_tokens > r.input_tokens
end

@testset "Anthropic retrieve_model — an alias resolves to its dated id; a retired model is a 404" begin
    r = retrieve_model("claude-opus-4-5"; service=ANTHROPICServiceEndpoint)
    @test r isa ModelSuccess && r.model.id == "claude-opus-4-5-20251101" && r.model.input_token_limit == 200000
    r = retrieve_model(_LIVE_CLAUDE; service=ANTHROPICServiceEndpoint())
    @test r isa ModelSuccess && r.model.id == _LIVE_CLAUDE && r.model.output_token_limit > 0
    r = retrieve_model("claude-3-opus-20240229"; service=ANTHROPICServiceEndpoint)
    @test r isa ModelListFailure && r.status == 404
    listed = list_models(; service=ANTHROPICServiceEndpoint)
    @test listed isa ModelListSuccess && length(listed.raw) == 1   # one page of up to 1000 models
end

@testset "Message Batches — two requests: create, retrieve, list, cancel; once ended, results and delete" begin
    # Two 16-token claude-haiku-5-5 requests, cancelled at once: a request canceled before it runs is not billed,
    # and one already sent costs its tokens at the Batch rates. A cancelled batch ended within a minute in the
    # runs of 2026-11-16; the poll gives it five, and the gate does not wait for a batch to run its requests.
    q() = _live_claude("Reply with exactly: ok"; max_tokens=16)
    created = create_batch(["one" => q(), "two" => q()])
    @test created isa BatchSuccess
    if created isa BatchSuccess
        id = created.response.id
        @test created.response.status == "in_progress" && created.response.request_counts["processing"] == 2
        @test retrieve_batch(id; service=ANTHROPICServiceEndpoint) isa BatchSuccess
        page = list_batches(; limit=20, service=ANTHROPICServiceEndpoint)
        @test page isa BatchListSuccess && any(b -> b.id == id, page.response.data)
        stopped = cancel_batch(id; service=ANTHROPICServiceEndpoint)
        @test stopped isa BatchSuccess && stopped.response.status in ("canceling", "ended")
        ended = poll_batch(id; service=ANTHROPICServiceEndpoint, interval=5.0, timeout=300.0)
        @test ended isa BatchSuccess && ended.response.status == "ended"
        r = batch_results(id; service=ANTHROPICServiceEndpoint)
        @test r isa BatchResultsSuccess && sort(first.(r.results)) == ["one", "two"]
        r isa BatchResultsSuccess && @info "Anthropic batch results" id kinds = [typeof(x.second) for x in r.results] cost = sum(
            (estimated_cost(x.second) for x in r.results if x.second isa LLMSuccess); init=0.0)
        d = delete_batch(id; service=ANTHROPICServiceEndpoint)
        @test d isa BatchDeleteSuccess && d.id == id
    end
end

@testset "Message Batches — two requests run to their end: batch-priced successes, each on its request's Chat" begin
    # Two 16-token claude-haiku-5-5 requests, left to run: a batch of 2026-11-16 ended in under five minutes, and the
    # poll gives it ten. Each success bills at the Batch rates, half the standard ones; read with the requests, each
    # result's self is its request's Chat with the reply appended.
    requests = ["one" => _live_claude("Reply with exactly: ok"; max_tokens=16),
                "two" => _live_claude("Reply with exactly: ok"; max_tokens=16)]
    created = create_batch(requests)
    @test created isa BatchSuccess
    if created isa BatchSuccess
        id = created.response.id
        seconds = @elapsed (ended = poll_batch(id; service=ANTHROPICServiceEndpoint, interval=15.0, timeout=600.0))
        done = ended isa BatchSuccess && ended.response.status == "ended"
        @test done
        if done
            r = batch_results(id; service=ANTHROPICServiceEndpoint, requests)
            @test r isa BatchResultsSuccess && sort(first.(r.results)) == ["one", "two"]
            for (cid, x) in (r isa BatchResultsSuccess ? r.results : [])
                @test x isa LLMSuccess
                x isa LLMSuccess || continue
                standard = LLMSuccess(message=x.message, self=x.self, model=x.model, usage=UniLM._standard_tier(x.usage))
                @test x.usage.service_tier == "batch" && 0 < estimated_cost(x) < estimated_cost(standard)
                chat = Dict(requests)[cid]
                @test x.self !== chat && length(x.self.messages) == length(chat.messages) + 1 && last(x.self.messages) === x.message
            end
            r isa BatchResultsSuccess && @info "Anthropic batch run to its end" id seconds cost = sum(
                (estimated_cost(x) for (_, x) in r.results if x isa LLMSuccess); init=0.0)
            @test delete_batch(id; service=ANTHROPICServiceEndpoint) isa BatchDeleteSuccess
        else
            cancel_batch(id; service=ANTHROPICServiceEndpoint)   # the poll gave up: stop the batch
        end
    end
end

@testset "Anthropic Skills — a skill and a second version, read, listed, a version's zip; the old version and the skill deleted" begin
    # Free: the Skills API bills no tokens. Everything made here is deleted, the skill (with its versions) in a `finally`.
    a = ANTHROPICServiceEndpoint
    name = "unilm-live-" * string(rand(UInt32); base=36)
    function skill_dir(version)
        dir = mkpath(joinpath(mktempdir(), name))
        write(joinpath(dir, "SKILL.md"), "---\nname: $name\ndescription: Replies with the word pong (version $version). " *
                                         "Use it when asked to ping.\n---\nReply with the single word pong.\n")
        write(joinpath(dir, "notes.txt"), "UniLM live skill, version $version\n")
        dir
    end
    s = create_skill(skill_dir(1); display_name="UniLM live skill", service=a)
    @info "Anthropic create_skill live" result = (s isa SkillSuccess ? s.response.raw : s)
    @test s isa SkillSuccess
    if s isa SkillSuccess
        id = s.response.id
        try
            @test (s.response.name, s.response.description, s.response.default_version) == ("UniLM live skill", nothing, nothing)
            @test s.response.raw["source"] == Dict("type" => "custom") && abs(s.response.created_at - time()) < 600
            first_version = s.response.latest_version
            v = create_skill_version(id, skill_dir(2); service=a)
            @test v isa SkillVersionSuccess && (v.response.skill_id, v.response.name) == (id, name)
            @test v.response.version == v.response.id != first_version && occursin("version 2", v.response.description)
            r = retrieve_skill(id; service=a)
            @test r isa SkillSuccess && r.response.latest_version == v.response.version
            latest = retrieve_skill_version(id, "latest"; service=a)
            @test latest isa SkillVersionSuccess && latest.response.version == v.response.version
            versions = list_skill_versions(id; service=a)   # newest first
            @test versions isa SkillVersionListSuccess && [x.version for x in versions.response.data] == [v.response.version, first_version]
            listed = list_skills(; source="custom", limit=1000, service=a)
            @test listed isa SkillListSuccess && any(x -> x.id == id, listed.response.data)
            zip = skill_content(id; version=v.response.version, service=a)
            @info "Anthropic skill_content live" result = (zip isa FileContentSuccess ? length(zip.content) : zip)
            @test zip isa FileContentSuccess && zip.content[1:4] == b"PK\x03\x04"
            d = delete_skill_version(id, first_version; service=a)
            @test d isa SkillDeleteSuccess && d.id == first_version
        finally
            del = delete_skill(id; service=a)
            @test del isa SkillDeleteSuccess && del.id == id
        end
        gone = retrieve_skill(id; service=a)
        @test gone isa SkillFailure && gone.status == 404
    end
end

end  # if ANTHROPIC_API_KEY
