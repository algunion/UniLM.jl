# Live witnesses for usage and cost on OpenAI: a 12-hour prompt-cache write is reported and
# priced, and a Chat reply carries the model that served it and the tier it was billed at.
# Minimal tokens on gpt-6-luna at reasoning effort "none".
if !haskey(ENV, "OPENAI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping OpenAI usage and cost witnesses (set UNILM_LIVE=1 and OPENAI_API_KEY to run live)"
else

HTTP.close_idle_connections!()

@testset "Responses: a 12-hour cache write is reported and priced" begin
    # OpenAI caches prompt prefixes of at least 1,024 tokens. The fixed text is about 2,200
    # tokens; the fresh nonce line in front makes the whole prompt a prefix no earlier
    # request wrote, so this call writes it, at the 12-hour retention it asks for.
    text = join(("The quick brown fox number $i jumps over the lazy dog near the river bank." for i in 1:130), " ")
    r = respond("Request $(time_ns())-$(rand(UInt32)).\n$text\nReply with the single word OK.";
                model="gpt-6-luna", reasoning=Reasoning(effort="none"), max_output_tokens=16,
                prompt_cache_options=PromptCacheOptions(ttl="12h"))
    @test r isa ResponseSuccess
    u = token_usage(r)
    @info "12-hour cache write" u.prompt_tokens u.completion_tokens u.cached_tokens u.cache_write_tokens u.cache_write_extended_tokens u.service_tier cost = estimated_cost(r)
    @test u.cache_write_tokens > 0
    @test u.cache_write_tokens == u.cache_write_extended_tokens
    @test u.service_tier == "default"
    # The exact bill: GPT-6 writes a 12-hour cache at 2.5 times the input rate (OpenAI's prompt-caching
    # guide), a 30-minute one at 1.25 times; the rest at gpt-6-luna's standard rates.
    rates = DEFAULT_PRICING["gpt-6-luna"]
    fresh = u.prompt_tokens - u.cached_tokens - u.cache_write_tokens
    bill = rates.input * (fresh + 2.5 * u.cache_write_extended_tokens + 1.25 * (u.cache_write_tokens - u.cache_write_extended_tokens)) +
           rates.cached_input * u.cached_tokens + rates.output * u.completion_tokens
    @test estimated_cost(r) ≈ bill
end

@testset "Chat: the served model and the billed tier" begin
    r = chatrequest!(; systemprompt="Reply with the single word OK.", userprompt="Ready?",
                     model="gpt-6-luna", reasoning_effort="none", max_completion_tokens=8)
    @test r isa LLMSuccess
    @info "Chat served model and tier" r.model r.usage cost = estimated_cost(r)
    @test startswith(something(r.model, ""), "gpt-6-luna")
    @test r.usage.service_tier == "default"
end

end
