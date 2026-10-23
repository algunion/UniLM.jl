using Test, Markdown, UniLM
using Documenter: MarkdownAST
isdefined(Main, :ProviderFacts) || include(joinpath(@__DIR__, "ProviderFacts.jl")); using .ProviderFacts
hasvalue(n) = n.element isa MarkdownAST.JuliaValue || any(hasvalue, n.children)
@testset "ProviderFacts — every entry's facts come from the code" begin
    @test entry(:anthropic).default_model == UniLM.default_model(ANTHROPICServiceEndpoint) == "claude-opus-5-5"
    @test entry(:openai).default_model == "gpt-6-sol" && entry(:jev).default_model == "jev-latest"
    @test entry(:openai_decisions).default_model == "gpt-6-luna"
    @test entry(:ollama).default_model === nothing && entry(:ollama_decisions).default_model === nothing
    @test all(e -> e.chapter == ProviderFacts.llm, filter(e -> e.key in (:openai, :anthropic, :gemini), REGISTRY))
    @test length(unique(e.start_id for e in REGISTRY)) == length(REGISTRY) == 11
end
@testset "ProviderFacts — tables have one row per entry and link each start page" begin
    t = provider_table(ProviderFacts.llm).content[1]
    @test t isa Markdown.Table && length(t.rows) == 1 + count(e -> e.chapter == ProviderFacts.llm, REGISTRY)
    @test occursin("(@ref anthropic_start)", Markdown.plain(provider_table(ProviderFacts.llm)))
    @test occursin("embeddings", Markdown.plain(at_a_glance(:anthropic)))          # listed as not available
end
@testset "ProviderFacts — every generated table converts to MarkdownAST without raw values" begin
    for md in (provider_table(ProviderFacts.llm), provider_table(ProviderFacts.decisions), at_a_glance(:anthropic),
               at_a_glance(:jev), capability_matrix(), on_your_provider(:streaming), openai_model_rules(),
               claude_family_table(), gemini_thinking_table(), price_table(:anthropic))
        @test all(r -> all(c -> c isa AbstractVector, r), only(md.content).rows)
        @test !hasvalue(convert(MarkdownAST.Node, md))
    end
end
@testset "ProviderFacts — rule and price tables cover every row of the code's tables" begin
    @test count(==('\n'), Markdown.plain(claude_family_table())) >= length(UniLM._CLAUDE_FAMILIES) + 2
    @test all(k -> occursin(k, Markdown.plain(gemini_thinking_table())), keys(UniLM._GEMINI_THINKING_LEVELS))
    @test sort([m for p in (:openai, :anthropic, :gemini, :deepseek, :jev) for m in price_rows(p)]) ==
          sort(collect(keys(UniLM.DEFAULT_PRICING)))                               # every row in exactly one table
    @test "claude-haiku-5-5" in price_rows(:anthropic)
end
