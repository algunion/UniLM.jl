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
               at_a_glance(:jev), capability_matrix(), on_your_provider(ProviderFacts.llm, :streaming), openai_model_rules(),
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
@testset "ProviderFacts — a chapter's tables name only that chapter's entries" begin
    linked(md) = [chopprefix(only(only(r)).url, "@ref ") for r in only(md.content).rows[2:end]]
    chapter_of(id) = only(e for e in REGISTRY if id in values(e.features)).chapter
    @test all(id -> chapter_of(id) == ProviderFacts.llm, linked(on_your_provider(ProviderFacts.llm, :images)))
    @test all(id -> chapter_of(id) == ProviderFacts.decisions, linked(on_your_provider(ProviderFacts.decisions, :images)))
    supports(key) = [Symbol(c.code) for c in only(at_a_glance(key).content).rows[2][4] if c isa Markdown.Code]
    @test all(e -> e.chapter == ProviderFacts.llm || (s = supports(e.key); !isempty(s) && s ⊆ ProviderFacts.DECISION_CAPABILITIES),
              REGISTRY)
end
@testset "ProviderFacts — the registry and price checks refuse a broken table, naming what is wrong" begin
    remake(e; changes...) = ProviderEntry(; (f => get(changes, f, getfield(e, f)) for f in fieldnames(ProviderEntry))...)
    function refusal(entries)                       # the entry and field the check names; `nothing` when it accepts
        try ProviderFacts._validated(entries); nothing
        catch err; err isa ProviderFacts.RegistryError || rethrow(); (err.key, err.field) end
    end
    openai, gemini, jev, azure = entry(:openai), entry(:gemini), entry(:jev), entry(:azure)
    @test refusal(copy(REGISTRY)) === nothing
    @test refusal([openai, openai]) == (:openai, :key)
    @test refusal([remake(azure; endpoints=ServiceEndpointSpec[])]) == (:azure, :endpoints)
    @test refusal([remake(azure; pages=String[])]) == (:azure, :pages)
    @test refusal([remake(azure; pages=["providers/azure.html"])]) == (:azure, :pages)
    @test refusal([remake(jev; default_model="jev-1.13.0")]) == (:jev, :default_model)
    @test refusal([remake(openai; default_model=nothing)]) == (:openai, :default_model)
    @test refusal([openai, remake(gemini; start_id="openai_start")]) == (:gemini, :start_id)
    @test refusal([openai, remake(gemini; features=merge(gemini.features, Dict(:setup => "openai_setup")))]) ==
          (:gemini, :features)
    @test refusal([remake(jev; features=merge(jev.features, Dict(:images => "jev_images__next")))]) == (:jev, :features)
    row = UniLM.DEFAULT_PRICING["gpt-6-sol"]
    function price_refusal(ids, unprefixed...)       # the id and tables the partition names; `nothing` when it accepts
        try ProviderFacts._price_partition(Dict(id => row for id in ids), unprefixed...); nothing
        catch err; err isa ProviderFacts.PriceTableError || rethrow(); (err.id, err.tables) end
    end
    @test price_refusal(collect(keys(UniLM.DEFAULT_PRICING))) === nothing
    @test price_refusal(["chat-latest", "mistral-large-latest"]) == ("mistral-large-latest", Symbol[])
    @test price_refusal(["chat-latest", "gpt-6-sol"], ("chat-latest" => :openai, "gpt-6-sol" => :jev)) ==
          ("gpt-6-sol", [:openai, :jev])
    @test_throws ArgumentError ProviderFacts._price_partition(Dict("gpt-6-sol" => row))  # "chat-latest" is unpriced
end
