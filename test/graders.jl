# test/graders.jl — OpenAI's grader verbs (run, validate) and the typed graders. Local servers only.

@testset "Graders — run_grader and validate_grader" begin
    g = StringCheckGrader(name="exact", input="{{sample.output_text}}", reference="{{item.answer}}", operation="eq")
    wire = Dict("type" => "string_check", "name" => "exact", "input" => "{{sample.output_text}}",
                "reference" => "{{item.answer}}", "operation" => "eq")
    ran = """{"reward":1,"metadata":{"name":"exact","type":"string_check","errors":{},"execution_time":0.01,
              "scores":{},"token_usage":null,"sampled_model_name":null},"sub_rewards":{},
              "model_grader_token_usage_per_model":{}}"""
    rs, seen = _with_scripted((n, _) -> _json(200, n == 1 ? ran : JSON.json(Dict("grader" => wire)))) do
        [run_grader(g, "Paris"; item=Dict("answer" => "Paris"), service=URLProbe), validate_grader(g; service=URLProbe)]
    end
    @test [(s.method, s.target) for s in seen] == [("POST", "/v1/fine_tuning/alpha/graders/run"),
                                                    ("POST", "/v1/fine_tuning/alpha/graders/validate")]
    @test JSON.parse(seen[1].body; dicttype=Dict{String,Any}) ==
          Dict("grader" => wire, "model_sample" => "Paris", "item" => Dict("answer" => "Paris"))
    @test JSON.parse(seen[2].body; dicttype=Dict{String,Any}) == Dict("grader" => wire)
    @test rs[1] isa GraderRunSuccess && rs[1].response.reward === 1.0 && rs[1].response.metadata["type"] == "string_check"
    @test rs[1].response.sub_rewards == Dict{String,Any}() && rs[1].response.raw["model_grader_token_usage_per_model"] == Dict()
    @test rs[2] isa GraderValidateSuccess && rs[2].grader == wire
    # No item: none is sent. A raw Dict grader — the score_model, label_model and multi graders — goes as given.
    sm = Dict("type" => "score_model", "name" => "s", "model" => "gpt-4.1-mini",
              "input" => [Dict("role" => "user", "content" => "Score {{sample.output_text}}")])
    _, seen = _with_scripted((_, _) -> _json(200, ran)) do
        run_grader(sm, "x"; service=URLProbe)
    end
    @test JSON.parse(only(seen).body; dicttype=Dict{String,Any}) == Dict("grader" => sm, "model_sample" => "x")
end

@testset "Graders — the typed graders' wire forms and refusals" begin
    lower(g) = JSON.parse(JSON.json(g); dicttype=Dict{String,Any})
    @test lower(TextSimilarityGrader(name="t", input="{{sample.output_text}}", reference="r", evaluation_metric="bleu")) ==
          Dict("type" => "text_similarity", "name" => "t", "input" => "{{sample.output_text}}", "reference" => "r",
               "evaluation_metric" => "bleu")
    @test lower(PythonGrader(name="py", source="def grade(sample, item): return 1.0")) ==
          Dict("type" => "python", "name" => "py", "source" => "def grade(sample, item): return 1.0")
    @test lower(PythonGrader(name="py", source="s", image_tag="2025-05-08"))["image_tag"] == "2025-05-08"
    # A typed grader inside a multi grader, or inside a reinforcement `method`, lowers there too.
    sc = StringCheckGrader(name="a", input="i", reference="r", operation="ilike")
    @test lower(Dict("type" => "multi", "name" => "m", "graders" => Dict("a" => sc), "calculate_output" => "a")) ==
          Dict("type" => "multi", "name" => "m", "graders" => Dict("a" => lower(sc)), "calculate_output" => "a")
    @test all(op -> StringCheckGrader(name="a", input="i", reference="r", operation=op) isa Grader, ("eq", "ne", "like", "ilike"))
    for m in ("cosine", "fuzzy_match", "bleu", "gleu", "meteor", "rouge_1", "rouge_2", "rouge_3", "rouge_4", "rouge_5", "rouge_l")
        @test TextSimilarityGrader(name="t", input="i", reference="r", evaluation_metric=m) isa Grader
    end
    err = try StringCheckGrader(name="a", input="i", reference="r", operation="equals"); nothing catch e; e end
    @test err isa ArgumentError && occursin("StringCheckGrader", err.msg) && occursin("\"ilike\"", err.msg)
    err = try TextSimilarityGrader(name="t", input="i", reference="r", evaluation_metric="rouge"); nothing catch e; e end
    @test err isa ArgumentError && occursin("TextSimilarityGrader", err.msg) && occursin("\"rouge_l\"", err.msg)
end

@testset "Graders — malformed replies, failures, seams" begin
    g = StringCheckGrader(name="a", input="i", reference="r", operation="eq")
    # The reference requires `reward`, `metadata` and `sub_rewards` of a run and `grader` of a validation.
    for body in ("{}", """{"reward":"1","metadata":{},"sub_rewards":{}}""", """{"reward":1,"sub_rewards":{}}""",
                 """{"reward":1,"metadata":{}}""")
        @test _answered(() -> run_grader(g, "x"; service=URLProbe), 200; body) isa FineTuningCallError
    end
    @test _answered(() -> validate_grader(g; service=URLProbe), 200; body="{}") isa FineTuningCallError
    r = _answered(() -> validate_grader(g; service=URLProbe), 400; body="""{"error":{"message":"bad grader"}}""",
                  headers=["x-request-id" => "req_g"])
    @test r isa FineTuningFailure && r.status == 400 && r.request_id == "req_g"
    for call in (kw -> run_grader(g, "x"; kw...), kw -> validate_grader(Dict("type" => "python"); kw...))
        @test _seam_timeout(call((; service=SeamProbe, config=_TINY_DEADLINE)), FineTuningCallError)
        @test_throws ArgumentError call((; service=GEMINIOpenAIServiceEndpoint))
        tok = CancelToken(); cancel!(tok)
        r, seen = _with_scripted(() -> call((; service=URLProbe, cancel=tok)), (_, _) -> _json(200, "{}"))
        @test isempty(seen) && r isa FineTuningCallError && r.cause isa UniLMCancelled
    end
end
