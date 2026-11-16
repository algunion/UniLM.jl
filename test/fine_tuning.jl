@testset "Fine-tuning API — config seam wiring" begin
    @test _seam_timeout(create_fine_tuning_job(model="gpt-4o", training_file="file_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _seam_timeout(retrieve_fine_tuning_job("ft_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _seam_timeout(cancel_fine_tuning_job("ft_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _seam_timeout(list_fine_tuning_jobs(service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _seam_timeout(list_fine_tuning_events("ft_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _seam_timeout(list_fine_tuning_checkpoints("ft_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
end

@testset "Fine-tuning API — events and checkpoints page with limit/after" begin
    t = _recorded_targets() do
        list_fine_tuning_events(_HOSTILE_ID; limit=2, after=_HOSTILE_ID, service=URLProbe)
        list_fine_tuning_checkpoints(_HOSTILE_ID; limit=3, after=_HOSTILE_ID, service=URLProbe)
        list_fine_tuning_events("ftjob-abc123"; after="ftevent-9", service=URLProbe)
        list_fine_tuning_checkpoints("ftjob-abc123"; limit=5, service=URLProbe)
    end
    @test t == ["/v1/fine_tuning/jobs/$_HOSTILE_ENC/events?limit=2&after=$_HOSTILE_ENC",
                "/v1/fine_tuning/jobs/$_HOSTILE_ENC/checkpoints?limit=3&after=$_HOSTILE_ENC",
                "/v1/fine_tuning/jobs/ftjob-abc123/events?after=ftevent-9",
                "/v1/fine_tuning/jobs/ftjob-abc123/checkpoints?limit=5"]
end

@testset "Fine-tuning API — a failure keeps the request id the service sent" begin
    r = _answered(() -> retrieve_fine_tuning_job("ft_x"; service=URLProbe), 404; headers=["x-request-id" => "req_ft"])
    @test r isa UniLM.FineTuningFailure && r.request_id == "req_ft"
end

@testset "Fine-tuning API — a separator-bearing job id stays one path segment" begin
    t = _recorded_targets() do
        retrieve_fine_tuning_job(_HOSTILE_ID; service=URLProbe)
        cancel_fine_tuning_job(_HOSTILE_ID; service=URLProbe)
        list_fine_tuning_events(_HOSTILE_ID; service=URLProbe)
        list_fine_tuning_checkpoints(_HOSTILE_ID; service=URLProbe)
        list_fine_tuning_jobs(; limit=2, after=_HOSTILE_ID, service=URLProbe)
    end
    @test t == ["/v1/fine_tuning/jobs/$_HOSTILE_ENC",
                "/v1/fine_tuning/jobs/$_HOSTILE_ENC/cancel",
                "/v1/fine_tuning/jobs/$_HOSTILE_ENC/events",
                "/v1/fine_tuning/jobs/$_HOSTILE_ENC/checkpoints",
                "/v1/fine_tuning/jobs?limit=2&after=$_HOSTILE_ENC"]
    g = _recorded_targets() do
        list_fine_tuning_events("ftjob-abc123"; service=URLProbe)
        list_fine_tuning_jobs(; after="ftjob-abc123", service=URLProbe)
    end
    @test g == ["/v1/fine_tuning/jobs/ftjob-abc123/events",
                "/v1/fine_tuning/jobs?after=ftjob-abc123"]
end


@testset "Fine-tuning API — a list reply without data or has_more is a call error, not an empty last page" begin
    lists = () -> [list_fine_tuning_jobs(; service=URLProbe), list_fine_tuning_events("ftjob_1"; service=URLProbe),
                   list_fine_tuning_checkpoints("ftjob_1"; service=URLProbe)]
    for body in ("{}", """{"data":[]}""", """{"has_more":false}""", """{"data":null,"has_more":false}""",
                 """{"data":{},"has_more":false}""", """{"data":[],"has_more":"no"}""")
        rs = first(_with_scripted(lists, (_, _) -> _json(200, body)))
        @test all(r -> r isa FineTuningCallError, rs)
    end
    r = _answered(() -> list_fine_tuning_jobs(; service=URLProbe), 200; body="{}")
    @test r isa FineTuningCallError && occursin("\"data\"", r.error)
    rs = first(_with_scripted(lists, (_, _) -> _json(200, """{"object":"list","data":[{"id":"x"}],"has_more":true}""")))
    @test all(r -> r isa FineTuningListSuccess && r.response.has_more && r.response.data[1]["id"] == "x", rs)
end

@testset "Fine-tuning API — a job reply without its status is a call error" begin
    # OpenAI's FineTuningJob requires `status`; a reply without one is no job of known state.
    jobs = () -> [create_fine_tuning_job(; model="m", training_file="file_1", service=URLProbe),
                  retrieve_fine_tuning_job("ftjob_1"; service=URLProbe), cancel_fine_tuning_job("ftjob_1"; service=URLProbe)]
    for body in ("""{"id":"ftjob_1"}""", """{"id":"ftjob_1","status":null}""", """{"id":"ftjob_1","status":3}""",
                 """{"status":"queued"}""")
        rs = first(_with_scripted(jobs, (_, _) -> _json(200, body)))
        @test all(r -> r isa FineTuningCallError, rs)
    end
    r = _answered(() -> retrieve_fine_tuning_job("ftjob_1"; service=URLProbe), 200; body="""{"id":"ftjob_1"}""")
    @test r isa FineTuningCallError && occursin("\"status\"", r.error)
    rs = first(_with_scripted(jobs, (_, _) -> _json(200, """{"id":"ftjob_1","status":"queued","model":"m"}""")))
    @test all(r -> r isa FineTuningSuccess && r.response.status == "queued" && r.response.model == "m", rs)
end

@testset "Fine-tuning API — create_fine_tuning_job sends a seed and W&B integrations" begin
    job = (_, _) -> _json(200, """{"id":"ftjob_1","status":"queued"}""")
    r, seen = _with_scripted(job) do
        create_fine_tuning_job(; model="gpt-4.1-mini", training_file="file_1", seed=42,
            integrations=[WandbIntegration(project="p"),
                          WandbIntegration(project="q", name="run", entity="team", tags=["a", "b"])],
            service=URLProbe)
    end
    @test r isa FineTuningSuccess
    b = JSON.parse(only(seen).body; dicttype=Dict{String,Any})
    @test b["seed"] == 42
    @test b["integrations"] == [Dict("type" => "wandb", "wandb" => Dict("project" => "p")),
        Dict("type" => "wandb", "wandb" => Dict("project" => "q", "name" => "run", "entity" => "team", "tags" => ["a", "b"]))]
    _, seen = _with_scripted(job) do
        create_fine_tuning_job(; model="gpt-4.1-mini", training_file="file_1", seed=0, integrations=WandbIntegration[],
                               service=URLProbe)
    end
    b = JSON.parse(only(seen).body; dicttype=Dict{String,Any})
    @test b["seed"] == 0 && b["integrations"] == []
    _, seen = _with_scripted(job) do
        create_fine_tuning_job(; model="gpt-4.1-mini", training_file="file_1", service=URLProbe)
    end
    @test !any(k -> haskey(JSON.parse(only(seen).body), k), ("seed", "integrations"))
    # The reference bounds the seed to 0..2147483647: refused before anything is sent.
    for s in (-1, 2147483648)
        err = try create_fine_tuning_job(; model="m", training_file="f", seed=s, service=SeamProbe); nothing catch e; e end
        @test err isa ArgumentError && occursin("create_fine_tuning_job", err.msg) && occursin("2147483647", err.msg)
    end
end

@testset "Fine-tuning API — pause and resume a job" begin
    rs, seen = _with_scripted((n, _) -> _json(200, """{"id":"ftjob_1","status":"$(n == 1 ? "pausing" : "running")"}""")) do
        [pause_fine_tuning_job(_HOSTILE_ID; service=URLProbe), resume_fine_tuning_job(_HOSTILE_ID; service=URLProbe)]
    end
    @test [(s.method, s.target) for s in seen] == [("POST", "/v1/fine_tuning/jobs/$_HOSTILE_ENC/pause"),
                                                    ("POST", "/v1/fine_tuning/jobs/$_HOSTILE_ENC/resume")]
    @test rs[1] isa FineTuningSuccess && rs[1].response.status == "pausing"
    @test rs[2] isa FineTuningSuccess && rs[2].response.status == "running"
    # A job that cannot pause in its state is OpenAI's 400, kept with its request id.
    r = _answered(() -> pause_fine_tuning_job("ftjob_1"; service=URLProbe), 400;
                  body="""{"error":{"message":"cannot pause"}}""", headers=["x-request-id" => "req_p"])
    @test r isa FineTuningFailure && r.status == 400 && r.request_id == "req_p"
    @test _answered(() -> resume_fine_tuning_job("ftjob_1"; service=URLProbe), 200; body="""{"id":"ftjob_1"}""") isa
          FineTuningCallError
    for verb in (pause_fine_tuning_job, resume_fine_tuning_job)
        @test _seam_timeout(verb("ftjob_1"; service=SeamProbe, config=_TINY_DEADLINE), FineTuningCallError)
        @test_throws ArgumentError verb("ftjob_1"; service=GEMINIOpenAIServiceEndpoint)
        tok = CancelToken(); cancel!(tok)
        r, seen = _with_scripted(() -> verb("ftjob_1"; service=URLProbe, cancel=tok), (_, _) -> _json(200, "{}"))
        @test isempty(seen) && r isa FineTuningCallError && r.cause isa UniLMCancelled
    end
end

@testset "Fine-tuning API — list_fine_tuning_jobs filters on metadata" begin
    # The reference's deep-object filter: `metadata[k]=v`, each key and value one encoded query part.
    t = _recorded_targets() do
        list_fine_tuning_jobs(; metadata=Dict("team" => "a b", "k/1" => ""), service=URLProbe)
        list_fine_tuning_jobs(; limit=2, after="ftjob_1", metadata=Dict("env" => "prod"), service=URLProbe)
        list_fine_tuning_jobs(; metadata=Dict(_HOSTILE_ID => _HOSTILE_ID), service=URLProbe)
        list_fine_tuning_jobs(; metadata=Dict{String,String}(), service=URLProbe)
    end
    @test t == ["/v1/fine_tuning/jobs?metadata%5Bk%2F1%5D=&metadata%5Bteam%5D=a%20b",
                "/v1/fine_tuning/jobs?limit=2&after=ftjob_1&metadata%5Benv%5D=prod",
                "/v1/fine_tuning/jobs?metadata%5B$_HOSTILE_ENC%5D=$_HOSTILE_ENC",
                "/v1/fine_tuning/jobs"]
end

@testset "Fine-tuning API — checkpoint permissions: create, list and delete" begin
    ck = "ft:gpt-4.1-mini:org:x:ckpt-step-10"
    enc = "ft%3Agpt-4.1-mini%3Aorg%3Ax%3Ackpt-step-10"
    perm = """{"object":"checkpoint.permission","id":"cp_1","created_at":1,"project_id":"proj_1"}"""
    page = """{"object":"list","data":[$perm],"first_id":"cp_1","last_id":"cp_1","has_more":false}"""
    deleted = """{"object":"checkpoint.permission","id":"cp_1","deleted":true}"""
    rs, seen = _with_scripted((n, _) -> _json(200, n <= 3 ? page : deleted)) do
        [create_fine_tuning_checkpoint_permission(ck, ["proj_1", "proj_2"]; service=URLProbe),
         list_fine_tuning_checkpoint_permissions(ck; project_id="proj_1", limit=5, after="cp_0", order="ascending",
                                                 service=URLProbe),
         list_fine_tuning_checkpoint_permissions(_HOSTILE_ID; service=URLProbe),
         delete_fine_tuning_checkpoint_permission(ck, _HOSTILE_ID; service=URLProbe)]
    end
    @test [(s.method, s.target) for s in seen] == [
        ("POST", "/v1/fine_tuning/checkpoints/$enc/permissions"),
        ("GET", "/v1/fine_tuning/checkpoints/$enc/permissions?limit=5&after=cp_0&project_id=proj_1&order=ascending"),
        ("GET", "/v1/fine_tuning/checkpoints/$_HOSTILE_ENC/permissions"),
        ("DELETE", "/v1/fine_tuning/checkpoints/$enc/permissions/$_HOSTILE_ENC")]
    @test JSON.parse(seen[1].body; dicttype=Dict{String,Any}) == Dict("project_ids" => ["proj_1", "proj_2"])
    @test rs[1] isa FineTuningListSuccess && rs[1].response.data[1]["project_id"] == "proj_1" && !rs[1].response.has_more
    @test rs[2] isa FineTuningListSuccess && rs[3] isa FineTuningListSuccess
    @test rs[4] isa FineTuningDeleteSuccess && rs[4].id == "cp_1" && rs[4].deleted
    # A delete counts when its record names what it deleted and says "deleted": true; a list needs its page.
    for body in ("""{"deleted":true}""", """{"id":"cp_1","deleted":false}""", """{"id":"cp_1"}""", "{}",
                 """{"id":"cp_1","deleted":true}""", """{"object":"checkpoint","id":"cp_1","deleted":true}""")
        @test _answered(() -> delete_fine_tuning_checkpoint_permission(ck, "cp_1"; service=URLProbe), 200; body) isa
              FineTuningCallError
    end
    r = _answered(() -> delete_fine_tuning_checkpoint_permission(ck, "cp_1"; service=URLProbe), 200;
                  body="""{"object":"checkpoint","id":"cp_1","deleted":true}""")
    @test r isa FineTuningCallError && occursin("\"checkpoint.permission\"", r.error)
    @test _answered(() -> create_fine_tuning_checkpoint_permission(ck, ["p"]; service=URLProbe), 200; body="{}") isa
          FineTuningCallError
    # The reference asks for an admin key; a project key's refusal is the family's typed failure.
    r = _answered(() -> list_fine_tuning_checkpoint_permissions(ck; service=URLProbe), 403;
                  body="""{"error":{"message":"admin key required"}}""", headers=["x-request-id" => "req_cp"])
    @test r isa FineTuningFailure && r.status == 403 && r.request_id == "req_cp"
    err = try list_fine_tuning_checkpoint_permissions(ck; order="newest", service=SeamProbe); nothing catch e; e end
    @test err isa ArgumentError && occursin("\"ascending\"", err.msg) && occursin("\"descending\"", err.msg)
    calls = (kw -> create_fine_tuning_checkpoint_permission(ck, ["p"]; kw...),
             kw -> list_fine_tuning_checkpoint_permissions(ck; kw...),
             kw -> delete_fine_tuning_checkpoint_permission(ck, "cp_1"; kw...))
    for call in calls
        @test _seam_timeout(call((; service=SeamProbe, config=_TINY_DEADLINE)), FineTuningCallError)
        @test_throws ArgumentError call((; service=AZUREServiceEndpoint))
        tok = CancelToken(); cancel!(tok)
        r, seen = _with_scripted(() -> call((; service=URLProbe, cancel=tok)), (_, _) -> _json(200, "{}"))
        @test isempty(seen) && r isa FineTuningCallError && r.cause isa UniLMCancelled
    end
end

@testset "Fine-tuning API — the grader endpoints are not wrapped" begin
    # OpenAI is deprecating graders (its graders guide); a reinforcement `method` Dict still carries one as sent.
    for name in (:Grader, :StringCheckGrader, :TextSimilarityGrader, :PythonGrader, :GraderRun, :GraderRunSuccess,
                 :GraderValidateSuccess, :run_grader, :validate_grader)
        @test !isdefined(UniLM, name)
    end
    method = Dict("type" => "reinforcement", "reinforcement" => Dict("grader" => Dict("type" => "string_check",
        "name" => "exact", "input" => "{{sample.output_text}}", "reference" => "{{item.answer}}", "operation" => "eq")))
    _, seen = _with_scripted(() -> create_fine_tuning_job(; model="o4-mini", training_file="file_x", method,
                                                          service=URLProbe), (_, _) -> _json(200, "{}"))
    @test JSON.parse(only(seen).body; dicttype=Dict{String,Any})["method"] == method
end
