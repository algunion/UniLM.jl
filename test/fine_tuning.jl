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
