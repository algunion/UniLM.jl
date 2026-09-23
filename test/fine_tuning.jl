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
