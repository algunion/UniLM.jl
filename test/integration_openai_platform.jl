# test/integration_openai_platform.jl — live: OpenAI's fine-tuning, grader, webhook-endpoint, safety and content
# provenance verbs. Free calls only: no job is created (the training file is made up), the ids read are made
# up, the grader is a string_check one (no model samples), and an endpoint this suite makes is deleted in a
# `finally`. Needs UNILM_LIVE=1 and OPENAI_API_KEY.
if !haskey(ENV, "OPENAI_API_KEY") || get(ENV, "UNILM_LIVE", "") != "1"
    @info "Skipping OpenAI platform integration tests (set UNILM_LIVE=1 and OPENAI_API_KEY to run live)"
else

HTTP.close_idle_connections!()

@testset "Fine-tuning API live — a job list filtered on metadata; pause and resume of a made-up job are 404s" begin
    l = list_fine_tuning_jobs(; limit=1, metadata=Dict("unilm-live" => "none"))
    @test l isa FineTuningListSuccess && l.response.data == [] && !l.response.has_more
    for verb in (pause_fine_tuning_job, resume_fine_tuning_job)
        r = verb("ftjob-unilmlivemadeup000000")
        @info "$verb on a made-up job" status = (r isa FineTuningFailure ? r.status : r)
        @test r isa FineTuningFailure && r.status == 404
    end
end

@testset "Fine-tuning API live — a seed and integrations pass the request's validation" begin
    # OpenAI validates the body first: a seed outside 0..2147483647 is its 400 naming `seed` (witnessed on
    # 2026-11-16), which UniLM refuses before sending. This body passes, and the call stops at the organization's
    # access to training (403 `training_not_available` on 2026-11-16) or at the made-up training file (a 400).
    r = create_fine_tuning_job(; model="gpt-4.1-mini-2025-04-14", training_file="file-unilmlivemadeup000000", seed=42,
                               integrations=[WandbIntegration(project="unilm-live")])
    @info "create_fine_tuning_job with a made-up training file" status = (r isa FineTuningFailure ? r.status : r) body = (r isa FineTuningFailure ? r.response : "")
    @test r isa FineTuningFailure && r.status in (400, 403)
    err = JSON.parse(r.response)["error"]
    @test err["param"] ∉ ("seed", "integrations") && !occursin("Unrecognized request argument", err["message"])
end

@testset "Fine-tuning API live — checkpoint permissions refuse a project key" begin
    ck = "ft:gpt-4.1-mini-2025-04-14:unilm-live:made-up:ckpt-step-1"
    rs = [create_fine_tuning_checkpoint_permission(ck, ["proj_unilmlivemadeup"]),
          list_fine_tuning_checkpoint_permissions(ck; limit=1),
          delete_fine_tuning_checkpoint_permission(ck, "cp_unilmlivemadeup")]
    @info "checkpoint permissions with the project key" statuses = [r isa FineTuningFailure ? r.status : r for r in rs] bodies = [r isa FineTuningFailure ? r.response : "" for r in rs]
    @test all(r -> r isa FineTuningFailure && r.status in (401, 403), rs)
end

@testset "Graders live — validate and run a string_check grader" begin
    g = StringCheckGrader(name="exact", input="{{sample.output_text}}", reference="{{item.answer}}", operation="eq")
    v = validate_grader(g)
    @test v isa GraderValidateSuccess && v.grader["type"] == "string_check" && v.grader["operation"] == "eq"
    hit, miss = run_grader(g, "Paris"; item=Dict("answer" => "Paris")), run_grader(g, "Lyon"; item=Dict("answer" => "Paris"))
    @info "run_grader live" hit = (hit isa GraderRunSuccess ? hit.response.raw : hit) miss = (miss isa GraderRunSuccess ? miss.response.reward : miss)
    @test hit isa GraderRunSuccess && hit.response.reward == 1.0 && hit.response.metadata["type"] == "string_check"
    @test miss isa GraderRunSuccess && miss.response.reward == 0.0
    bad = validate_grader(Dict("type" => "string_check", "name" => "x", "input" => "a", "reference" => "b", "operation" => "equals"))
    @test bad isa FineTuningFailure && bad.status == 400
end

@testset "Webhook endpoints live — list the event types and the endpoints; make, test and delete one when the key may" begin
    types = list_webhook_event_types()
    @info "list_webhook_event_types live" result = (types isa WebhookEventTypesSuccess ? types.event_types : types)
    l = list_webhook_endpoints(; limit=1)
    @info "list_webhook_endpoints live" result = (l isa WebhookEndpointListSuccess ? length(l.response.data) : l)
    if l isa WebhookEndpointFailure
        # The key may not manage endpoints: the refusal is a typed failure, and so is every other verb's.
        @test l.status in (401, 403) && types isa WebhookEndpointFailure && types.status in (401, 403)
    else
        @test l isa WebhookEndpointListSuccess && types isa WebhookEventTypesSuccess && "batch.completed" in types.event_types
        made = create_webhook_endpoint(; name="unilm-live", url="https://example.com/unilm-live-webhook",
                                       event_types=["batch.completed"])
        @test made isa WebhookEndpointSuccess && startswith(something(made.response.signing_secret, ""), "whsec_")
        if made isa WebhookEndpointSuccess
            try
                id = made.response.id
                @test retrieve_webhook_endpoint(id) isa WebhookEndpointSuccess
                u = update_webhook_endpoint(id; event_types=["batch.completed", "batch.failed"])
                @test u isa WebhookEndpointSuccess && sort(u.response.event_types) == ["batch.completed", "batch.failed"]
                rot = rotate_webhook_endpoint_secret(id)
                @test rot isa WebhookEndpointSuccess && rot.response.signing_secret != made.response.signing_secret
                t = test_webhook_endpoint(id, "batch.completed")
                @info "test_webhook_endpoint live" result = (t isa WebhookTestSuccess ? t.response.raw : t)
                @test t isa WebhookTestSuccess || (t isa WebhookEndpointFailure && t.status == 500)
            finally
                d = delete_webhook_endpoint(made.response.id)
                @test d isa WebhookEndpointDeleteSuccess && d.id == made.response.id
            end
        end
        # Nothing this suite made is left behind.
        after = list_webhook_endpoints(; limit=100)
        @test after isa WebhookEndpointListSuccess && !any(e -> e.name == "unilm-live", after.response.data)
    end
    @test retrieve_webhook_endpoint("we_unilmlivemadeup") isa WebhookEndpointFailure
end

@testset "Safety live — a made-up case and alert are refused; a PNG's provenance check" begin
    c = retrieve_safety_case("C-unilmlivemadeup")
    a = retrieve_safety_alert("alert_" * "0"^32)
    @info "safety reads of made-up ids" case = (c isa SafetyFailure ? (c.status, c.response) : c) alert = (a isa SafetyFailure ? a.status : a)
    # A case read needs the key's `api.safety.read` scope: without it OpenAI answers 403 (2026-11-16), with it 404.
    @test c isa SafetyFailure && c.status in (403, 404)
    @test a isa SafetyFailure && a.status == 404
    p = check_content_provenance(joinpath(@__DIR__, "fixtures", "ollama", "red_circle.png"))
    @info "check_content_provenance live" result = (p isa ProvenanceCheckSuccess ? p.response.raw : p)
    @test p isa ProvenanceCheckSuccess && !isempty(p.response.results) &&
          all(r -> r.type in ("c2pa", "synthid") && r.outcome == "not_detected", p.response.results)
end

end  # if OPENAI_API_KEY
