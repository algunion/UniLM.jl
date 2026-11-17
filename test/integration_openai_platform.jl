# test/integration_openai_platform.jl — live: OpenAI's fine-tuning, webhook-endpoint, safety, content
# provenance and skill verbs. Free calls but one container the skills testset loads its skills into (one
# 1 GB session, $0.0075): no job is created (the training file is made up), the ids read are made up, and
# an endpoint or a skill this suite makes is deleted in a `finally`. Needs UNILM_LIVE=1 and OPENAI_API_KEY.
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
    # These verbs send an admin key only; the project key sent as one is refused (no admin key is kept for the suite).
    ck = "ft:gpt-4.1-mini-2025-04-14:unilm-live:made-up:ckpt-step-1"
    key = ENV["OPENAI_API_KEY"]
    rs = [create_fine_tuning_checkpoint_permission(ck, ["proj_unilmlivemadeup"]; admin_key=key),
          list_fine_tuning_checkpoint_permissions(ck; limit=1, admin_key=key),
          delete_fine_tuning_checkpoint_permission(ck, "cp_unilmlivemadeup"; admin_key=key)]
    @info "checkpoint permissions with the project key" statuses = [r isa FineTuningFailure ? r.status : r for r in rs] bodies = [r isa FineTuningFailure ? r.response : "" for r in rs]
    @test all(r -> r isa FineTuningFailure && r.status in (401, 403), rs)
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

@testset "Skills API live — a skill and a second version, read, listed, downloaded, loaded into a container; deleted" begin
    # The Skills API bills nothing; the container bills one 1 GB session at its 5-minute minimum ($0.0075 by OpenAI's
    # pricing page) and is deleted at once. Everything made here is deleted, the skill in a `finally`.
    name = "unilm-live-" * string(rand(UInt32); base=36)
    function skill_dir(version)
        dir = mkpath(joinpath(mktempdir(), name))
        write(joinpath(dir, "SKILL.md"), "---\nname: $name\ndescription: Replies with the word pong (version $version). " *
                                         "Use it when asked to ping.\n---\nReply with the single word pong.\n")
        write(joinpath(dir, "notes.txt"), "UniLM live skill, version $version\n")
        dir
    end
    s = create_skill(skill_dir(1))
    @info "create_skill live" result = (s isa SkillSuccess ? s.response.raw : s)
    @test s isa SkillSuccess
    if s isa SkillSuccess
        id = s.response.id
        try
            @test (s.response.name, s.response.default_version, s.response.latest_version) == (name, "1", "1")
            @test occursin("version 1", s.response.description) && abs(s.response.created_at - time()) < 600
            v = create_skill_version(id, skill_dir(2); default=true)
            @test v isa SkillVersionSuccess && (v.response.skill_id, v.response.version, v.response.name) == (id, "2", name)
            # "A newly uploaded version can briefly return 404": read it back until it is visible, for up to 60 s
            # (on 17 November 2026 it stayed 404 for more than 10 s, and every later step depends on it).
            got = retrieve_skill_version(id, "2")
            for _ in 1:30
                got isa SkillFailure && got.status == 404 || break
                sleep(2)
                got = retrieve_skill_version(id, "2")
            end
            @test got isa SkillVersionSuccess && got.response.id == v.response.id && occursin("version 2", got.response.description)
            r = retrieve_skill(id)
            @test r isa SkillSuccess && (r.response.default_version, r.response.latest_version) == ("2", "2")
            @test update_skill(id; default_version=1).response.default_version == "1"
            @test update_skill(id; default_version="2").response.default_version == "2"
            listed = list_skills(; limit=100, order="desc")
            @test listed isa SkillListSuccess && any(x -> x.id == id, listed.response.data)
            versions = list_skill_versions(id; order="asc")
            @test versions isa SkillVersionListSuccess && [x.version for x in versions.response.data] == ["1", "2"]
            # The zips: the default version's, and version 1's.
            zip = skill_content(id)
            one = skill_content(id; version=1)
            @test zip isa FileContentSuccess && zip.content[1:4] == b"PK\x03\x04"
            @test one isa FileContentSuccess && one.content[1:4] == b"PK\x03\x04" && one.content != zip.content
            # The skill by reference and version 1's zip inline, in one container. An inline skill's name and
            # description are its SKILL.md's: OpenAI answered others with HTTP 400 "Inline skill name/description
            # must match the values in SKILL.md/Skills.md front matter." (17 November 2026).
            c = create_container(; name="unilm-live-skills",
                                 skills=[SkillReference(r.response),
                                         InlineSkill(one.content; name, description=s.response.description)])
            @info "create_container with a skill reference and an inline skill live" result = (c isa ContainerSuccess ? c.response.raw : c)
            @test c isa ContainerSuccess
            if c isa ContainerSuccess
                @test delete_container(c.response.id) isa ContainerDeleteSuccess
            end
            d = delete_skill_version(id, "1")
            @test d isa SkillDeleteSuccess && d.id == only(filter(x -> x.version == "1", versions.response.data)).id
            # The listing showed the deleted version for a moment (17 November 2026): read it until it is gone.
            listed_versions() = (l = list_skill_versions(id); l isa SkillVersionListSuccess ? [x.version for x in l.response.data] : l)
            after = listed_versions()
            for _ in 1:10
                after == ["2"] && break
                sleep(1)
                after = listed_versions()
            end
            @test after == ["2"]
        finally
            del = delete_skill(id)
            @test del isa SkillDeleteSuccess && del.id == id
        end
        gone = retrieve_skill(id)
        @test gone isa SkillFailure && gone.status == 404
    end
end

end  # if OPENAI_API_KEY
