@testset "Conversations API — config seam wiring" begin
    @test _seam_timeout(create_conversation(service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _seam_timeout(retrieve_conversation("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _seam_timeout(update_conversation("conv_x", Dict("k"=>"v"); service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _seam_timeout(delete_conversation("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _seam_timeout(add_conversation_items("conv_x", Any[]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _seam_timeout(list_conversation_items("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _seam_timeout(delete_conversation_item("conv_x", "item_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
end

@testset "Conversations API — a failure keeps the request id the service sent" begin
    r = _answered(() -> retrieve_conversation("conv_x"; service=URLProbe), 404;
                  headers=["x-request-id" => "req_conv"])
    @test r isa UniLM.ConversationFailure && r.request_id == "req_conv"
end

@testset "Conversations API — separator-bearing ids stay single path segments" begin
    t = _recorded_targets() do
        retrieve_conversation(_HOSTILE_ID; service=URLProbe)
        update_conversation(_HOSTILE_ID, Dict("k" => "v"); service=URLProbe)
        delete_conversation(_HOSTILE_ID; service=URLProbe)
        add_conversation_items(_HOSTILE_ID, Any[]; service=URLProbe)
        list_conversation_items(_HOSTILE_ID; limit=2, order=_HOSTILE_ID, after=_HOSTILE_ID, service=URLProbe)
        delete_conversation_item(_HOSTILE_ID, _HOSTILE_ID; service=URLProbe)
    end
    @test t == ["/v1/conversations/$_HOSTILE_ENC",
                "/v1/conversations/$_HOSTILE_ENC",
                "/v1/conversations/$_HOSTILE_ENC",
                "/v1/conversations/$_HOSTILE_ENC/items",
                "/v1/conversations/$_HOSTILE_ENC/items?limit=2&order=$_HOSTILE_ENC&after=$_HOSTILE_ENC",
                "/v1/conversations/$_HOSTILE_ENC/items/$_HOSTILE_ENC"]
    g = _recorded_targets() do
        delete_conversation_item("conv_abc123", "msg_abc123"; service=URLProbe)
        list_conversation_items("conv_abc123"; order="desc", after="msg_abc123", service=URLProbe)
    end
    @test g == ["/v1/conversations/conv_abc123/items/msg_abc123",
                "/v1/conversations/conv_abc123/items?order=desc&after=msg_abc123"]
end
