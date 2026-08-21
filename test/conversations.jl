@testset "Conversations API — config seam wiring" begin
    @test _reached_seam(create_conversation(service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(retrieve_conversation("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(update_conversation("conv_x", Dict("k"=>"v"); service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(delete_conversation("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(add_conversation_items("conv_x", Any[]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(list_conversation_items("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(delete_conversation_item("conv_x", "item_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
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
