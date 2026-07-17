@testset "Conversations API — config seam wiring" begin
    @test _reached_seam(create_conversation(service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(retrieve_conversation("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(update_conversation("conv_x", Dict("k"=>"v"); service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(delete_conversation("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(add_conversation_items("conv_x", Any[]; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(list_conversation_items("conv_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
    @test _reached_seam(delete_conversation_item("conv_x", "item_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ConversationCallError)
end
