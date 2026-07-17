@testset "Fine-tuning API — config seam wiring" begin
    @test _reached_seam(create_fine_tuning_job(model="gpt-4o", training_file="file_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _reached_seam(retrieve_fine_tuning_job("ft_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _reached_seam(cancel_fine_tuning_job("ft_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _reached_seam(list_fine_tuning_jobs(service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _reached_seam(list_fine_tuning_events("ft_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
    @test _reached_seam(list_fine_tuning_checkpoints("ft_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.FineTuningCallError)
end
