@testset "Containers API — config seam wiring" begin
    cpath = tempname() * ".txt"
    write(cpath, "probe")
    try
        @test _reached_seam(create_container(name="c", service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
        @test _reached_seam(retrieve_container("cntr_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
        @test _reached_seam(list_containers(service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
        @test _reached_seam(delete_container("cntr_x"; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
        @test _reached_seam(add_container_file("cntr_x", cpath; service=SeamProbe, config=_TINY_DEADLINE), UniLM.ContainerCallError)
    finally
        rm(cpath; force=true)
    end
end
