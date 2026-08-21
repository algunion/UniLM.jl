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

@testset "Containers API — a separator-bearing id stays one path segment" begin
    cpath = tempname() * ".txt"
    write(cpath, "x")
    try
        t = _recorded_targets() do
            retrieve_container(_HOSTILE_ID; service=URLProbe)
            delete_container(_HOSTILE_ID; service=URLProbe)
            add_container_file(_HOSTILE_ID, cpath; service=URLProbe)
            list_containers(; limit=2, after=_HOSTILE_ID, service=URLProbe)
        end
        @test t == ["/v1/containers/$_HOSTILE_ENC",
                    "/v1/containers/$_HOSTILE_ENC",
                    "/v1/containers/$_HOSTILE_ENC/files",
                    "/v1/containers?limit=2&after=$_HOSTILE_ENC"]
        g = _recorded_targets() do
            add_container_file("cntr_abc123", cpath; service=URLProbe)
            list_containers(; after="cntr_abc123", service=URLProbe)
        end
        @test g == ["/v1/containers/cntr_abc123/files", "/v1/containers?after=cntr_abc123"]
    finally
        rm(cpath; force=true)
    end
end
