# Unit tests for RequestConfig and its resolution channels.

@testset "constructor rejects NaN and non-positive; Inf disables" begin
    float_fields = (:connect_timeout, :request_timeout, :stream_idle_timeout,
                    :total_deadline, :mcp_connect_timeout, :mcp_request_timeout)
    for f in float_fields, bad in (NaN, 0.0, -1.0)
        @test_throws ArgumentError RequestConfig(; (f => bad,)...)
    end
    for f in float_fields
        cfg = RequestConfig(; (f => Inf,)...)
        @test getfield(cfg, f) == Inf
    end
    @test_throws ArgumentError RequestConfig(max_attempts=0)
    @test_throws ArgumentError RequestConfig(max_attempts=-3)
    @test RequestConfig(max_attempts=1).max_attempts == 1
    # defaults are themselves valid and match the documented values
    d = RequestConfig()
    @test d.connect_timeout == 10.0
    @test d.request_timeout == 600.0
    @test d.stream_idle_timeout == 120.0
    @test d.total_deadline == 900.0
    @test d.max_attempts == 3
    @test d.mcp_connect_timeout == 120.0
    @test d.mcp_request_timeout == 120.0
end

@testset "copy-with-overrides revalidates and touches only named fields" begin
    base = RequestConfig()
    c = RequestConfig(base; request_timeout=5.0, max_attempts=7)
    @test c.request_timeout == 5.0
    @test c.max_attempts == 7
    for f in fieldnames(RequestConfig)
        f in (:request_timeout, :max_attempts) && continue
        @test getfield(c, f) == getfield(base, f)
    end
    @test_throws ArgumentError RequestConfig(base; connect_timeout=NaN)
    @test_throws ArgumentError RequestConfig(base; total_deadline=0.0)
    @test_throws ArgumentError RequestConfig(base; max_attempts=0)
    @test_throws MethodError RequestConfig(base; not_a_field=1.0)
end
