@testset "ImageGeneration" begin
    @testset "minimal creation" begin
        ig = ImageGeneration(prompt="A cute robot")
        @test ig.model == "gpt-image-1.5"
        @test ig.prompt == "A cute robot"
        @test ig.service == UniLM.OPENAIServiceEndpoint
        @test isnothing(ig.n)
        @test isnothing(ig.size)
        @test isnothing(ig.quality)
        @test isnothing(ig.background)
        @test isnothing(ig.output_format)
        @test isnothing(ig.output_compression)
        @test isnothing(ig.user)
    end

    @testset "full creation" begin
        ig = ImageGeneration(
            prompt="A logo",
            model="gpt-image-1.5",
            n=2,
            size="1024x1024",
            quality="high",
            background="transparent",
            output_format="png",
            output_compression=nothing,
            user="user_123"
        )
        @test ig.prompt == "A logo"
        @test ig.n == 2
        @test ig.size == "1024x1024"
        @test ig.quality == "high"
        @test ig.background == "transparent"
        @test ig.output_format == "png"
        @test ig.user == "user_123"
    end

    @testset "JSON serialization" begin
        ig = ImageGeneration(prompt="test", quality="high", size="1024x1024")
        lowered = JSON.lower(ig)
        @test lowered[:model] == "gpt-image-1.5"
        @test lowered[:prompt] == "test"
        @test lowered[:quality] == "high"
        @test lowered[:size] == "1024x1024"
        # nil fields excluded
        @test !haskey(lowered, :service)
        @test !haskey(lowered, :n)
        @test !haskey(lowered, :background)
        @test !haskey(lowered, :output_format)
        @test !haskey(lowered, :output_compression)
        @test !haskey(lowered, :user)
    end

    @testset "JSON round-trip" begin
        ig = ImageGeneration(prompt="A sunset", n=2, size="1536x1024", background="opaque")
        json_str = JSON.json(ig)
        parsed = JSON.parse(json_str)
        @test parsed["model"] == "gpt-image-1.5"
        @test parsed["prompt"] == "A sunset"
        @test parsed["n"] == 2
        @test parsed["size"] == "1536x1024"
        @test parsed["background"] == "opaque"
        @test !haskey(parsed, "service")
        @test !haskey(parsed, "quality")
    end
end

@testset "ImageObject" begin
    @testset "defaults" begin
        io = ImageObject()
        @test isnothing(io.b64_json)
        @test isnothing(io.revised_prompt)
    end

    @testset "with data" begin
        io = ImageObject(b64_json="aGVsbG8=", revised_prompt="A cute robot in watercolor")
        @test io.b64_json == "aGVsbG8="
        @test io.revised_prompt == "A cute robot in watercolor"
    end
end

@testset "ImageResponse" begin
    ir = ImageResponse(
        created=1713833628,
        data=[ImageObject(b64_json="aGVsbG8="), ImageObject(b64_json="d29ybGQ=")],
        usage=Dict{String,Any}("total_tokens" => 100),
        raw=Dict{String,Any}("created" => 1713833628)
    )
    @test ir.created == 1713833628
    @test length(ir.data) == 2
    @test ir.usage["total_tokens"] == 100
end

@testset "image_data" begin
    @testset "extract base64 data" begin
        ir = ImageResponse(
            created=1,
            data=[ImageObject(b64_json="abc123"), ImageObject(b64_json="def456")],
            raw=Dict{String,Any}()
        )
        imgs = image_data(ir)
        @test length(imgs) == 2
        @test imgs[1] == "abc123"
        @test imgs[2] == "def456"
    end

    @testset "skip nil entries" begin
        ir = ImageResponse(
            created=1,
            data=[ImageObject(b64_json="abc123"), ImageObject()],
            raw=Dict{String,Any}()
        )
        imgs = image_data(ir)
        @test length(imgs) == 1
    end

    @testset "on ImageSuccess" begin
        ir = ImageResponse(
            created=1,
            data=[ImageObject(b64_json="xyz")],
            raw=Dict{String,Any}()
        )
        s = ImageSuccess(response=ir)
        @test s isa UniLM.LLMRequestResponse
        @test image_data(s) == ["xyz"]
    end
end

@testset "Result types" begin
    @testset "ImageSuccess" begin
        ir = ImageResponse(created=1, data=ImageObject[], raw=Dict{String,Any}())
        s = ImageSuccess(response=ir)
        @test s isa UniLM.LLMRequestResponse
    end

    @testset "ImageFailure" begin
        f = ImageFailure(response="error body", status=400)
        @test f isa UniLM.LLMRequestResponse
        @test f.response == "error body"
        @test f.status == 400
    end

    @testset "ImageCallError" begin
        e = ImageCallError(error="timeout")
        @test e isa UniLM.LLMRequestResponse
        @test e.error == "timeout"
        @test isnothing(e.status)

        e2 = ImageCallError(error="server error", status=503)
        @test e2.status == 503
    end
end

@testset "parse_image_response" begin
    function make_response(body::Dict; status=200)
        body_bytes = Vector{UInt8}(JSON.json(body))
        HTTP.Response(status, [], body_bytes)
    end

    @testset "basic response" begin
        body = Dict(
            "created" => 1713833628,
            "data" => [
                Dict("b64_json" => "aGVsbG8="),
                Dict("b64_json" => "d29ybGQ=")
            ],
            "usage" => Dict(
                "total_tokens" => 100,
                "input_tokens" => 50,
                "output_tokens" => 50
            )
        )
        resp = make_response(body)
        ir = UniLM.parse_image_response(resp)
        @test ir.created == 1713833628
        @test length(ir.data) == 2
        @test ir.data[1].b64_json == "aGVsbG8="
        @test ir.data[2].b64_json == "d29ybGQ="
        @test ir.usage["total_tokens"] == 100
    end

    @testset "response with revised_prompt" begin
        body = Dict(
            "created" => 1713833628,
            "data" => [
                Dict("b64_json" => "abc", "revised_prompt" => "A cute baby sea otter swimming")
            ]
        )
        resp = make_response(body)
        ir = UniLM.parse_image_response(resp)
        @test ir.data[1].revised_prompt == "A cute baby sea otter swimming"
    end

    @testset "empty data" begin
        body = Dict("created" => 1, "data" => [])
        resp = make_response(body)
        ir = UniLM.parse_image_response(resp)
        @test isempty(ir.data)
    end
end

@testset "save_image" begin
    @testset "saves decoded data to file" begin
        # "hello" base64-encoded is "aGVsbG8="
        tmpfile = tempname()
        try
            save_image("aGVsbG8=", tmpfile)
            @test isfile(tmpfile)
            @test read(tmpfile, String) == "hello"
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end

    @testset "returns filepath" begin
        tmpfile = tempname()
        try
            result = save_image("dGVzdA==", tmpfile)
            @test result == tmpfile
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end
end

@testset "generate_image error handling" begin
    @testset "generate_image(::ImageGeneration) with invalid API key" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            ig = ImageGeneration(prompt="test")
            result = generate_image(ig)
            @test result isa ImageCallError || result isa ImageFailure
        end
    end

    @testset "generate_image(prompt; kwargs...) convenience" begin
        withenv("OPENAI_API_KEY" => "sk-invalid-test-key") do
            result = generate_image("test prompt", size="1024x1024")
            @test result isa ImageCallError || result isa ImageFailure
        end
    end
end

@testset "Constants - images path" begin
    @test UniLM.IMAGES_GENERATIONS_PATH == "/v1/images/generations"
end
