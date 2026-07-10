# ============================================================================
# P0 regression suite — falsification harness for the 2026-07-10 review
# (grounding/quality-review-2026-07-10.md). Every test states the FIXED
# contract wrapped in @test_broken. Fix PRs flip @test_broken → @test.
# An "Error (unexpected pass)" means the finding is refuted: delete the test
# and downgrade the finding. Fully offline (zero-spend).
# ============================================================================

using Sockets

# Streaming SSE mock: writes each element of `chunks` as its own flush with a
# 0.2s gap, so the client's readavailable() sees them as separate reads.
# (TCP may still coalesce under load — the affected tests note the flake
# direction: coalescing can only turn Broken into an unexpected pass, never
# into a spurious CI failure.)
function sse_mock_server(chunks::Vector{String})
    tcp = Sockets.listen(Sockets.localhost, 0)
    port = Int(Sockets.getsockname(tcp)[2])
    close(tcp)
    server = HTTP.listen!("127.0.0.1", port; stream=true, verbose=false) do http::HTTP.Stream
        while !eof(http)
            readavailable(http)              # drain the request body
        end
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.startwrite(http)
        for c in chunks
            write(http, c)
            flush(http)
            sleep(0.2)
        end
    end
    server, "http://127.0.0.1:$port"
end

@testset "P0 regression suite (review 2026-07-10)" begin

end
