# ─── Custom voices: OpenAI's voices and voice consents, Gemini's Voices API ───
# Included by test/audio.jl, whose helpers (`_SP_WAV`, `_mp_values`) it uses.

# Each voices wire at the scripted endpoint (VoicesProbe, platform_seam_fixtures.jl).
const _OAVoices = VoicesProbe{UniLM._OpenAIVoices}
const _GVoices = VoicesProbe{UniLM._GeminiVoices}

# Replies as the providers gave them (2026-11-17).
const _OA_VOICE = """{"id":"voice_6afc","created_at":1794902010,"object":"audio.voice","name":"Narrator","type":"prompt",
                      "prompt":"A calm, clear adult narrator."}"""
const _OA_CONSENT = """{"object":"audio.voice_consent","id":"cons_1234","name":"John Doe","language":"en-US","created_at":1734220800}"""
const _G_VOICE = """{"id":"voice_yqke","model":"models/gemini-3.8-flash-tts","type":"prompted","expire_time":"2027-11-17T07:53:46Z",
                     "display_name":"Narrator","prompted":{"input":"A calm, clear adult narrator."},"language_code":"en-US",
                     "sample_audio":{"mime_type":"audio/wav","data":"UklGRg=="}}"""

_vrefusal(f) = try f(); nothing catch x; x end

@testset "voices: the capability, and each service's own" begin
    @test has_capability(OPENAIServiceEndpoint, :voices) && has_capability(GEMINIServiceEndpoint, :voices)
    @test !has_capability(ANTHROPICServiceEndpoint, :voices) && !has_capability(OllamaEndpoint(), :voices)
    e = _vrefusal(() -> list_voices(; service=ANTHROPICServiceEndpoint))
    @test e isa ArgumentError && occursin("Voices API is not supported by", e.msg)
    @test UniLM._voices_wire(GEMINIServiceEndpoint) isa UniLM._GeminiVoices
    @test UniLM._voices_wire(OPENAIServiceEndpoint) isa UniLM._OpenAIVoices
    @test UniLM._voices_url(OPENAIServiceEndpoint) == "https://api.openai.com/v1/audio/voices"
    @test UniLM._voice_consents_url(OPENAIServiceEndpoint) == "https://api.openai.com/v1/audio/voice_consents"
    @test UniLM._voices_url(GEMINIServiceEndpoint) == "https://generativelanguage.googleapis.com/v1beta/voices"
end

@testset "create_voice on OpenAI: a prompt voice as JSON, a voice from a recording as a form" begin
    r, seen = _with_scripted((_, _) -> _json(200, _OA_VOICE)) do
        create_voice(; service=_OAVoices, name="Narrator", prompt="A calm, clear adult narrator.",
                     script_hint="Hello there.", model="auto")
    end
    @test r isa VoiceSuccess
    v = r.response
    @test (v.id, v.name, v.type, v.created_at, v.key, v.raw["prompt"]) ==
          ("voice_6afc", "Narrator", "prompt", 1794902010, nothing, "A calm, clear adult narrator.")
    req = only(seen)
    @test req.method == "POST" && req.target == "/voices"
    @test JSON.parse(req.body) == Dict("type" => "prompt", "name" => "Narrator", "prompt" => "A calm, clear adult narrator.",
                                       "script_hint" => "Hello there.", "model" => "auto")
    # Shown by its typed fields; the raw voice (a Gemini one holds a preview of a megabyte or more) by its size.
    @test sprint(show, v) == "Voice(id=\"voice_6afc\", name=\"Narrator\", type=\"prompt\", created_at=1794902010, key=nothing, raw=<6 keys>)"
    s, seen2 = _with_scripted((_, _) -> _json(200, replace(_OA_VOICE, "\"prompt\"" => "\"audio_sample\"", count=1))) do
        create_voice(; service=_OAVoices, name="Mine", audio_sample=_SP_WAV, consent="cons_1234")
    end
    @test s isa VoiceSuccess
    body = only(seen2).body
    @test (_mp_values(body, "type"), _mp_values(body, "name"), _mp_values(body, "consent")) ==
          (["audio_sample"], ["Mine"], ["cons_1234"])
    @test occursin("name=\"audio_sample\"; filename=\"secret_word.wav\"", body) && occursin(String(read(_SP_WAV)), body)
    # A reply that is no voice is a call error; the provider's status is its failure.
    for bad in ("""{"object":"audio.voice","id":"v","name":"n","type":"prompt"}""",       # no created_at
                """{"object":"list","id":"v","name":"n","type":"prompt","created_at":1}""")
        b, _ = _with_scripted((_, _) -> _json(200, bad)) do
            create_voice(; service=_OAVoices, name="n", prompt="p")
        end
        @test b isa VoiceCallError
    end
    f, _ = _with_scripted((_, _) -> _json(404, "{\"error\":{\"message\":\"Your organization does not have access to this endpoint.\"}}";
                                          headers=["x-request-id" => "req_v"])) do
        create_voice(; service=_OAVoices, name="n", audio_sample=_SP_WAV, consent="cons_1")
    end
    @test f isa VoiceFailure && f.status == 404 && f.request_id == "req_v" && occursin("does not have access", f.response)
end

@testset "create_voice: what a service does not take is refused before any request" begin
    big = tempname() * ".wav"
    write(big, zeros(UInt8, 10 * 1024^2 + 1))
    cases = (
        (_OAVoices, (; name="n"), "give a prompt"),
        (_OAVoices, (; name="n", prompt="p", audio_sample=_SP_WAV), "one of the two"),
        (_OAVoices, (; name="n", prompt="  "), "the prompt is blank"),
        (_OAVoices, (; prompt="p"), "1 to 256 characters (got 0)"),
        (_OAVoices, (; name=repeat("n", 257), prompt="p"), "1 to 256 characters (got 257)"),
        (_OAVoices, (; name="n", prompt=repeat("p", 2001)), "at most 2000 characters"),
        (_OAVoices, (; name="n", prompt="p", script_hint=" "), "script_hint is 1 to 2000 characters"),
        (_OAVoices, (; name="n", prompt="p", consent="cons_1"), "a consent goes with an audio_sample"),
        (_OAVoices, (; name="n", prompt="p", store=true, language="en-US"), "OpenAI takes no store, language"),
        (_OAVoices, (; name="n", prompt="p", base_voice="Puck"), "OpenAI takes no base_voice"),
        (_OAVoices, (; name="n", audio_sample=_SP_WAV), "names its speaker's consent recording"),
        (_OAVoices, (; name="n", audio_sample="/no/such.wav", consent="cons_1"), "audio_sample not found"),
        (_OAVoices, (; name="n", audio_sample=big, consent="cons_1"), "at most 10 MiB"),
        (_OAVoices, (; name="n", audio_sample=_SP_WAV, consent="cons_1", model="auto"), "a model for a prompt voice only"),
        (_GVoices, (; prompt="p", consent="cons_1"), "Gemini takes no consent"),
        (_GVoices, (; prompt="p", script_hint="Hi"), "Gemini takes no script_hint"),
        (_GVoices, (; audio_sample=_SP_WAV), "pass consent_audio"),
        (_GVoices, (; prompt="p", consent_audio=_SP_WAV), "consent_audio goes with an audio_sample"),
        (_GVoices, (; audio_sample=_SP_WAV, consent_audio=_SP_WAV, base_voice="Puck"), "a base_voice goes with a prompt"),
        (_GVoices, (; prompt="p", pitch="shrill"), "pitch is \"low\", \"medium\" or \"high\""))
    try
        _, seen = _with_scripted((_, _) -> _json(200, "{}")) do
            for (svc, kw, says) in cases
                e = _vrefusal(() -> create_voice(; service=svc, kw...))
                @test e isa ArgumentError && occursin(says, e.msg)
            end
        end
        @test isempty(seen)
    finally
        rm(big; force=true)
    end
end

@testset "create_voice on Gemini: a designed voice, a replicated one, stored or not" begin
    r, seen = _with_scripted((_, _) -> _json(200, _G_VOICE)) do
        create_voice(; service=_GVoices, name="Narrator", prompt="A calm, clear adult narrator.", model="gemini-3.8-flash-tts",
                     language="en-US", gender="male", pitch="low", base_voice="Puck")
    end
    @test r isa VoiceSuccess
    v = r.response
    @test (v.id, v.name, v.type, v.created_at, v.key) == ("voice_yqke", "Narrator", "prompted", nothing, nothing)
    @test v.raw["sample_audio"]["data"] == "UklGRg=="
    @test only(seen).method == "POST" && only(seen).target == "/voices"
    @test JSON.parse(only(seen).body) == Dict("store" => true, "voice" => Dict(
        "type" => "prompted", "model" => "gemini-3.8-flash-tts", "display_name" => "Narrator", "language_code" => "en-US",
        "gender" => "male", "pitch" => "low",
        "prompted" => Dict("input" => "A calm, clear adult narrator.", "base_voice" => "Puck")))
    clip = base64encode(read(_SP_WAV))
    k, seen2 = _with_scripted((_, _) -> _json(200, """{"type":"replicated","key":"voicekey_c3RhdGVsZXNz","expire_time":"2026-11-24T00:00:00Z"}""")) do
        create_voice(; service=_GVoices, audio_sample=_SP_WAV, consent_audio=_SP_WAV, store=false)
    end
    @test k isa VoiceSuccess && (k.response.id, k.response.key, k.response.type) == (nothing, "voicekey_c3RhdGVsZXNz", "replicated")
    @test JSON.parse(only(seen2).body) == Dict("store" => false, "voice" => Dict("type" => "replicated", "replicated" => Dict(
        "source_audio" => Dict("mime_type" => "audio/wav", "data" => clip),
        "consent_audio" => Dict("mime_type" => "audio/wav", "data" => clip))))
    # A voice with neither an id nor a key is none speak could use; a voice without its type is malformed.
    for bad in ("""{"type":"replicated"}""", """{"id":"voice_x"}""")
        b, _ = _with_scripted((_, _) -> _json(200, bad)) do
            create_voice(; service=_GVoices, prompt="p")
        end
        @test b isa VoiceCallError
    end
end

@testset "list_voices and retrieve_voice on Gemini; OpenAI has neither" begin
    page = """{"voices":[{"id":"achernar","type":"prebuilt","display_name":"Achernar","language_code":"en-US","gender":"female"},
                         $(_G_VOICE)],"next_page_token":"EY5O"}"""
    r, seen = _with_scripted((_, _) -> _json(200, page)) do
        list_voices(; service=_GVoices, limit=2, after="tok/1", type=["prebuilt", "prompted"], language="en-US",
                    gender="female", search="warm calm")
    end
    @test r isa VoiceListSuccess && r.response.has_more && r.response.raw["next_page_token"] == "EY5O"
    @test [(v.id, v.type, v.name) for v in r.response.data] == [("achernar", "prebuilt", "Achernar"), ("voice_yqke", "prompted", "Narrator")]
    @test only(seen).method == "GET" && only(seen).target ==
          "/voices?page_size=2&page_token=tok%2F1&type=prebuilt&type=prompted&language_code=en-US&gender=female&search=warm%20calm"
    last_page, _ = _with_scripted((_, _) -> _json(200, "{}")) do
        list_voices(; service=_GVoices)
    end
    @test last_page isa VoiceListSuccess && isempty(last_page.response.data) && !last_page.response.has_more
    g, seen2 = _with_scripted((_, _) -> _json(200, _G_VOICE)) do
        retrieve_voice(_HOSTILE_ID; service=_GVoices)
    end
    @test g isa VoiceSuccess && g.response.id == "voice_yqke"
    @test only(seen2).method == "GET" && only(seen2).target == "/voices/" * _HOSTILE_ENC
    nf, _ = _with_scripted((_, _) -> _json(404, "{\"error\":{\"code\":404,\"status\":\"NOT_FOUND\"}}")) do
        retrieve_voice("Puck"; service=_GVoices)
    end
    @test nf isa VoiceFailure && nf.status == 404
    _, seen3 = _with_scripted((_, _) -> _json(200, "{}")) do
        for f in (() -> list_voices(; service=_OAVoices), () -> retrieve_voice("voice_x"; service=_OAVoices))
            e = _vrefusal(f)
            @test e isa ArgumentError && occursin("OpenAI's API has no voice", e.msg) && occursin("GEMINIServiceEndpoint", e.msg)
        end
        for limit in (0, 1001)
            e = _vrefusal(() -> list_voices(; service=_GVoices, limit))
            @test e isa ArgumentError && occursin("1 to 1000 voices", e.msg)
        end
    end
    @test isempty(seen3)
end

@testset "delete_voice: OpenAI's deletion record, Gemini's empty reply" begin
    d, seen = _with_scripted((_, _) -> _json(200, """{"id":"voice_6afc","object":"audio.voice.deleted","deleted":true}""")) do
        delete_voice(_HOSTILE_ID; service=_OAVoices)
    end
    @test d isa VoiceDeleteSuccess && (d.id, d.deleted, d.object) == ("voice_6afc", true, "audio.voice.deleted")
    @test only(seen).method == "DELETE" && only(seen).target == "/voices/" * _HOSTILE_ENC
    for bad in ("""{"id":"v","object":"audio.voice","deleted":true}""", """{"id":"v","object":"audio.voice.deleted","deleted":false}""",
                """{"object":"audio.voice.deleted","deleted":true}""")
        b, _ = _with_scripted((_, _) -> _json(200, bad)) do
            delete_voice("v"; service=_OAVoices)
        end
        @test b isa VoiceCallError
    end
    g, seen2 = _with_scripted((_, _) -> _json(200, "{}")) do
        delete_voice("voice_yqke"; service=_GVoices)
    end
    @test g isa VoiceDeleteSuccess && (g.id, g.deleted, g.object) == ("voice_yqke", true, nothing)
    @test only(seen2).method == "DELETE" && only(seen2).target == "/voices/voice_yqke"
    nf, _ = _with_scripted((_, _) -> _json(404, "{\"error\":{\"message\":\"No voice found with id 'v'.\"}}")) do
        delete_voice("v"; service=_OAVoices)
    end
    @test nf isa VoiceFailure && nf.status == 404
end

@testset "voice consents: OpenAI's create, list, get, update and delete" begin
    c, seen = _with_scripted((_, _) -> _json(200, _OA_CONSENT)) do
        create_voice_consent(_SP_WAV; name="John Doe", language="en-US", service=_OAVoices)
    end
    @test c isa VoiceConsentSuccess &&
          (c.response.id, c.response.name, c.response.language, c.response.created_at) == ("cons_1234", "John Doe", "en-US", 1734220800)
    body = only(seen).body
    @test only(seen).method == "POST" && only(seen).target == "/voice_consents"
    @test (_mp_values(body, "name"), _mp_values(body, "language")) == (["John Doe"], ["en-US"])
    @test occursin("name=\"recording\"; filename=\"secret_word.wav\"", body)
    page = """{"object":"list","data":[$(_OA_CONSENT)],"first_id":"cons_1234","last_id":"cons_1234","has_more":true}"""
    l, seen2 = _with_scripted((_, _) -> _json(200, page)) do
        list_voice_consents(; limit=1, after="cons_0", service=_OAVoices)
    end
    @test l isa VoiceConsentListSuccess && l.response.has_more && only(l.response.data).id == "cons_1234" &&
          l.response.raw["last_id"] == "cons_1234"
    @test only(seen2).method == "GET" && only(seen2).target == "/voice_consents?limit=1&after=cons_0"
    (g, u, d), seen3 = _with_scripted((n, _) -> _json(200, n == 3 ?
            """{"object":"audio.voice_consent","id":"cons_1234","deleted":true}""" : _OA_CONSENT)) do
        (retrieve_voice_consent(_HOSTILE_ID; service=_OAVoices), update_voice_consent("cons_1234"; name="Jane", service=_OAVoices),
         delete_voice_consent("cons_1234"; service=_OAVoices))
    end
    @test g isa VoiceConsentSuccess && u isa VoiceConsentSuccess
    @test d isa VoiceDeleteSuccess && (d.id, d.deleted, d.object) == ("cons_1234", true, "audio.voice_consent")
    @test [(q.method, q.target) for q in seen3] == [("GET", "/voice_consents/" * _HOSTILE_ENC),
        ("POST", "/voice_consents/cons_1234"), ("DELETE", "/voice_consents/cons_1234")]
    @test JSON.parse(seen3[2].body) == Dict("name" => "Jane")
    # A page without has_more, or a consent without its language, is no reply of that kind.
    for bad in ("""{"object":"list","data":[]}""", """{"object":"list","data":[{"object":"audio.voice_consent","id":"c","name":"n","created_at":1}],"has_more":false}""")
        b, _ = _with_scripted((_, _) -> _json(200, bad)) do
            list_voice_consents(; service=_OAVoices)
        end
        @test b isa VoiceCallError
    end
    # Before any request: Gemini keeps no consents, a blank name or language, an out-of-range limit, a missing file.
    _, seen4 = _with_scripted((_, _) -> _json(200, "{}")) do
        for (f, says) in ((() -> list_voice_consents(; service=_GVoices), "voice consents are OpenAI's"),
                          (() -> create_voice_consent(_SP_WAV; name=" ", language="en-US", service=_OAVoices), "the name is blank"),
                          (() -> create_voice_consent(_SP_WAV; name="n", language="", service=_OAVoices), "the language is blank"),
                          (() -> create_voice_consent("/no/such.wav"; name="n", language="en-US", service=_OAVoices), "recording not found"),
                          (() -> list_voice_consents(; limit=101, service=_OAVoices), "limit is 1 to 100"),
                          (() -> update_voice_consent("c"; name="  ", service=_OAVoices), "the name is blank"))
            e = _vrefusal(f)
            @test e isa ArgumentError && occursin(says, e.msg)
        end
    end
    @test isempty(seen4)
end

@testset "voice verbs: a spent budget and a cancelled token send nothing" begin
    _, seen = _with_scripted((_, _) -> _json(200, _OA_VOICE)) do
        @test _seam_timeout(create_voice(; service=_OAVoices, name="n", prompt="p", config=_TINY_DEADLINE), VoiceCallError)
        @test _seam_timeout(list_voices(; service=_GVoices, config=_TINY_DEADLINE), VoiceCallError)
        tok = CancelToken(); cancel!(tok)
        r = delete_voice("v"; service=_OAVoices, cancel=tok)
        @test r isa VoiceCallError && r.cause isa UniLMCancelled
    end
    @test isempty(seen)
end
