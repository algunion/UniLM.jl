# test/cancel_everywhere.jl
# The `cancel` keyword of every platform, lifecycle and audio verb: a token cancelled before the
# call sends nothing and the call returns its family's call error; an explicit token wins over the
# ambient one of `with_cancel`, and an explicit `nothing` is the ambient one; a cancel while the
# server holds the reply ends the call within 2 s on each seam; `realtime_connect` keeps its
# exception. Local servers only.

# The Realtime WebSocket that `speak` and `translate` open, at a local address set per test.
struct _CERealtime <: UniLM.ServiceEndpoint end
const _ce_ws_base = Ref("")
UniLM._realtime_ws_url(::Type{_CERealtime}) = _ce_ws_base[]
UniLM.auth_header(::Type{_CERealtime}) = ["Authorization" => "Bearer t", "Content-Type" => "application/json"]
UniLM.provider_capabilities(::Type{_CERealtime}) = Set([:realtime, :speech])

# Gemini TTS at the scripted endpoint.
struct _CEGeminiSpeech <: UniLM.ServiceEndpoint end
UniLM._speech_wire(::Type{_CEGeminiSpeech}) = UniLM._GeminiSpeechWire()
UniLM._gemini_speech_base(::Type{_CEGeminiSpeech}) = _url_probe_base[]
UniLM.auth_header(::Type{_CEGeminiSpeech}) = ["x-goog-api-key" => "t", "Content-Type" => "application/json"]

const _CE_WAV = joinpath(@__DIR__, "fixtures", "ollama", "secret_word.wav")   # 16-bit PCM

# Every verb that sends a request from the platform, lifecycle, model lookup, token counting, audio, skills,
# Gemini caching, File Search, stored completion, embeddings and voices files, and the image verbs (streamed here).
const _CE_VERBS = (
    upload_file, list_files, retrieve_file, delete_file, file_content, poll_file, register_files,
    create_upload, add_upload_part, complete_upload, cancel_upload,
    create_vector_store, retrieve_vector_store, update_vector_store, list_vector_stores, delete_vector_store,
    search_vector_store, add_vector_store_file, list_vector_store_files, retrieve_vector_store_file,
    update_vector_store_file, delete_vector_store_file, vector_store_file_content,
    create_file_batch, retrieve_file_batch, cancel_file_batch, list_file_batch_files,
    create_conversation, retrieve_conversation, update_conversation, delete_conversation,
    add_conversation_items, list_conversation_items, retrieve_conversation_item, delete_conversation_item,
    create_container, retrieve_container, list_containers, delete_container, add_container_file,
    list_container_files, retrieve_container_file, container_file_content, delete_container_file,
    create_batch, retrieve_batch, cancel_batch, list_batches, batch_results, delete_batch, update_batch,
    create_fine_tuning_job, retrieve_fine_tuning_job, cancel_fine_tuning_job, pause_fine_tuning_job,
    resume_fine_tuning_job, list_fine_tuning_jobs, list_fine_tuning_events, list_fine_tuning_checkpoints,
    create_fine_tuning_checkpoint_permission, list_fine_tuning_checkpoint_permissions,
    delete_fine_tuning_checkpoint_permission,
    create_webhook_endpoint, list_webhook_endpoints, retrieve_webhook_endpoint, update_webhook_endpoint,
    delete_webhook_endpoint, rotate_webhook_endpoint_secret, test_webhook_endpoint, list_webhook_event_types,
    retrieve_safety_case, retrieve_safety_alert, check_content_provenance,
    retrieve_model, delete_model, count_tokens,
    moderate, speak, transcribe, translate, mint_realtime_secret,
    get_response, delete_response, list_input_items, cancel_response, compact_response, count_input_tokens,
    create_skill, list_skills, retrieve_skill, update_skill, delete_skill, skill_content, create_skill_version,
    list_skill_versions, retrieve_skill_version, delete_skill_version,
    generate_image, edit_image,
    create_cache, retrieve_cache, list_caches, update_cache, delete_cache,
    create_file_search_store, retrieve_file_search_store, list_file_search_stores, delete_file_search_store,
    upload_to_file_search_store, import_to_file_search_store, retrieve_file_search_operation,
    poll_file_search_operation, list_file_search_documents, retrieve_file_search_document, delete_file_search_document,
    list_chat_completions, retrieve_chat_completion, update_chat_completion, delete_chat_completion,
    list_chat_completion_messages, embeddingrequest!,
    create_voice, list_voices, retrieve_voice, delete_voice, create_voice_consent, list_voice_consents,
    retrieve_voice_consent, update_voice_consent, delete_voice_consent)

# Each method of those verbs, and each wire a verb speaks, as (label, the family's call error,
# kw -> the call with keywords `kw`). Every call is valid, so only the token decides whether it is
# sent. The endpoints are built when called: the scripted endpoint's address is known once it listens.
function _ce_rows(file::String)
    s = URLProbe
    ep() = GenericOpenAIEndpoint(_url_probe_base[], "")
    ollama() = OllamaEndpoint(base_url=_url_probe_base[])
    item = [Dict("type" => "message", "role" => "user", "content" => "hi")]
    # A count's headers are its provider's, which read the provider's key.
    counted(request) = kw -> withenv(() -> count_tokens(request; kw...), "ANTHROPIC_API_KEY" => "sk-ant-local-0000",
                                     "GEMINI_API_KEY" => "g-local", "OPENAI_API_KEY" => "sk-local")
    hi = [Message(Val(:user), "hi")]
    af, gf = FilesProbe{UniLM._AnthropicFiles}, GeminiFilesProbe
    osk, ask = SkillsProbe{UniLM._OpenAISkills}, SkillsProbe{UniLM._AnthropicSkills}
    skill = mkpath(joinpath(mktempdir(), "ce-skill"))
    write(joinpath(skill, "SKILL.md"), "---\nname: ce-skill\ndescription: Says hi\n---\nSay hi.\n")
    fs, store, doc = FileSearchProbe, "fileSearchStores/s1", "fileSearchStores/s1/documents/d1"
    oeb, geb = EmbedBatchProbe{UniLM._OpenAIBatches}, EmbedBatchProbe{UniLM._GeminiBatches}
    oe() = ["a" => Embeddings("hi"; service=oeb, model="text-embedding-3-small")]
    ge() = ["a" => Embeddings("hi"; service=geb, model="gemini-embedding-2")]
    oav, gv = VoicesProbe{UniLM._OpenAIVoices}, VoicesProbe{UniLM._GeminiVoices}
    Tuple{String,Type,Function}[
        ("upload_file(::FileUpload)", FileCallError,
         kw -> upload_file(FileUpload(file=file, purpose="batch", service=s); kw...)),
        ("upload_file(path, purpose)", FileCallError, kw -> upload_file(file, "batch"; service=s, kw...)),
        ("list_files", FileCallError, kw -> list_files(; service=s, kw...)),
        ("retrieve_file", FileCallError, kw -> retrieve_file("file_1"; service=s, kw...)),
        ("delete_file", FileCallError, kw -> delete_file("file_1"; service=s, kw...)),
        ("file_content", FileCallError, kw -> file_content("file_1"; service=s, kw...)),
        ("upload_file(path), Anthropic", FileCallError, kw -> upload_file(file; service=af, kw...)),
        ("upload_file(path), Gemini", FileCallError, kw -> upload_file(file; service=gf, kw...)),
        ("list_files, Anthropic", FileCallError, kw -> list_files(; service=af, kw...)),
        ("list_files, Gemini", FileCallError, kw -> list_files(; service=gf, kw...)),
        ("retrieve_file, Anthropic", FileCallError, kw -> retrieve_file("file_1"; service=af, kw...)),
        ("retrieve_file, Gemini", FileCallError, kw -> retrieve_file("files/f1"; service=gf, kw...)),
        ("delete_file, Anthropic", FileCallError, kw -> delete_file("file_1"; service=af, kw...)),
        ("delete_file, Gemini", FileCallError, kw -> delete_file("files/f1"; service=gf, kw...)),
        # Gemini serves no download of an uploaded file: file_content refuses it before any request.
        ("file_content, Anthropic", FileCallError, kw -> file_content("file_1"; service=af, kw...)),
        ("poll_file, OpenAI", FileCallError, kw -> poll_file("file_1"; service=s, kw...)),
        ("poll_file, Anthropic", FileCallError, kw -> poll_file("file_1"; service=af, kw...)),
        ("poll_file, Gemini", FileCallError, kw -> poll_file("files/f1"; service=gf, kw...)),
        ("register_files", FileCallError, kw -> register_files(["gs://bucket/o"]; service=gf, kw...)),
        ("create_upload", UploadCallError,
         kw -> create_upload(; filename="a.jsonl", purpose="batch", bytes=3, mime_type="application/jsonl",
                             service=s, kw...)),
        ("add_upload_part", UploadCallError, kw -> add_upload_part("upload_1", UInt8[1, 2]; service=s, kw...)),
        ("complete_upload", UploadCallError, kw -> complete_upload("upload_1", ["part_1"]; service=s, kw...)),
        ("cancel_upload", UploadCallError, kw -> cancel_upload("upload_1"; service=s, kw...)),
        ("create_vector_store", VectorStoreCallError, kw -> create_vector_store(; name="vs", service=s, kw...)),
        ("retrieve_vector_store", VectorStoreCallError, kw -> retrieve_vector_store("vs_1"; service=s, kw...)),
        ("update_vector_store", VectorStoreCallError, kw -> update_vector_store("vs_1"; name="vs", service=s, kw...)),
        ("list_vector_stores", VectorStoreCallError, kw -> list_vector_stores(; service=s, kw...)),
        ("delete_vector_store", VectorStoreCallError, kw -> delete_vector_store("vs_1"; service=s, kw...)),
        ("search_vector_store", VectorStoreCallError, kw -> search_vector_store("vs_1", "q"; service=s, kw...)),
        ("add_vector_store_file", VectorStoreCallError,
         kw -> add_vector_store_file("vs_1", "file_1"; service=s, kw...)),
        ("list_vector_store_files", VectorStoreCallError, kw -> list_vector_store_files("vs_1"; service=s, kw...)),
        ("retrieve_vector_store_file", VectorStoreCallError,
         kw -> retrieve_vector_store_file("vs_1", "file_1"; service=s, kw...)),
        ("update_vector_store_file", VectorStoreCallError,
         kw -> update_vector_store_file("vs_1", "file_1", Dict("k" => "v"); service=s, kw...)),
        ("delete_vector_store_file", VectorStoreCallError,
         kw -> delete_vector_store_file("vs_1", "file_1"; service=s, kw...)),
        ("vector_store_file_content", VectorStoreCallError,
         kw -> vector_store_file_content("vs_1", "file_1"; service=s, kw...)),
        ("create_file_batch", VectorStoreCallError, kw -> create_file_batch("vs_1", ["file_1"]; service=s, kw...)),
        ("create_file_batch(::Vector{VectorStoreFileEntry})", VectorStoreCallError,
         kw -> create_file_batch("vs_1", [VectorStoreFileEntry(file_id="file_1")]; service=s, kw...)),
        ("retrieve_file_batch", VectorStoreCallError,
         kw -> retrieve_file_batch("vs_1", "vsfb_1"; service=s, kw...)),
        ("cancel_file_batch", VectorStoreCallError, kw -> cancel_file_batch("vs_1", "vsfb_1"; service=s, kw...)),
        ("list_file_batch_files", VectorStoreCallError,
         kw -> list_file_batch_files("vs_1", "vsfb_1"; service=s, kw...)),
        ("create_conversation", ConversationCallError, kw -> create_conversation(; service=s, kw...)),
        ("retrieve_conversation", ConversationCallError, kw -> retrieve_conversation("conv_1"; service=s, kw...)),
        ("update_conversation", ConversationCallError,
         kw -> update_conversation("conv_1", Dict("k" => "v"); service=s, kw...)),
        ("delete_conversation", ConversationCallError, kw -> delete_conversation("conv_1"; service=s, kw...)),
        ("add_conversation_items", ConversationCallError,
         kw -> add_conversation_items("conv_1", item; service=s, kw...)),
        ("list_conversation_items", ConversationCallError,
         kw -> list_conversation_items("conv_1"; service=s, kw...)),
        ("retrieve_conversation_item", ConversationCallError,
         kw -> retrieve_conversation_item("conv_1", "item_1"; service=s, kw...)),
        ("delete_conversation_item", ConversationCallError,
         kw -> delete_conversation_item("conv_1", "item_1"; service=s, kw...)),
        ("create_container", ContainerCallError, kw -> create_container(; name="c", service=s, kw...)),
        ("retrieve_container", ContainerCallError, kw -> retrieve_container("cntr_1"; service=s, kw...)),
        ("list_containers", ContainerCallError, kw -> list_containers(; service=s, kw...)),
        ("delete_container", ContainerCallError, kw -> delete_container("cntr_1"; service=s, kw...)),
        ("add_container_file", ContainerCallError, kw -> add_container_file("cntr_1", file; service=s, kw...)),
        ("list_container_files", ContainerCallError, kw -> list_container_files("cntr_1"; service=s, kw...)),
        ("retrieve_container_file", ContainerCallError,
         kw -> retrieve_container_file("cntr_1", "cfile_1"; service=s, kw...)),
        ("container_file_content", ContainerCallError,
         kw -> container_file_content("cntr_1", "cfile_1"; service=s, kw...)),
        ("delete_container_file", ContainerCallError,
         kw -> delete_container_file("cntr_1", "cfile_1"; service=s, kw...)),
        ("create_batch", BatchCallError, kw -> create_batch("file_1", "/v1/responses"; service=s, kw...)),
        ("retrieve_batch", BatchCallError, kw -> retrieve_batch("batch_1"; service=s, kw...)),
        ("cancel_batch", BatchCallError, kw -> cancel_batch("batch_1"; service=s, kw...)),
        ("list_batches", BatchCallError, kw -> list_batches(; service=s, kw...)),
        ("create_batch(requests), OpenAI", BatchCallError,
         kw -> create_batch(["a" => Chat(service=_OAB, model="gpt-6-luna", messages=hi)]; kw...)),
        ("create_batch(requests), Anthropic", BatchCallError,
         kw -> create_batch(["a" => Chat(service=_ANB, model="claude-haiku-5-5", messages=hi)]; kw...)),
        ("create_batch(requests), Gemini", BatchCallError,
         kw -> create_batch(["a" => Chat(service=_GMB, model="gemini-3.8-flash", messages=hi)]; kw...)),
        ("retrieve_batch, Anthropic", BatchCallError, kw -> retrieve_batch("msgbatch_1"; service=_ANB, kw...)),
        ("retrieve_batch, Gemini", BatchCallError, kw -> retrieve_batch("batches/b1"; service=_GMB, kw...)),
        ("cancel_batch, Anthropic", BatchCallError, kw -> cancel_batch("msgbatch_1"; service=_ANB, kw...)),
        ("cancel_batch, Gemini", BatchCallError, kw -> cancel_batch("batches/b1"; service=_GMB, kw...)),
        ("list_batches, Anthropic", BatchCallError, kw -> list_batches(; service=_ANB, kw...)),
        ("list_batches, Gemini", BatchCallError, kw -> list_batches(; service=_GMB, kw...)),
        ("batch_results, OpenAI", BatchCallError, kw -> batch_results("batch_1"; service=s, kw...)),
        ("batch_results, Anthropic", BatchCallError,
         kw -> batch_results("msgbatch_1"; service=_ANB, kw...)),
        ("batch_results, Gemini", BatchCallError, kw -> batch_results("batches/b1"; service=_GMB, kw...)),
        ("delete_batch, Anthropic", BatchCallError, kw -> delete_batch("msgbatch_1"; service=_ANB, kw...)),
        ("delete_batch, Gemini", BatchCallError, kw -> delete_batch("batches/b1"; service=_GMB, kw...)),
        ("update_batch, Gemini", BatchCallError,
         kw -> update_batch("batches/b1"; priority=1, service=_GMB, kw...)),
        ("create_fine_tuning_job", FineTuningCallError,
         kw -> create_fine_tuning_job(; model="m", training_file="file_1", service=s, kw...)),
        ("retrieve_fine_tuning_job", FineTuningCallError,
         kw -> retrieve_fine_tuning_job("ftjob_1"; service=s, kw...)),
        ("cancel_fine_tuning_job", FineTuningCallError, kw -> cancel_fine_tuning_job("ftjob_1"; service=s, kw...)),
        ("list_fine_tuning_jobs", FineTuningCallError, kw -> list_fine_tuning_jobs(; service=s, kw...)),
        ("list_fine_tuning_events", FineTuningCallError,
         kw -> list_fine_tuning_events("ftjob_1"; service=s, kw...)),
        ("list_fine_tuning_checkpoints", FineTuningCallError,
         kw -> list_fine_tuning_checkpoints("ftjob_1"; service=s, kw...)),
        ("pause_fine_tuning_job", FineTuningCallError, kw -> pause_fine_tuning_job("ftjob_1"; service=s, kw...)),
        ("resume_fine_tuning_job", FineTuningCallError, kw -> resume_fine_tuning_job("ftjob_1"; service=s, kw...)),
        ("create_fine_tuning_checkpoint_permission", FineTuningCallError,
         kw -> create_fine_tuning_checkpoint_permission("ft:m:ckpt", ["proj_1"]; admin_key="sk-admin", service=s, kw...)),
        ("list_fine_tuning_checkpoint_permissions", FineTuningCallError,
         kw -> list_fine_tuning_checkpoint_permissions("ft:m:ckpt"; admin_key="sk-admin", service=s, kw...)),
        ("delete_fine_tuning_checkpoint_permission", FineTuningCallError,
         kw -> delete_fine_tuning_checkpoint_permission("ft:m:ckpt", "cp_1"; admin_key="sk-admin", service=s, kw...)),
        ("create_webhook_endpoint", WebhookEndpointCallError,
         kw -> create_webhook_endpoint(; name="n", url="https://example.com/hook", event_types=["batch.completed"],
                                       service=s, kw...)),
        ("list_webhook_endpoints", WebhookEndpointCallError, kw -> list_webhook_endpoints(; service=s, kw...)),
        ("retrieve_webhook_endpoint", WebhookEndpointCallError,
         kw -> retrieve_webhook_endpoint("we_1"; service=s, kw...)),
        ("update_webhook_endpoint", WebhookEndpointCallError,
         kw -> update_webhook_endpoint("we_1"; name="n", service=s, kw...)),
        ("delete_webhook_endpoint", WebhookEndpointCallError, kw -> delete_webhook_endpoint("we_1"; service=s, kw...)),
        ("rotate_webhook_endpoint_secret", WebhookEndpointCallError,
         kw -> rotate_webhook_endpoint_secret("we_1"; service=s, kw...)),
        ("test_webhook_endpoint", WebhookEndpointCallError,
         kw -> test_webhook_endpoint("we_1", "batch.completed"; service=s, kw...)),
        ("list_webhook_event_types", WebhookEndpointCallError, kw -> list_webhook_event_types(; service=s, kw...)),
        ("retrieve_safety_case", SafetyCallError, kw -> retrieve_safety_case("C-1"; service=s, kw...)),
        ("retrieve_safety_alert", SafetyCallError, kw -> retrieve_safety_alert("alert_1"; service=s, kw...)),
        ("check_content_provenance", SafetyCallError, kw -> check_content_provenance(_CE_WAV; service=s, kw...)),
        ("retrieve_model, OpenAI", ModelListCallError,
         kw -> retrieve_model("gpt-6-luna"; service=ListingProbe{UniLM._OpenAIModels}, kw...)),
        ("retrieve_model, Anthropic", ModelListCallError,
         kw -> retrieve_model("claude-haiku-5-5"; service=ListingProbe{UniLM._AnthropicModels}, kw...)),
        ("retrieve_model, Gemini", ModelListCallError,
         kw -> retrieve_model("gemini-3.8-flash"; service=ListingProbe{UniLM._GeminiModels}, kw...)),
        ("delete_model, OpenAI", ModelListCallError,
         kw -> delete_model("ft:m:org:x:1"; service=ListingProbe{UniLM._OpenAIModels}, kw...)),
        ("delete_model, Ollama", OllamaCallError, kw -> delete_model("phonon"; service=ollama(), kw...)),
        ("count_tokens, Anthropic", TokenCountCallError,
         counted(Chat(service=_CountAt(ANTHROPICServiceEndpoint), model="claude-haiku-5-5", messages=hi))),
        ("count_tokens, Gemini", TokenCountCallError,
         counted(Chat(service=_CountAt(GEMINIServiceEndpoint), model="gemini-3.8-flash", messages=hi))),
        ("count_tokens, OpenAI", TokenCountCallError,
         counted(Respond(service=_CountAt(OPENAIServiceEndpoint), model="gpt-6-luna", input="hi"))),
        ("moderate", ModerationCallError, kw -> moderate("hi"; service=s, kw...)),
        ("mint_realtime_secret", RealtimeCallError, kw -> mint_realtime_secret(; service=s, kw...)),
        ("speak(::SpeechRequest), Realtime", AudioCallError,
         kw -> speak(SpeechRequest(input="Hi.", service=_CERealtime); kw...)),
        ("speak(input), Realtime", AudioCallError, kw -> speak("Hi."; service=_CERealtime, kw...)),
        ("speak(input), Gemini TTS", AudioCallError, kw -> speak("Hi."; service=_CEGeminiSpeech, kw...)),
        ("speak(input), Ollama", AudioCallError, kw -> speak("Hi."; service=ollama(), model="phonon", kw...)),
        ("transcribe(::TranscriptionRequest)", AudioCallError,
         kw -> transcribe(TranscriptionRequest(file=_CE_WAV, service=s); kw...)),
        ("transcribe(path)", AudioCallError, kw -> transcribe(_CE_WAV; service=s, kw...)),
        ("translate, Realtime", AudioCallError, kw -> translate(_CE_WAV; service=_CERealtime, kw...)),
        ("translate, Ollama", AudioCallError,
         kw -> translate(_CE_WAV; service=ollama(), model="gemma4:e4b", kw...)),
        ("get_response", ResponseCallError, kw -> get_response("resp_1"; service=ep(), kw...)),
        ("delete_response", ResponseCallError, kw -> delete_response("resp_1"; service=ep(), kw...)),
        ("list_input_items", ResponseCallError, kw -> list_input_items("resp_1"; service=ep(), kw...)),
        ("cancel_response", ResponseCallError, kw -> cancel_response("resp_1"; service=ep(), kw...)),
        ("compact_response", ResponseCallError, kw -> compact_response(; model="m", input="hi", service=ep(), kw...)),
        ("count_input_tokens", ResponseCallError,
         kw -> count_input_tokens(; model="m", input="hi", service=ep(), kw...)),
        ("create_skill, OpenAI", SkillCallError, kw -> create_skill(skill; service=osk, kw...)),
        ("create_skill, Anthropic", SkillCallError, kw -> create_skill(skill; service=ask, kw...)),
        ("list_skills, OpenAI", SkillCallError, kw -> list_skills(; service=osk, kw...)),
        ("list_skills, Anthropic", SkillCallError, kw -> list_skills(; service=ask, kw...)),
        ("retrieve_skill, OpenAI", SkillCallError, kw -> retrieve_skill("skill_1"; service=osk, kw...)),
        ("retrieve_skill, Anthropic", SkillCallError, kw -> retrieve_skill("skill_1"; service=ask, kw...)),
        # Anthropic has no default version: update_skill refuses it before any request.
        ("update_skill, OpenAI", SkillCallError, kw -> update_skill("skill_1"; default_version=2, service=osk, kw...)),
        ("delete_skill, OpenAI", SkillCallError, kw -> delete_skill("skill_1"; service=osk, kw...)),
        ("delete_skill, Anthropic", SkillCallError, kw -> delete_skill("skill_1"; service=ask, kw...)),
        ("skill_content, OpenAI", SkillCallError, kw -> skill_content("skill_1"; service=osk, kw...)),
        ("skill_content, Anthropic", SkillCallError, kw -> skill_content("skill_1"; version="1", service=ask, kw...)),
        ("create_skill_version, OpenAI", SkillCallError, kw -> create_skill_version("skill_1", skill; service=osk, kw...)),
        ("create_skill_version, Anthropic", SkillCallError,
         kw -> create_skill_version("skill_1", skill; service=ask, kw...)),
        ("list_skill_versions, OpenAI", SkillCallError, kw -> list_skill_versions("skill_1"; service=osk, kw...)),
        ("list_skill_versions, Anthropic", SkillCallError, kw -> list_skill_versions("skill_1"; service=ask, kw...)),
        ("retrieve_skill_version, OpenAI", SkillCallError,
         kw -> retrieve_skill_version("skill_1", "1"; service=osk, kw...)),
        ("retrieve_skill_version, Anthropic", SkillCallError,
         kw -> retrieve_skill_version("skill_1", "1"; service=ask, kw...)),
        ("delete_skill_version, OpenAI", SkillCallError, kw -> delete_skill_version("skill_1", "1"; service=osk, kw...)),
        ("delete_skill_version, Anthropic", SkillCallError,
         kw -> delete_skill_version("skill_1", "1"; service=ask, kw...)),
        ("generate_image, streamed", ImageCallError,
         kw -> generate_image("p"; model="m", stream=true, service=s, kw...)),
        ("edit_image, streamed", ImageCallError,
         kw -> edit_image(ImageRef(file_id="file_1"), "p"; model="m", stream=true, service=s, kw...)),
        ("create_cache", CacheCallError,
         kw -> create_cache(Chat(service=CacheProbe, model="gemini-3.8-flash", messages=hi); kw...)),
        ("retrieve_cache", CacheCallError, kw -> retrieve_cache("cachedContents/c1"; service=CacheProbe, kw...)),
        ("list_caches", CacheCallError, kw -> list_caches(; service=CacheProbe, kw...)),
        ("update_cache", CacheCallError, kw -> update_cache("cachedContents/c1"; ttl=60, service=CacheProbe, kw...)),
        ("delete_cache", CacheCallError, kw -> delete_cache("cachedContents/c1"; service=CacheProbe, kw...)),
        ("create_file_search_store", FileSearchCallError, kw -> create_file_search_store(; service=fs, kw...)),
        ("retrieve_file_search_store", FileSearchCallError, kw -> retrieve_file_search_store(store; service=fs, kw...)),
        ("list_file_search_stores", FileSearchCallError, kw -> list_file_search_stores(; service=fs, kw...)),
        ("delete_file_search_store", FileSearchCallError, kw -> delete_file_search_store(store; service=fs, kw...)),
        ("upload_to_file_search_store", FileSearchCallError,
         kw -> upload_to_file_search_store(store, file; service=fs, kw...)),
        ("import_to_file_search_store", FileSearchCallError,
         kw -> import_to_file_search_store(store, "files/f1"; service=fs, kw...)),
        ("retrieve_file_search_operation", FileSearchCallError,
         kw -> retrieve_file_search_operation(store * "/operations/o1"; service=fs, kw...)),
        ("poll_file_search_operation", FileSearchCallError,
         kw -> poll_file_search_operation(store * "/operations/o1"; service=fs, kw...)),
        ("list_file_search_documents", FileSearchCallError, kw -> list_file_search_documents(store; service=fs, kw...)),
        ("retrieve_file_search_document", FileSearchCallError,
         kw -> retrieve_file_search_document(doc; service=fs, kw...)),
        ("delete_file_search_document", FileSearchCallError, kw -> delete_file_search_document(doc; service=fs, kw...)),
        ("list_chat_completions", StoredCompletionCallError, kw -> list_chat_completions(; service=StoreProbe, kw...)),
        ("retrieve_chat_completion", StoredCompletionCallError,
         kw -> retrieve_chat_completion("chatcmpl-1"; service=StoreProbe, kw...)),
        ("update_chat_completion", StoredCompletionCallError,
         kw -> update_chat_completion("chatcmpl-1", Dict("k" => "v"); service=StoreProbe, kw...)),
        ("delete_chat_completion", StoredCompletionCallError,
         kw -> delete_chat_completion("chatcmpl-1"; service=StoreProbe, kw...)),
        ("list_chat_completion_messages", StoredCompletionCallError,
         kw -> list_chat_completion_messages("chatcmpl-1"; service=StoreProbe, kw...)),
        ("embeddingrequest!, Gemini", EmbeddingCallError,
         kw -> embeddingrequest!(Embeddings("hi"; service=GeminiEmbedProbe, model="gemini-embedding-2"); kw...)),
        ("create_batch(embeddings), OpenAI", BatchCallError, kw -> create_batch(oe(); kw...)),
        ("create_batch(embeddings), Gemini", BatchCallError, kw -> create_batch(ge(); kw...)),
        ("batch_results(requests=embeddings), OpenAI", BatchCallError,
         kw -> batch_results("batch_1"; service=oeb, requests=oe(), kw...)),
        ("batch_results(requests=embeddings), Gemini", BatchCallError,
         kw -> batch_results("batches/b1"; service=geb, requests=ge(), kw...)),
        # A streamed transcription is a Task; its result is the call's.
        ("transcribe(path), streamed", AudioCallError, kw -> fetch(transcribe(_CE_WAV; stream=true, service=s, kw...))),
        ("transcribe(path), Gemini", AudioCallError, kw -> transcribe(_CE_WAV; service=GeminiTranscribeProbe, kw...)),
        ("create_voice, OpenAI", VoiceCallError, kw -> create_voice(; name="n", prompt="p", service=oav, kw...)),
        ("create_voice, Gemini", VoiceCallError, kw -> create_voice(; prompt="p", service=gv, kw...)),
        # OpenAI's API has no voice listing or lookup, and only OpenAI keeps voice consents: refused before any request.
        ("list_voices, Gemini", VoiceCallError, kw -> list_voices(; service=gv, kw...)),
        ("retrieve_voice, Gemini", VoiceCallError, kw -> retrieve_voice("voice_1"; service=gv, kw...)),
        ("delete_voice, OpenAI", VoiceCallError, kw -> delete_voice("voice_1"; service=oav, kw...)),
        ("delete_voice, Gemini", VoiceCallError, kw -> delete_voice("voice_1"; service=gv, kw...)),
        ("create_voice_consent", VoiceCallError,
         kw -> create_voice_consent(_CE_WAV; name="n", language="en-US", service=oav, kw...)),
        ("list_voice_consents", VoiceCallError, kw -> list_voice_consents(; service=oav, kw...)),
        ("retrieve_voice_consent", VoiceCallError, kw -> retrieve_voice_consent("cons_1"; service=oav, kw...)),
        ("update_voice_consent", VoiceCallError, kw -> update_voice_consent("cons_1"; name="n", service=oav, kw...)),
        ("delete_voice_consent", VoiceCallError, kw -> delete_voice_consent("cons_1"; service=oav, kw...)),
    ]
end

# A token's outcome, as the image verbs return it: the family's call error, no HTTP status, and the
# cancellation as its cause.
_ce_cancelled(r, ::Type{T}) where {T} =
    r isa T && isnothing(r.status) && r.cause isa UniLMCancelled && r.cause.source === :token

@testset "every method of every platform, lifecycle and audio verb takes `cancel` as its last keyword" begin
    @test length(_CE_VERBS) == 130
    for f in _CE_VERBS, m in methods(f)
        named = filter(k -> !endswith(String(k), "..."), Base.kwarg_decl(m))
        @test (f, last(named)) == (f, :cancel)
    end
end

@testset "cancel= on every platform, lifecycle and audio verb: a cancelled token sends nothing" begin
    file = tempname() * ".jsonl"
    write(file, "{}\n")
    hits = Threads.Atomic{Int}(0)
    fired = cancel!(CancelToken())
    # The result of `call()`, or what it threw, and the requests it sent.
    sent(call) = (before = hits[]; r = try call() catch e; e end; (r, hits[] - before))
    try
        results, _ = _with_scripted((_, _) -> (Threads.atomic_add!(hits, 1); _json(200, "{}"))) do
            _ce_ws_base[] = replace(_url_probe_base[], r"^http" => "ws")
            [(label, T, (
                 "a cancelled token" => sent(() -> call((; cancel=fired))),
                 # `cancel=nothing` is the ambient token, as no keyword is.
                 "cancel=nothing in a cancelled scope" => sent(() -> with_cancel(() -> call((; cancel=nothing)), fired)),
                 "no keyword in a cancelled scope" => sent(() -> with_cancel(() -> call((;)), fired)),
                 # A live token passed to the call wins over a cancelled ambient one.
                 "a live token in a cancelled scope" =>
                     sent(() -> with_cancel(() -> call((; cancel=CancelToken())), fired))))
             for (label, T, call) in _ce_rows(file)]
        end
        @test length(results) == 182
        for (label, T, modes) in results, (mode, (r, n)) in modes
            @testset "$label, $mode" begin
                if startswith(mode, "a live token")
                    @test !_ce_cancelled(r, T)
                    @test n >= 1        # the request goes out
                else
                    @test _ce_cancelled(r, T)
                    @test n == 0        # nothing sent
                end
            end
        end
    finally
        rm(file; force=true)
    end
end

# Seconds from `cancel!` to the call's result, per seam, for the harness that runs these tests under
# coverage and CPU contention to read.
const _CE_LATENCY = Dict{String,Float64}()

# Run `call(tok)` on its own task while the scripted endpoint holds each reply up to 20 s, and cancel
# `tok` 0.3 s after the request arrived. Returns the result (or what the call threw), the seconds from
# the cancel to the result, and the requests the endpoint saw.
function _ce_cancel_held(call::Function)
    arrived, release = Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false)
    hold = (_, _) -> (arrived[] = true; timedwait(() -> release[], 20.0); _json(200, "{}"))
    (r, after), seen = _with_scripted(hold) do
        tok = CancelToken()
        t = Threads.@spawn (try call(tok) catch e; e end, time_ns())
        timedwait(() -> arrived[] || istaskdone(t), 25.0)
        sleep(0.3)
        cancelled_at = time_ns()
        cancel!(tok)
        finished = timedwait(() -> istaskdone(t), 25.0) === :ok
        release[] = true
        finished || return (nothing, Inf)
        res, done_at = fetch(t)
        (res, (Int64(done_at) - Int64(cancelled_at)) / 1e9)
    end
    (; r, after, requests=length(seen))
end

# Run `call(tok)` against a local Realtime WebSocket that opens the session and then answers no client
# event, and cancel `tok` 0.3 s after the first client event arrived. Returns the result, the seconds
# from the cancel to it, and the connections the server took.
function _ce_cancel_in_session(call::Function)
    opened, heard = Threads.Atomic{Int}(0), Threads.Atomic{Bool}(false)
    srv = HTTP.WebSockets.listen!("127.0.0.1", 0) do ws
        Threads.atomic_add!(opened, 1)
        try
            HTTP.WebSockets.send(ws, JSON.json(Dict("type" => "session.created")))
            for _ in ws
                heard[] = true
            end
        catch e
            e isa HTTP.WebSockets.WebSocketError || e isa Base.IOError || rethrow()
        end
    end
    _ce_ws_base[] = "ws://" * HTTP.WebSockets.server_addr(srv)
    try
        tok = CancelToken()
        t = Threads.@spawn (try call(tok) catch e; e end, time_ns())
        timedwait(() -> heard[] || istaskdone(t), 25.0)
        sleep(0.3)
        cancelled_at = time_ns()
        cancel!(tok)
        finished = timedwait(() -> istaskdone(t), 25.0) === :ok
        finished || return (; r=nothing, after=Inf, connections=opened[])
        r, done_at = fetch(t)
        (; r, after=(Int64(done_at) - Int64(cancelled_at)) / 1e9, connections=opened[])
    finally
        close(srv)
    end
end

@testset "cancel=: a cancel while the server holds the reply ends the call within 2 s, on every seam" begin
    cfg = RequestConfig(max_attempts=3, total_deadline=60.0)
    file = tempname() * ".jsonl"
    write(file, "{}\n")
    try
        for (seam, T, call) in (
                ("_http", BatchCallError, tok -> cancel_batch("batch_1"; service=URLProbe, config=cfg, cancel=tok)),
                ("_http_with_retries", ModerationCallError,
                 tok -> moderate("hi"; service=URLProbe, config=cfg, cancel=tok)),
                ("multipart, _http", FileCallError,
                 tok -> upload_file(file, "batch"; service=URLProbe, config=cfg, cancel=tok)),
                ("multipart, _http_with_retries", AudioCallError,
                 tok -> transcribe(_CE_WAV; service=URLProbe, config=cfg, cancel=tok)),
                ("_lifecycle_request", ResponseCallError,
                 tok -> get_response("resp_1"; service=GenericOpenAIEndpoint(_url_probe_base[], ""), config=cfg,
                                     cancel=tok)))
            @testset "$seam" begin
                o = _ce_cancel_held(call)
                _CE_LATENCY[seam] = o.after
                @test _ce_cancelled(o.r, T)
                @test o.after <= 2.0
                @test o.requests == 1   # sent once, never retried
            end
        end
    finally
        rm(file; force=true)
    end
    @testset "Realtime WebSocket" begin
        session = RequestConfig(connect_timeout=10.0, stream_idle_timeout=Inf, total_deadline=Inf)
        o = _ce_cancel_in_session(tok -> speak("Hi."; service=_CERealtime, config=session, cancel=tok))
        _CE_LATENCY["Realtime WebSocket"] = o.after
        @test _ce_cancelled(o.r, AudioCallError)
        @test o.after <= 2.0
        @test o.connections == 1
    end
end

@testset "realtime_connect keeps its exception: no `cancel` keyword, and its session ignores a cancelled token" begin
    @test_throws MethodError realtime_connect(_ -> nothing; service=_CERealtime, cancel=CancelToken())
    srv = HTTP.WebSockets.listen!("127.0.0.1", 0) do ws
        try
            HTTP.WebSockets.send(ws, JSON.json(Dict("type" => "session.created")))
            for _ in ws end
        catch e
            e isa HTTP.WebSockets.WebSocketError || e isa Base.IOError || rethrow()
        end
    end
    _ce_ws_base[] = "ws://" * HTTP.WebSockets.server_addr(srv)
    try
        ev = with_cancel(() -> realtime_connect(realtime_receive; service=_CERealtime), cancel!(CancelToken()))
        @test ev["type"] == "session.created"
    finally
        close(srv)
    end
end
