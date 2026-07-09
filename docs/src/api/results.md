# Result Types

Abstract and concrete types for handling API call outcomes. All result types
are subtypes of [`LLMRequestResponse`](@ref).

## Abstract Type

```@docs
LLMRequestResponse
```

## Chat Completions Results

```@docs
LLMSuccess
LLMFailure
LLMCallError
```

### Pattern Matching

```julia
result = chatrequest!(chat)

if result isa LLMSuccess
    println(result.message.content)
    println(result.message.finish_reason)  # "stop"
elseif result isa LLMFailure
    @warn "HTTP $(result.status): $(result.response)"
elseif result isa LLMCallError
    @error "Call error: $(result.error)"
end
```

## Responses API Results

See also the [Responses API reference](responses.md).

- [`ResponseSuccess`](@ref)
- [`ResponseFailure`](@ref)
- [`ResponseCallError`](@ref)

### Pattern Matching

```julia
result = respond("Tell me a joke")

if result isa ResponseSuccess
    println(output_text(result))
    println(result.response.status)  # "completed"
elseif result isa ResponseFailure
    @warn "HTTP $(result.status)"
elseif result isa ResponseCallError
    @error result.error
end
```

## Image Generation Results

See also the [Images API reference](images.md).

- [`ImageSuccess`](@ref)
- [`ImageFailure`](@ref)
- [`ImageCallError`](@ref)

### Pattern Matching

```julia
result = generate_image("A robot writing Julia code")

if result isa ImageSuccess
    save_image(image_data(result)[1], "robot.png")
elseif result isa ImageFailure
    @warn "HTTP $(result.status): $(result.response)"
elseif result isa ImageCallError
    @error result.error
end
```

## Type Hierarchy

All result types share the abstract parent [`LLMRequestResponse`](@ref):

```
LLMRequestResponse   (abstract parent of every result type below)
│
├─ Chat Completions   LLMSuccess · LLMFailure · LLMCallError
├─ Responses API      ResponseSuccess · ResponseFailure · ResponseCallError
├─ Embeddings         EmbeddingSuccess · EmbeddingFailure · EmbeddingCallError
├─ Image Generation   ImageSuccess · ImageFailure · ImageCallError
├─ FIM Completion     FIMSuccess · FIMFailure · FIMCallError
├─ Files              FileSuccess · FileListSuccess · FileContentSuccess · FileDeleteSuccess · FileFailure · FileCallError
├─ Vector Stores      VectorStoreSuccess · VectorStoreListSuccess · VectorStoreFileSuccess · VectorStoreBatchSuccess · VectorStoreDeleteSuccess · VectorStoreFailure · VectorStoreCallError
├─ Conversations      ConversationSuccess · ConversationItemSuccess · ConversationItemListSuccess · ConversationDeleteSuccess · ConversationFailure · ConversationCallError
├─ Moderations        ModerationSuccess · ModerationFailure · ModerationCallError
├─ Audio              SpeechSuccess · TranscriptionSuccess · AudioFailure · AudioCallError
├─ Batch              BatchSuccess · BatchListSuccess · BatchFailure · BatchCallError
├─ Fine-tuning        FineTuningSuccess · FineTuningListSuccess · FineTuningFailure · FineTuningCallError
├─ Containers         ContainerSuccess · ContainerListSuccess · ContainerDeleteSuccess · ContainerFailure · ContainerCallError
├─ Uploads            UploadSuccess · UploadPartSuccess · UploadFailure · UploadCallError
├─ Videos             VideoSuccess · VideoListSuccess · VideoContentSuccess · VideoFailure · VideoCallError
└─ Realtime           RealtimeSecretSuccess · RealtimeFailure · RealtimeCallError
```

Every `*Success` wraps a parsed response object; every `*Failure` carries the HTTP status and
body; every `*CallError` wraps a transport/exception. Pattern-match on the family you called
(see each API-reference page for the concrete fields).
