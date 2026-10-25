# [Attachments](@id attachments_guide)

```@setup attachments
using UniLM
```

A user message can carry media besides its text — images, sound clips and PDF documents — for a model
that reads them. Attach a file to ask about a picture, a recording or a report; how each provider sends
the media, and which kinds it takes, is on its entry.

```@example attachments
service, model = OPENAIServiceEndpoint, "gpt-5.4-mini"
nothing # hide
```

The examples use only the names bound above. `service` and `model` are the provider and model they run
on: bind your own there. The table [at the end](@ref attachments_providers) links each provider's rules.

## [Images, audio and documents](@id chat_attachments)

An [`ImageAttachment`](@ref) (PNG, JPEG, GIF or WebP), an [`AudioAttachment`](@ref) (WAV or MP3) and a
[`DocumentAttachment`](@ref) (PDF) each hold the media's bytes, read from a file or given as a byte
vector, and recognise the format by the bytes' leading signature, never by a file name. Empty bytes,
bytes in another format, and a URL or a data URL in place of a file throw `ArgumentError`. An image
may set the `detail` level a model reads it at — `"auto"`, `"low"`, `"high"` or `"original"` — and a
document keeps a `filename`: the file's own name, or the `filename` keyword's for bytes.

```@example attachments
image = ImageAttachment("../assets/red_circle.png"; detail="low")
document = DocumentAttachment("../assets/bridge_report.pdf")
clip = AudioAttachment("../assets/secret_word.wav")
image, document, clip
```

The format comes from the bytes alone, so a picture read as a sound clip is refused:

```@example attachments
refusal = try AudioAttachment("../assets/red_circle.png"); nothing catch err; err end
refusal isa ArgumentError || error("expected the image to be refused as audio")
print(refusal.msg)
```

Which kinds and `detail` levels a provider takes is on its entry. A provider whose entry lists an
attachment among its refusals throws `ArgumentError` for it before the request; one that sends every
kind leaves it to the server to answer for a model that does not read it.

## [Sending attachments](@id attachments_send)

A user message takes its attachments after its text, `Message(Val(:user), text, attachments...)`;
attachments go on user messages only:

```@example attachments
message = Message(Val(:user), "What colour is the shape? One word.", image)
message.attachments
```

```julia
chat = Chat(service=service, model=model)
push!(chat, Message(Val(:system), "You are concise."))
push!(chat, message)
result = chatrequest!(chat)
```

Each provider writes the attachments into its own request. An endpoint on the OpenAI-compatible wire
sends an image as an `image_url` part, with its `detail` when it sets one, a sound clip as an
`input_audio` part and a document as a `file` part, all base64, and the server decides what to do with
each; the other encoders are on their entries.

A [`Respond`](@ref) takes media as parts of an [`InputMessage`](@ref)'s content —
[`input_image`](@ref) for an image, and [`input_file`](@ref)`(document)` for a `DocumentAttachment` —
and, on some providers, the attachments themselves; which parts a provider takes is on its entry.

## [On your provider](@id attachments_providers)

```@eval
Main.ProviderFacts.on_your_provider(Main.ProviderFacts.llm, :media)
```
