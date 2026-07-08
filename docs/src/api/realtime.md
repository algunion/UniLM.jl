# [Realtime API](@id realtime_api)

Low-latency speech and text over a WebSocket. This client covers the WebSocket
event transport and ephemeral client-secret minting; audio is exchanged as
base64 PCM inside events. WebRTC media capture and SIP telephony are out of
scope. OpenAI only.

## Session and Result Types

```@docs
RealtimeSession
RealtimeSecretSuccess
RealtimeFailure
RealtimeCallError
```

## Transport Functions

```@docs
realtime_connect
realtime_send
realtime_receive
```

## Event Builders

```@docs
realtime_event
session_update
input_audio_append
response_create
```

## Client Secret

```@docs
mint_realtime_secret
```

## Usage

```julia
# Mint an ephemeral client secret for a client-side (browser) connection
secret = mint_realtime_secret()
secret isa RealtimeSecretSuccess && println("client secret: ", secret.value)

# Open a WebSocket session, configure it, stream audio, and read events
realtime_connect(model="gpt-realtime-2") do session
    realtime_send(session, session_update(Dict("modalities" => ["text", "audio"])))
    realtime_send(session, input_audio_append(audio_b64))   # audio_b64 :: base64 PCM
    realtime_send(session, response_create())
    event = realtime_receive(session)                        # blocks for the next server event
    println(event["type"])
end
```
