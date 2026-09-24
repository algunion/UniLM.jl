# [Realtime API](@id realtime_api)

Low-latency speech and text over a WebSocket. This client covers the WebSocket
event transport and ephemeral client-secret minting; audio is exchanged as
base64 PCM inside events. WebRTC media capture and SIP telephony are out of
scope. OpenAI only.

## Bounds

[`realtime_connect`](@ref) takes a `config::Union{Nothing,RequestConfig}`,
resolves it the usual four ways, and captures it on the session's `config` field:

- **The open phase** is bounded by `connect_timeout` — a peer that accepts the
  TCP connection but never completes the upgrade throws
  `UniLMTimeout(:connect, …)` rather than blocking. An open that fails for another
  reason throws that error, and an upgrade that completes only after the timeout is
  closed without running the handler. Only `OPENAIServiceEndpoint` has a Realtime
  WebSocket: any other `service` throws `ArgumentError` before any I/O, so another
  endpoint's credentials are never sent to `api.openai.com`.
- **[`realtime_receive`](@ref)** is bounded by the session's
  `stream_idle_timeout` and throws `UniLMTimeout(:stream_idle, …)` on a breach.
  Unblocking a parked read means closing the socket, and HTTP.jl's WebSocket
  close allows the peer up to ~5 s to acknowledge, so the breach surfaces within
  `[limit, limit + ~5 s]`.
- **The session's lifetime is deliberately unbounded.** Once your handler runs,
  how long it stays connected is your decision — a Realtime session is meant to
  sit idle waiting for input.

Realtime throws rather than returning a typed result value, and `realtime_connect`
makes a single attempt; `max_attempts` does not apply. Neither the WebSocket open nor
`realtime_receive` observes a [`CancelToken`](@ref). [`mint_realtime_secret`](@ref) is
an ordinary HTTP verb that returns typed results: a `200` without a non-empty secret is
a `RealtimeCallError`, and a `RealtimeCallError` carries its exception in `cause` and
the provider's `request_id` when one was sent. A minted client secret is a live
credential, so [`RealtimeSecretSuccess`](@ref) redacts it when displayed — read
`.value` programmatically. See [Timeouts & Retries](@ref timeout_realtime).

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
