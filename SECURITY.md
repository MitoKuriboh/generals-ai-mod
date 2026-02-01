# Security Policy

## Reporting a Vulnerability

If you have been informed of or discovered a security vulnerability, exploit or other flaw, please contact a developer of TheSuperHackers to report it privately. Public channels on Discord or the GitHub should be avoided, as the exploit may be taken from a public post and used before it can be patched. The best method is to contact xezon on discord. He can be reached in the [Community Outpost Discord](https://discord.gg/WzxQDZersE).

---

## ML Bridge Security Considerations

### Buffer Overflow Prevention

The ML bridge uses `snprintf` with explicit buffer size limits throughout:

```cpp
// All string formatting uses bounded writes
snprintf(buffer, sizeof(buffer), "format", args);
```

Fixed-size buffers are used for message serialization:
- Message buffer: 4096 bytes (sufficient for ~44 state features as JSON)
- Length prefix: 4 bytes (uint32_t, little-endian)

### Named Pipe Security Model

The communication pipe `\\.\pipe\generals_ml_bridge` uses:
- **Local-only access**: Named pipes are local IPC; no network exposure
- **Synchronous I/O**: Blocking reads/writes prevent race conditions
- **Length-prefixed messages**: Prevents buffer over-reads by specifying exact payload size
- **JSON parsing**: Using standard JSON libraries with size limits

### IPC Isolation

- The Python ML server and game run as separate processes
- No shared memory regions
- Communication is message-based (request/response pattern)
- Malformed JSON is rejected without processing
- State extraction reads only from game's existing data structures

### Potential Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Pipe hijacking | Local access only; start server before game |
| Message injection | Length-prefix validation; JSON schema checking |
| DoS via large messages | Fixed buffer size; reject oversized payloads |
| Information disclosure | Pipe name is well-known but data is game state only |