# LiteRT-LM SDK Findings (v0.13.1)

Fecha de verificacion: 2026-07-02
Entorno inspeccionado: paquete instalado en `.venv`

## Evidencia de API oficial inspeccionada

- `litert_lm.Engine.create_conversation(...)` soporta oficialmente:
  - `messages`
  - `system_message`
  - `sampler_config`
  - `filter_channel_content_from_kv_cache`
  - `extra_context`
- Fuente inspeccionada:
  - `.venv/lib/python3.12/site-packages/litert_lm/engine.py`
- Firma observada:
  - `create_conversation(..., messages=..., ..., sampler_config=..., system_message=..., ...)`

## Conclusiones para inicializacion de contexto

1. Existe mecanismo oficial para bootstrap sin `send_message()`:
   - Usar `Engine.create_conversation(messages=..., system_message=...)`.
2. No es necesario inyectar `system` dentro del primer `user` para bootstrap.
3. La estrategia recomendada para este backend es:
   - `messages`: historial previo user/assistant (sin el ultimo turno incremental).
   - `system_message`: combinacion de perfil global + `system/developer` del request solo al crear la conversacion.

## Analisis de perfil global precargado una sola vez y reutilizable

Se evaluo si el SDK permite prellenar una sola vez el contexto global y clonarlo para nuevas conversaciones:

- La API publica incluye `Session` y `Conversation`.
- `Session` expone `run_prefill`, `run_decode` y `run_decode_async`, pero no existe API publica de `clone()` o snapshot/restauracion de KV cache.
- En `interfaces.py` aparece explicitamente un TODO: `Add clone() API once switching to advanced engine.`

Implicacion:

- Con API oficial actual (0.13.1), no hay mecanismo soportado para precomputar un KV base una sola vez y reutilizarlo para multiples conversaciones independientes.
- Lo maximo oficialmente soportado es inicializar cada conversacion con `system_message/messages` nativos al crearla.

## Decisiones aplicadas en el backend

- Se reemplazo el bootstrap manual por inicializacion nativa del SDK:
  - `ConversationManager` crea conversaciones con `create_conversation(messages=..., system_message=...)`.
- Se mantiene el perfil global cargado una sola vez al arranque.
- En turnos posteriores no se reinyecta sistema/memoria; se usa el estado de la `Conversation` (KV cache de la sesion activa).
