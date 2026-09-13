# LiteRT Session Server

Servidor HTTP ligero y compatible con OpenAI, optimizado para minimizar TTFT mediante reutilizacion de conversaciones persistentes de LiteRT-LM SDK (KV cache).

## Caracteristicas

- API compatible con OpenAI:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- Streaming SSE compatible con OpenAI (`stream=true`)
- Respuesta normal JSON (`stream=false`)
- Engine global singleton (inicializa una sola vez)
- ConversationManager en memoria con:
  - `conversation_id -> Conversation`
  - lock por conversacion
  - timeout por inactividad
  - limite maximo de conversaciones activas
- Configuracion por variables de entorno
- Perfil global de modelo cargado al inicio (`MODEL_PROFILE`)
- Despliegue con Docker Compose

## Estructura

```
app/
  main.py
  config.py
  engine.py
  conversation_manager.py
  openai_routes.py
  schemas.py
  utils.py
Dockerfile
docker-compose.yml
requirements.txt
```

## Variables de entorno

- `MODELS_DIR` (default: `/models` en contenedor, `/volume2/docker/litertlm/litert-home/models` en host)
- `PRELOAD_FIRST_MODEL` (default: `false`, permite arranque instantáneo con 0 MB RAM usados hasta la primera petición)
- `SERVER_PORT` (default: `8000`)
- `HOST_PORT` (default: `8005`, puerto publicado en el host)
- `SESSION_TIMEOUT` en segundos (default: `600`)
- `MAX_ACTIVE_CONVERSATIONS` (default: `1`)
- `MAX_NUM_IMAGES` (default: `0`, deshabilitado para ahorrar ~1GB en visión/audio)
- `MODEL_PROFILE` (default: `profiles/default.yaml`, perfil fallback si no existe `profiles/<model_id>.yaml`)
- `CONTEXT_ROLLOVER_THRESHOLD_TOKENS` (default: `2600`)
- `CONTEXT_ROLLOVER_HEADROOM_TOKENS` (default: `1500`)
- `CONTEXT_ROLLOVER_RECENT_MESSAGES` (default: `4`)
- `CONTEXT_ROLLOVER_RECENT_TOKEN_BUDGET` (default: `1024`)

## Descubrimiento Dinámico y Hot-Swapping

- **Descubrimiento en tiempo real (`GET /v1/models`)**:
  - Escanea dinámicamente `/models` detectando subcarpetas con `model.litertlm` (ej. `/models/gemma-4-E2B-it.litertlm/model.litertlm`) y archivos `.litertlm` directos.
  - Filtra automáticamente archivos de caché de XNNPACK (`*.xnnpack_cache`) y carpetas ocultas.
- **Lazy-Loading y Hot-Swapping (`POST /v1/chat/completions`)**:
  - Si el modelo solicitado difiere del que está en memoria, se adquiere un `asyncio.Lock` de exclusión mutua para evitar condiciones de carrera y OOM bajo el límite estricto de 3 GB de RAM.
  - Se cierran todas las sesiones activas de C++, se destruye el motor previo y se ejecuta recolección forzada de memoria (`gc.collect()` + `malloc_trim`).
  - Se carga el nuevo modelo y se aplica su perfil YAML correspondiente (`profiles/<model_id>.yaml` o fallback `profiles/default.yaml`).
  - Si el modelo solicitado no existe en `/models`, retorna HTTP 404 (Model not found).

## Rolling Context Automatico

- El backend monitorea continuamente `Conversation.token_count` del SDK y proyecta los tokens del siguiente turno.
- Mientras la conversacion se mantenga por debajo de `CONTEXT_ROLLOVER_THRESHOLD_TOKENS`, se reutiliza la misma `Conversation` y su KV cache.
- Si se supera el umbral, `ConversationManager` ejecuta un rollover transparente:
  - genera un resumen breve,
  - crea una nueva `Conversation` con `Engine.create_conversation(...)`,
  - conserva el prompt de sistema del perfil,
  - inyecta un resumen muy corto,
  - y rehidrata solo los ultimos N mensajes recientes sin superar `CONTEXT_ROLLOVER_RECENT_TOKEN_BUDGET`.
- Se conserva internamente el mismo `conversation_id`, por lo que OpenWebUI no percibe el cambio.
- El rollover emite logs con tokens antes/despues para facilitar observabilidad.

## Perfil global del modelo

- El backend carga una sola vez el perfil YAML al iniciar el servidor.
- El perfil soporta `system_prompt`, `memory` y `generation` (por ejemplo `temperature`, `top_p`, etc.).
- El `system_prompt`/`memory` del perfil se inyecta solo en el bootstrap de la conversacion para preservar KV cache en turnos siguientes.
- Si el cliente envia mensajes `system`/`developer`, se combinan con el perfil solo durante ese bootstrap inicial.
- El bootstrap usa API nativa del SDK (`Engine.create_conversation(messages=..., system_message=...)`), sin `send_message()` artificial.

Ejemplo de archivo: `profiles/default.yaml`.

Detalles tecnicos verificados de la API instalada: `docs/litert-lm-sdk-findings.md`.

## Estrategia de conversation_id

Por defecto:

`SHA256(API Key + Modelo + System Prompt + Primer mensaje del chat)`

La estrategia vive en `app/utils.py` y se puede reemplazar facilmente implementando otra clase que cumpla `ConversationIdStrategy`.

## Ejecutar

1. Opcional: copiar `.env.example` a `.env` y ajustar variables.
2. Ejecutar:

```bash
docker compose up -d
```

El servicio quedara disponible en:

- `http://localhost:${HOST_PORT}/v1/models`
- `http://localhost:${HOST_PORT}/v1/chat/completions`
- `http://localhost:${HOST_PORT}/healthz`
- `http://localhost:${HOST_PORT}/internal/profile` (solo desarrollo)

## Nota de rendimiento

- Si el `conversation_id` ya existe, se reutiliza la misma `Conversation` del SDK para conservar KV cache.
- El servidor evita recrear `Engine` y evita reconstruir contexto completo mientras la conversacion siga activa.
