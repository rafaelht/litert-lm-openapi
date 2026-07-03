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

- `MODEL_PATH` (default: `/models/gemma-4-E2B-it.litertlm/model.litertlm`)
- `SERVER_PORT` (default: `8000`)
- `HOST_PORT` (default: `8001`, puerto publicado en el host)
- `SESSION_TIMEOUT` en segundos (default: `1800`)
- `MAX_ACTIVE_CONVERSATIONS` (default: `1000`)
- `MAX_NUM_IMAGES` (default: `4`, habilita entradas multimodales de imagen)
- `MODEL_PROFILE` (default: `profiles/default.yaml`)

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
