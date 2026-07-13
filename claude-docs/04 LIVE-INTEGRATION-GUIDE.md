# Live Integration Guide — chat-translate in a FastAPI + React app

A developer-facing walkthrough of how the `chat-translate` SDK was wired into a real
group-chat feature: a FastAPI/SQLModel/Postgres backend with a faststream+Redis job
queue, and a React + Chakra-UI v3 frontend. Where the `01–03` docs explain *why* the
SDK is shaped the way it is, this doc is the *how* — the concrete models, services,
endpoints, jobs, and UI we added to ship "everyone writes in their own language and
reads in their own language" in production.

The reference deployment uses the **local GGUF backend** (`LocalInstructionTunedProvider`
over a TranslateGemma `.gguf` via `llama-cpp-python`). The same wiring works for the
DeepL or Ollama providers — only the provider construction in one function changes.

---

## 1. The shape of the integration

The SDK gives you three primitives; the app supplies persistence, a job queue, and a UI:

| SDK primitive | What the app does with it |
|---|---|
| `resolve_source` (ELD detector) | Run **once at ingest**; store the resolved source language on the message row. |
| `create_room_translator` → `RoomTranslator` | Build **one process-wide singleton** (the ~4 GB model loads lazily on first use). |
| `mask_non_translatable` / restore | Handled inside the room translator; we only persist the **restored** output. |

Two stores, tied by the resolved source language:
- the SDK's **in-process cache** holds pre-restore (masked) provider output;
- a new **`messagetranslation` table** holds the final restored text per `(message, target_lang)`.

Three flows move data between them:

```
POST /messages   ── ingest ──▶  resolve source, persist message,
                                 create `pending` rows for room reader langs,
                                 enqueue fan-out job
                                        │
                                        ▼
job: fan-out    ── worker ──▶   load model (once), translate each pending row,
                                 flip pending → complete (or failed)
                                        │
GET /messages    ── replay ──▶  serve persisted `complete` rows; anything missing/
                                 retryable-failed → mark pending + enqueue ONE
                                 chat-level backfill job (read path never runs the model)
                                        │
                                        ▼
job: backfill   ── worker ──▶   ensure a fillable pending row, reuse fan-out
```

The golden rule that fell out of live testing: **the request path never loads or invokes
the model.** Reads are pure DB; all inference happens on the Redis worker. A cold model
or a bad inference can never blank the chat or add request latency.

---

## 2. Install the SDK

The SDK is a private git dependency; the native runtime is an **optional extra** so normal
runs and CI never build llama.cpp or download the model.

```toml
# pyproject.toml
[project]
dependencies = [
  # ...
  "chat-translate @ git+ssh://git@github.com/klv-ai/chat-translate.git#subdirectory=python",
]

[project.optional-dependencies]
# The app pins llama-cpp itself; the chat-translate `[local]` extra is not required here.
local = ["llama-cpp-python>=0.3.0"]

[tool.hatch.metadata]
# hatchling rejects direct-reference (git) deps unless this is set.
allow-direct-references = true
```

```bash
uv sync                 # normal install — no native build, no model
uv sync --extra local   # translation host — builds llama-cpp-python
```

Put the model under a gitignored dir (e.g. `backend/gguf_model_files/…Q8_0.gguf`) and
point config at it. Ship a slim default image and a separate translation image that runs
`uv sync --extra local` with the GGUF baked or mounted.

---

## 3. Configuration

A nested pydantic-settings model, prefix `TRANSLATION__`, **off by default**. When
`enabled` is false neither `llama_cpp` nor the model is ever imported.

```python
# app/core/config.py
class TranslationSettings(BaseModel):
    enabled: bool = False
    model_path: str | None = None            # abs path to the .gguf
    max_retries: int = 3                      # failed-row retry cap (see §5)
    language_subset: list[str] = ["en","fr","es","de","it","pt","nl","pl"]
    fallback_lang: str = "en"
    short_text_letters: int = 10
    short_text_policy: str = "prefer_prior"
    temperature: float = 0.0
    cache_max_entries: int = 10_000
    cache_ttl_seconds: float | None = None

    @computed_field
    @property
    def is_enabled(self) -> bool:            # both the flag AND a model path
        return bool(self.enabled and self.model_path)
```

`.env` (lives one level above `backend/` in this repo):

```dotenv
TRANSLATION__ENABLED=true
TRANSLATION__MODEL_PATH="/abs/path/backend/gguf_model_files/translategemma-4b-it.Q8_0.gguf"
TRANSLATION__LANGUAGE_SUBSET=en,fr,es,de,it,pt,nl,pl
```

We deliberately **do not** expose `n_ctx`/`n_threads`/`n_gpu_layers` — this is
single-request local inference, not batched GPU serving. `n_ctx` is fixed at 4096 in code
(the llama.cpp default of 512 truncates longer messages + the chat template).

---

## 4. Data model

`Message` reuses `language` as the **resolved source** and carries a few telemetry columns;
translations live in a child table.

```python
# app/db/models.py
class Message(SQLModelPKUID, table=True):
    content: str
    language: str | None = Field(default="en", sa_type=String(10))   # resolved source
    masked_content: str | None = Field(default=None, sa_type=TEXT)   # cheap insurance
    source_provenance: str | None = Field(default=None, sa_type=String(20))  # detected|user_confirmed|ui_fallback
    source_confident: bool | None = Field(default=None)              # low-confidence = revisitable backlog
    detected_lang: str | None = Field(default=None, sa_type=String(10))  # raw detector output (telemetry)
    translations: list["MessageTranslation"] = Relationship(
        back_populates="message", cascade_delete=True
    )

class MessageTranslation(SQLModelPKUID, table=True):
    __table_args__ = (
        UniqueConstraint("message_uid", "target_lang", name="uq_message_translation_msg_lang"),
    )
    message_uid: uuid.UUID = Field(foreign_key="message.uid", ondelete="CASCADE")
    target_lang: str = Field(sa_type=String(10))
    content: str | None = Field(default=None, sa_type=TEXT)          # null while pending
    status: str = Field(default=TranslationStatus.pending, sa_type=String(12))  # pending|complete|failed
    attempts: int = Field(default=0)                                # retry counter (§5)
```

```python
# app/db/interfaces.py
class TranslationStatus(StrEnum):
    pending = "pending"     # row created, awaiting a job
    complete = "complete"   # restored text in `content`
    failed = "failed"       # retried up to settings.TRANSLATION.max_retries, then treated as missing
```

**Why one row per `(message, target)` and not a list on the message.** The response is
*personalised*: each viewer gets only their own language in `MessagePublic.translation`.
This keeps payloads lean, means the client never has to pick a language, and doesn't leak
which languages other members read. The `translations` relationship exists for the ORM
cascade and the fan-out job, **not** for the API shape.

`MessagePublic` gains two read-only fields:

```python
# app/db/dtos/messaging.py
class MessagePublic(MessageBase):
    # ...
    language: str | None = None          # resolved source of `content`
    translation: str | None = None       # viewer's-language text; null if same-lang/untranslatable/off
    translation_pending: bool = False    # a fan-out/backfill for the viewer's lang is in flight
```

The migration is a single hand-authored alembic revision (add the message columns, create
`messagetranslation`, add `attempts`). Apply with `alembic upgrade head`, then regenerate
the client (§10).

---

## 5. The translation service — the SDK integration point

Everything SDK-touching lives in `app/services/chat_translation_service.py`. Routes and
jobs call into it; they never import the SDK directly. Two categories of function:

### 5a. Singletons & primitives

```python
def get_room_translator() -> RoomTranslator | None:
    """Lazy, process-wide, double-checked singleton. None when disabled/unavailable."""
    if not settings.TRANSLATION.is_enabled:
        return None
    # ... build once under a lock, memoise the result (including None) ...

def _build_translator() -> RoomTranslator | None:
    s = settings.TRANSLATION
    try:
        from llama_cpp import Llama            # lazy: only imported when enabled
    except ImportError:
        return None                            # extra not installed → gracefully disabled
    from chat_translate import LocalInstructionTunedProvider
    llama = Llama(model_path=s.model_path, n_ctx=4096, verbose=False)
    provider = LocalInstructionTunedProvider(llama=llama, temperature=s.temperature)
    detector = eld_language_detector(EldLanguageDetector(), subset=s.language_subset, fallback=s.fallback_lang)
    cache = InMemoryTranslationCache(max_entries=s.cache_max_entries, ttl_seconds=s.cache_ttl_seconds)
    return create_room_translator(provider, detector=detector, cache=cache)
```

> **To switch providers**, this is the only function that changes — swap
> `LocalInstructionTunedProvider` for `DeepLProvider` / the Ollama provider and adjust the
> constructed client. Everything above the provider boundary is untouched.

The **ingest-side source resolver** is a separate, model-free ELD detector (memoised), so
POST can resolve a language without ever touching llama.cpp:

```python
def resolve_message_source(content: str, ui_lang: str) -> tuple[ResolvedSource, str]:
    m = mask_non_translatable(content, sentinels=BRACKET_SENTINELS)  # match the provider's scheme
    resolved = resolve_source(
        m.detection_text(), get_source_resolver(),
        ResolveSourceOptions(ui_lang=ui_lang,
                             short_text_letters=s.short_text_letters,
                             short_text_policy=cast(ShortTextPolicy, s.short_text_policy)),
    )
    return resolved, m.masked
```

> **Sentinel note (learned the hard way):** TranslateGemma's tokenizer eats the SDK's
> default Private-Use-Area sentinels, so the LLM providers use the `⟦id⟧` **bracket
> scheme**. Mask with `BRACKET_SENTINELS` at ingest so the stored masked form matches how
> the translator masks. On DeepL the PUA default is fine. This is a per-provider setting in
> the SDK; you don't have to think about it beyond passing the right constant.

Translation itself has a **sync** form (safe to call from FastAPI's threadpooled sync
routes) and an **async** form that offloads the blocking call — essential inside the async
job subscriber so inference never stalls the event loop:

```python
def translate_message(translator, content, source_lang, target) -> str | None:
    room = translator.translate_for_room(content, [target],
                                         ViewerTranslateOptions(source_lang=source_lang))
    vt = room.by_language.get(target)
    return None if (vt is None or not vt.translated) else vt.text

async def atranslate_message(translator, content, source_lang, target) -> str | None:
    return await anyio.to_thread.run_sync(translate_message, translator, content, source_lang, target)
```

### 5b. Route-facing API (thin routes, fat service)

The endpoints do nothing but verify membership and delegate. Four functions carry the work:

- **`viewer_lang(user, accept_language=None) -> str`** — the reading-language chain:
  `user.language` → `Accept-Language` → `fallback_lang`, each normalised to a base ISO code.
- **`ingest_message(...) -> Message`** — resolve source (guarded; degrade to `ui_lang` on
  any error), persist, and enqueue the pre-emptive fan-out.
- **`present_messages(...) -> list[MessagePublic]`** — build the viewer's list from
  persisted rows only, collect what's missing, publish **one** backfill job. Read-only.
- **`run_message_fanout` / `run_chat_backfill`** — the job bodies (§6), extracted so they're
  unit-testable with a fake translator.

The presentation decision is a **pure function** returning a `NamedTuple` — no I/O, no
mutation of the DTO (this replaced an earlier version that mutated a passed-in object):

```python
class TranslationView(NamedTuple):
    translation: str | None = None
    pending: bool = False
    needs_backfill: bool = False

def _translation_view(message, row, target) -> TranslationView:
    if message.message_type_id != MessageTypeOptions.message:        # system/join lines
        return TranslationView()
    source = normalize_lang(message.language) or settings.TRANSLATION.fallback_lang
    if not target or source == target:                              # same-language short-circuit
        return TranslationView()
    if row is None:
        return TranslationView(needs_backfill=True)
    if row.status == TranslationStatus.complete and row.content is not None:
        return TranslationView(translation=row.content)
    if row.status == TranslationStatus.pending:
        return TranslationView(pending=True)
    # failed: retry until the cap, then serve the original for good.
    if row.attempts < settings.TRANSLATION.max_retries:
        return TranslationView(needs_backfill=True)
    return TranslationView()
```

**Why `attempts`/`max_retries` matter:** without it, `failed` is terminal — a transient
failure (worker restart mid-load, an OOM) would permanently pin that `(message, lang)` to
its original. The counter lets the backfill re-queue failures a bounded number of times so
permanently-bad content can't loop forever. `mark_failed` increments it; `ensure_pending`
resets a retryable `failed` row back to `pending` and returns `None` once exhausted.

---

## 6. Job-queue wiring (faststream + Redis)

Two channels and two subscribers. Both delegate straight to the service; the blocking call
is offloaded to a thread inside `atranslate_message`, so the worker's event loop stays free.

```python
# app/interfaces/req_res.py
class JqChannel(StrEnum):
    # ...
    message_translation_request = "message_translation_request"   # POST fan-out (one message)
    chat_translation_backfill  = "chat_translation_backfill"      # GET backfill (many messages)

class MessageTranslationJobRequest(BaseModel):
    message_uid: UUID

class ChatTranslationBackfillRequest(BaseModel):
    target_lang: str
    message_uids: list[UUID]     # explicit uids, so a paged-back window is covered exactly
```

```python
# app/api/v1/redis_job_queue.py
@job_queue.subscriber(JqChannel.message_translation_request)
async def message_translation_request(req, logger, session=Depends(yieldNewPgSession)):
    done = await run_message_fanout(session, req.message_uid)

@job_queue.subscriber(JqChannel.chat_translation_backfill)
async def chat_translation_backfill(req, logger, session=Depends(yieldNewPgSession)):
    done = await run_chat_backfill(session, req.message_uids, req.target_lang)
```

Jobs are published from a **FastAPI `BackgroundTask`** so publishing happens after the
response is sent, and the broker is imported lazily to avoid a module cycle with the router:

```python
def _publish_job(payload, channel, background_tasks):
    async def _dispatch():
        from app.api.v1.redis_job_queue import job_queue
        await job_queue.broker.publish(payload, channel)
    background_tasks.add_task(_dispatch)
```

Fan-out targets are the room's distinct reader languages minus the source:

```python
targets = {normalize_lang(u.language) for u in chat.users if u.language} - {source, ""}
```

If Redis is down, no pending rows are created and no job is published — the GET backfill
path fills translations on demand later, so you never leave orphan `pending` rows.

---

## 7. API endpoints — verify and delegate

The whole point of the refactor: the router carries no translation logic.

```python
# app/api/v1/group_chat.py
@router.post("/{chat_uid}/messages", response_model=MessagePublic)
def send_message(session, current_user, chat_uid, message_in, request, background_tasks):
    if not GroupChatTable(session).user_is_in_chat(current_user.uid, chat_uid):
        raise HTTPException(403, "User is not in chat")
    ui_lang = viewer_lang(current_user, request.headers.get("accept-language"))
    return ingest_message(session, message_in, chat_uid, current_user.uid, ui_lang, background_tasks)

@router.get("/{chat_uid}/messages", response_model=list[MessagePublic])
def list_messages(session, current_user, chat_uid, background_tasks,
                  skip=0, limit=Query(50, le=100), lang: str | None = Query(None)):
    if not GroupChatTable(session).user_is_in_chat(current_user.uid, chat_uid):
        raise HTTPException(403, "User is not in chat")
    messages = MessageTable(session).fetch_for_chat(chat_uid, limit=limit, skip=skip)
    target = normalize_lang(lang) or viewer_lang(current_user)   # ?lang= manual override
    return present_messages(session, messages, target, background_tasks)
```

`?lang=` lets a client render any language on demand (handy for testing and a future
"view in…" switcher) without changing the user's stored reading language.

---

## 8. Reading language — `User.language` synced with the i18n selector

Chat translates to whatever `User.language` holds; the UI chrome only has `en/fr/it`
catalogs. These are kept distinct: `User.language` is a free ISO-639-1 code; the UI switches
only when it happens to match a shipped catalog.

**Backend — progressive backfill on login.** Null reading languages are seeded from the
browser on first login, so existing users get a sensible default with no manual choice:

```python
# app/api/v1/login.py — inside login_access_token
if not user.language:
    user.language = normalize_lang(request.headers.get("accept-language")) \
        or settings.TRANSLATION.fallback_lang
    session.add(user); session.commit()
```

`UserBase.language` (and `UserUpdateMe.language`) make it readable via `UserPublic` and
settable via `PATCH /users/me`.

**Frontend — the settings selector persists it and the auth hook mirrors it.**

```tsx
// UserSettings/Language.tsx — change the UI language AND persist the reading language
const onChange = (code: string) => {
  i18n.changeLanguage(code)
  mutation.mutate(code)   // UsersService.updateUserMe({ requestBody: { language: code } })
}
```

```ts
// hooks/useAuth.ts — mirror the persisted reading language into the UI when it's a catalog
useEffect(() => {
  const lang = user?.language
  if (lang && isUiLanguage(lang) && i18n.language.split("-")[0] !== lang.split("-")[0]) {
    i18n.changeLanguage(lang.split("-")[0])
  }
}, [user?.language])
```

`src/main.tsx` already sends `Accept-Language: i18n.language`, so the server-side backfill
receives the browser language at first login and the loop closes.

---

## 9. Frontend widget — translated by default, reveal the original

`MessagePublic` already carries `content` (original), `translation` (viewer's language), and
`translation_pending`. The message list polls every 5 s (TanStack Query
`refetchInterval: 5000`), so a back-filled translation appears on the next tick with no extra
plumbing. One presentational component holds all of it:

```tsx
// components/GroupChat/TranslatedMessage.tsx  (rendered inside the bubble)
const TranslatedMessage = ({ msg, isMe }: { msg: MessagePublic; isMe: boolean }) => {
  const { t } = useTranslation()
  const muted = isMe ? "whiteAlpha.700" : "fg.muted"

  // No translation → show the original (+ a subtle hint while a job is in flight).
  if (!msg.translation) {
    return (
      <>
        <Text whiteSpace="pre-wrap">{msg.content}</Text>
        {msg.translation_pending && (
          <Text fontSize="2xs" color={muted} opacity={0.8} fontStyle="italic">
            {t("viewsChats.translationPending")}
          </Text>
        )}
      </>
    )
  }

  // Translation present → show it, with a muted chevron-reveal of the original.
  return (
    <>
      <Text whiteSpace="pre-wrap">{msg.translation}</Text>
      <Collapsible.Root>
        <Collapsible.Trigger asChild>
          <HStack gap="1" cursor="pointer" mt="1" color={muted} opacity={0.8}>
            <Icon size="xs" asChild css={{
              transition: "transform 0.2s",
              "[data-state=open] &": { transform: "rotate(0deg)" },
              "[data-state=closed] &": { transform: "rotate(-90deg)" },
            }}>
              <LuChevronDown />
            </Icon>
            <Text fontSize="2xs">{t("viewsChats.showOriginal", { lang: msg.language })}</Text>
          </HStack>
        </Collapsible.Trigger>
        <Collapsible.Content>
          <Text whiteSpace="pre-wrap" fontSize="xs" color={muted} opacity={0.8} mt="1">
            {msg.content}
          </Text>
        </Collapsible.Content>
      </Collapsible.Root>
    </>
  )
}
```

`MessageBubble` just renders `<TranslatedMessage msg={msg} isMe={isMe} />` in place of the
old `{msg.content}`. Notes:
- Uses Chakra v3 `Collapsible` + Lucide `LuChevronDown` rotating on `data-state` — the
  house pattern for chevron reveals.
- `muted` flips to `whiteAlpha.700` on own-message bubbles (which are solid blue) so the
  secondary text stays legible; otherwise the `fg.muted` / `2xs` / `opacity 0.8` chat
  convention.
- New i18n keys `viewsChats.showOriginal` (`"Original ({{lang}})"`) and
  `viewsChats.translationPending` (`"Translating…"`) added to the `en/fr/it` catalogs.

---

## 10. Migration & client regeneration

```bash
# backend
alembic upgrade head           # apply the message columns + messagetranslation table + attempts

# frontend (after the OpenAPI spec changes, e.g. the ?lang= param)
npm run sync-api               # fetch openapi.json + regenerate src/client
```

The regenerated `MessagePublic` type carries `translation` / `translation_pending` /
`language`, which the widget consumes directly.

---

## 11. Testing without the model

The entire `LlamaLike` surface the SDK needs is a **single method**, so tests inject a fake
llama — no native build, no model file, fast:

```python
class FakeLlama:
    def create_chat_completion(self, *, messages, **kwargs):
        item = messages[-1]["content"][0]                 # structured instruction-tuned content
        return {"choices": [{"message": {"content": item["text"].upper()}}]}  # "translate" = upper-case
```

Wrap it with the real detector + cache via `LocalInstructionTunedProvider(llama=fake)` and
`create_room_translator(...)`. What we cover:
- **service unit tests** — `normalize_lang`, `viewer_lang`, source resolution against real
  ELD, same-language / untranslatable short-circuits (0 model calls), cache reuse, and the
  pure `_translation_view` decision table (missing → backfill, complete → serve, pending →
  signal, failed < cap → retry, failed ≥ cap → original).
- **API tests** — GET serves persisted rows, signals pending, publishes exactly one
  chat-level backfill job and writes **nothing** on the read path, honours `?lang=`; POST
  resolves + persists source. `run_message_fanout` / `run_chat_backfill` are driven directly
  with the fake.
- **isolation** — an autouse fixture forces `settings.TRANSLATION.enabled = False` so the
  suite never depends on a developer's `.env` (which may point at a real model + broker); the
  translation tests opt back in and mock `broker.publish`.

---

## 12. Operational notes & gotchas

- **Offload blocking inference in async workers.** The job subscribers are `async`; the SDK
  call must go through `anyio.to_thread` (`atranslate_message`) or it stalls that worker's
  event loop. Sync GET routes may call the sync form directly (FastAPI threadpools them).
- **Per-worker model memory.** `fastapi run --workers 4` loads a ~4 GB singleton *per
  worker*. Use fewer workers, or a dedicated translation worker, for real deploys.
- **Redis off is a supported state.** POST skips pending-row creation; GET lazy-fills via the
  backfill path. Never create `pending` rows you won't publish a job for.
- **Pending-row races are benign.** The unique `(message_uid, target_lang)` constraint plus
  `create_pending`/`upsert_complete` catching `IntegrityError` keep concurrent jobs from
  double-inserting; worst case is a wasted (cache-deduped) translation.
- **Legacy messages** with null `language` are treated as `fallback_lang` — the provider
  needs a non-null source.
- **Message edits** (not yet implemented) must invalidate that message's translation rows.
- **Don't gate chat translation on the UI-catalog list.** `User.language` can be any code in
  `language_subset`; the `en/fr/it` catalogs only govern the UI chrome.

---

## File map (reference deployment)

| Area | File |
|---|---|
| SDK wiring / service | `app/services/chat_translation_service.py` |
| Config | `app/core/config.py` (`TranslationSettings`) |
| Models | `app/db/models.py` (`Message`, `MessageTranslation`), `app/db/interfaces.py` (`TranslationStatus`) |
| Persistence | `app/db/tables/message_translation_table.py` |
| Job queue | `app/api/v1/redis_job_queue.py`, `app/interfaces/req_res.py` (`JqChannel`, payloads) |
| Endpoints | `app/api/v1/group_chat.py`, `app/api/v1/login.py` (language backfill) |
| Frontend widget | `frontend/src/components/GroupChat/TranslatedMessage.tsx` (+ `MessageBubble.tsx`) |
| Frontend language sync | `frontend/src/components/UserSettings/Language.tsx`, `frontend/src/hooks/useAuth.ts`, `frontend/src/constants/languages.ts` |
