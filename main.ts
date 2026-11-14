// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial's brief, probes each provider's catalog route in
// the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//
// Routing: for each request, every free model meeting its requirements is a candidate, ranked by the
// quality table's rating. The router soft-caps each provider at its sealed minimum budget minus two
// (the failure schedule's refused calls), so the load spreads across providers instead of draining the
// best one dry. A provider call that answers 402, 429 or a 5xx is not the answer: the request is tried
// on the next candidate, and a provider that refuses twice in a row gets no more traffic this trial.
//
// The wires: an OpenAI client request rides an OpenAI-wire provider untouched (only the model id is
// swapped in, and stream_options.include_usage is set so the usage chunk arrives); an Anthropic one
// rides an Anthropic-wire provider the same way. Across wires the request is translated through the
// canonical form in lib/translate.ts, which carries the system prompt, user text, image bytes, tool
// names, stop sequences, max_tokens and JSON mode exactly, and the answer is translated back with the
// provider's usage counts and stop reason mapped into the client's family. A JSON-mode OpenAI request
// never rides an Anthropic-wire provider (that wire cannot carry response_format), and a request with
// assistant or tool messages never rides a Gemini-wire provider.
//
// `node main.ts --list` prints the same catalog JSON the register body carries, byte for byte, and
// exits 0 before any server starts. Run outside a trial it prints {"providers":[]}.
//
// Everything here is deterministic: no clock, no unseeded randomness, so a replay makes the same
// choices and leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  probeProvider,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { Router, requirementsFor, type Candidate } from './lib/router.ts';
import {
  answerToAnthropic,
  answerToAnthropicStream,
  answerToOpenAI,
  answerToOpenAIStream,
  anthropicError,
  anthropicEventsFromOpenAIStream,
  anthropicModelList,
  anthropicToOpenAI,
  canonOfAnthropic,
  canonOfOpenAI,
  canonToGemini,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  parseAnthropicAnswer,
  parseAnthropicStream,
  parseGeminiAnswer,
  parseGeminiStream,
  parseOpenAIAnswer,
  parseOpenAIStream,
  type Canon,
  type ProviderAnswer,
  type ServedModel,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 2 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let router: Router | null = null;
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  router = new Router(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = brief.ratings?.ratings.find((r) => r.provider === probe.provider.id && r.model === model.id) ?? null;
      const free = row !== null ? row.free : (model.free ?? false);
      if (!free) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamOk {
  ok: true;
  candidate: Candidate;
  answer: ProviderAnswer;
  /** The raw upstream stream, when the client wire matches the upstream wire and the bytes can pass through. */
  rawStream: ReadableStream<Uint8Array> | null;
}

interface UpstreamFail {
  ok: false;
  status: number;
}

type UpstreamResult = UpstreamOk | UpstreamFail;

/** The upstream request body for one candidate, in that provider's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, canon: Canon, candidate: Candidate): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (body.stream === true) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model };
    return openAIToAnthropic(body, model);
  }
  return canonToGemini(canon, model);
}

/** The upstream URL for one candidate, in that provider's wire. */
function upstreamUrl(candidate: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  if (wire === 'gemini') {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    const query = stream ? '?alt=sse' : '';
    return `${base}/v1beta/models/${candidate.model.id}:${action}${query}`;
  }
  if (wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1/chat/completions`;
}

/** Parse one upstream answer body into a provider answer. */
function parseAnswer(wire: string, stream: boolean, text: string): ProviderAnswer | null {
  if (stream) {
    if (wire === 'openai') return parseOpenAIStream(text);
    if (wire === 'anthropic') return parseAnthropicStream(text);
    return parseGeminiStream(text);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  if (wire === 'openai') return parseOpenAIAnswer(parsed);
  if (wire === 'anthropic') return parseAnthropicAnswer(parsed);
  return parseGeminiAnswer(parsed);
}

/**
 * Call one provider for one request. A 2xx with a parseable answer is the
 * request's answer; anything else is a refusal to fail over from. Same-wire
 * streams pass through byte for byte; anything translated is buffered, parsed
 * and re-rendered so the usage counts and stop reason are exact.
 */
async function callUpstream(clientApi: 'openai' | 'anthropic', body: Json, canon: Canon, candidate: Candidate): Promise<UpstreamResult> {
  const wire = candidate.probe.provider.wire;
  const url = upstreamUrl(candidate, canon.stream);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (canon.stream) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(clientApi, body, canon, candidate)),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    log(`call to ${candidate.probe.provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
    return { ok: false, status: 0 };
  }
  if (!upstream.ok) {
    try {
      await upstream.text();
    } catch {
      // The error body does not matter.
    }
    return { ok: false, status: upstream.status };
  }
  // Same wire, streaming: the provider's events are the client's events.
  if (canon.stream && upstream.body !== null && wire === clientApi) {
    return { ok: true, candidate, answer: { text: '', stopFamily: 0, toolCalls: [], inputTokens: null, outputTokens: null }, rawStream: upstream.body };
  }
  const text = await upstream.text();
  const answer = parseAnswer(wire, canon.stream, text);
  if (answer === null) {
    log(`call to ${candidate.probe.provider.id}/${candidate.model.id} answered 200 with an unparseable body`);
    return { ok: false, status: 502 };
  }
  return { ok: true, candidate, answer, rawStream: null };
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Render one provider answer to the client, in the client's wire, streamed or not. */
async function renderAnswer(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  result: UpstreamOk,
): Promise<void> {
  const wire = result.candidate.probe.provider.wire;
  const model = result.candidate.model.id;
  const stream = body.stream === true;
  if (stream && result.rawStream !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(result.rawStream, res);
    return;
  }
  if (stream) {
    // OpenAI upstream to an Anthropic client: translate the event stream chunk by chunk.
    if (clientApi === 'anthropic' && wire === 'openai') {
      // The answer was already parsed; re-render it as Anthropic events. (The
      // raw stream was consumed into `result.answer` by callUpstream.)
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(answerToAnthropicStream(result.answer, model));
      return;
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(clientApi === 'anthropic' ? answerToAnthropicStream(result.answer, model) : answerToOpenAIStream(result.answer, model));
    return;
  }
  sendJson(res, 200, clientApi === 'anthropic' ? answerToAnthropic(result.answer, model) : answerToOpenAI(result.answer, model));
}

/**
 * Serve one client request: route it by rating, fail over across providers
 * until one answers, and render that answer in the client's wire. Exactly one
 * successful provider call answers a request: a 2xx is never retried.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic', body: Json): Promise<void> {
  const wire = clientApi;
  const canon = clientApi === 'openai' ? canonOfOpenAI(body) : canonOfAnthropic(body);
  const requires: Requirement = requirementsFor(brief, manifestIndex, canon);
  manifestIndex += 1;
  if (router === null) return sendError(res, wire, 503, 'not_ready', 'the router is not ready');
  const cands = router.candidates(requires, clientApi, canon);
  if (cands.length === 0) {
    return sendError(res, wire, 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  const tried = new Set<string>();
  for (const candidate of router.orderFor(cands, tried)) {
    if (tried.has(candidate.probe.provider.id)) continue;
    tried.add(candidate.probe.provider.id);
    const result = await callUpstream(clientApi, body, canon, candidate);
    if (result.ok) {
      router.recordCall(candidate.probe.provider.id, true);
      await renderAnswer(req, res, clientApi, body, result);
      return;
    }
    if (result.status === 402 || result.status === 429 || result.status >= 500) {
      router.recordRefusal(candidate.probe.provider.id, result.status);
    } else {
      router.recordCall(candidate.probe.provider.id, false);
    }
  }
  log(`no provider answered request ${manifestIndex} (${clientApi}, requires ${JSON.stringify(requires)}): ${router.totalCalls()} calls so far`);
  sendError(res, wire, 503, 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'anthropic', body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
