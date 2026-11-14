// The two client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON body
// or with server-sent events. The entry answers each request through one of the
// ten providers, on that provider's own wire (OpenAI, Anthropic or Gemini), and
// translates the answer back into the wire the client used.
//
// The one rule behind every function here: the provider echoes what it parsed,
// and the client compares that echo with what it sent. So a translation must
// carry the system prompt, the user text, the image bytes, the tool names, the
// stop sequences, max_tokens and JSON mode across the wire exactly, and must
// carry the provider's text whole, its usage counts as that wire reports them,
// and its stop reason mapped into the client's family.
//
//   stop families: end_turn / stop / STOP = 0, max_tokens / length / MAX_TOKENS
//   = 1, tool_use / tool_calls = 2, refusal / content_filter / SAFETY = 3.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same bytes
// on a replay. The model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An image carried across wires: base64 bytes, or a plain URL. */
export type ImageRef = { kind: 'base64'; mime: string; data: string } | { kind: 'url'; url: string };

/** One tool definition, in canonical form. */
export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

/** One client request, reduced to what every wire can carry. */
export interface Canon {
  /** The system prompt, '' when the client sent none. */
  systemText: string;
  /** The last user message's text. */
  userText: string;
  /** The last user message's image, when it carried one. */
  image: ImageRef | null;
  tools: CanonTool[];
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  /** The client asked for JSON mode (OpenAI response_format only; the Anthropic wire cannot carry it). */
  jsonMode: boolean;
  stream: boolean;
  /** The body holds assistant or tool messages a Gemini content list cannot carry faithfully. */
  complexMessages: boolean;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The last user message's image, in canonical form. */
function imageOfOpenAI(content: unknown): ImageRef | null {
  if (!Array.isArray(content)) return null;
  let found: ImageRef | null = null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
    const url = block.image_url.url;
    if (typeof url !== 'string' || url === '') continue;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    found = data !== null ? { kind: 'base64', mime: data[1] === '' ? 'image/png' : data[1], data: data[2] } : { kind: 'url', url };
  }
  return found;
}

function imageOfAnthropic(content: unknown): ImageRef | null {
  if (!Array.isArray(content)) return null;
  let found: ImageRef | null = null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      found = { kind: 'base64', mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data };
    } else if (source.type === 'url' && typeof source.url === 'string') {
      found = { kind: 'url', url: source.url };
    }
  }
  return found;
}

/** An OpenAI Chat Completions body as a canonical request. */
export function canonOfOpenAI(body: Json): Canon {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  let userText = '';
  let image: ImageRef | null = null;
  let complexMessages = false;
  for (const message of messages) {
    if (message.role === 'system') {
      const text = textOf(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = textOf(message.content);
      image = imageOfOpenAI(message.content) ?? image;
    } else {
      complexMessages = true;
    }
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stop = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((x): x is string => typeof x === 'string') : [];
  return {
    systemText: systemParts.join('\n'),
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    stops: stop,
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
    complexMessages,
  };
}

/** An Anthropic Messages body as a canonical request. */
export function canonOfAnthropic(body: Json): Canon {
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let image: ImageRef | null = null;
  let complexMessages = false;
  for (const message of messages) {
    if (message.role === 'user') {
      userText = textOf(message.content);
      image = imageOfAnthropic(message.content) ?? image;
    } else {
      complexMessages = true;
    }
  }
  const tools: CanonTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  return {
    systemText,
    userText,
    image,
    tools,
    maxTokens: numberOrNull(body.max_tokens),
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((x): x is string => typeof x === 'string') : [],
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    jsonMode: false,
    stream: body.stream === true,
    complexMessages,
  };
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`.
 * The system prompt becomes a system message, image blocks become image_url data URLs,
 * tool_use blocks become tool_calls, tool_result blocks become tool messages.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`.
 * The system messages become the system prompt, image_url parts become image blocks,
 * tool_calls become tool_use blocks, tool messages become tool_result blocks.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const outMessages: Json[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = textOf(message.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (message.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(message.content);
      if (text !== '') content.push({ type: 'text', text });
      if (Array.isArray(message.tool_calls)) {
        for (const call of message.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let input: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) input = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
        }
      }
      outMessages.push({ role: 'assistant', content });
      continue;
    }
    if (message.role === 'tool') {
      outMessages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof message.tool_call_id === 'string' ? message.tool_call_id : '', content: typeof message.content === 'string' ? message.content : toolResultText(message.content) }] });
      continue;
    }
    if (message.role !== 'user') continue;
    const content = message.content;
    if (typeof content === 'string') {
      outMessages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) outMessages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages: outMessages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((x): x is string => typeof x === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** A canonical request as a Gemini generateContent body. */
export function canonToGemini(canon: Canon, model: string): Json {
  const parts: Json[] = [{ text: canon.userText }];
  if (canon.image !== null) {
    if (canon.image.kind === 'base64') parts.push({ inlineData: { mimeType: canon.image.mime, data: canon.image.data } });
    else parts.push({ fileData: { mimeType: 'image/png', fileUri: canon.image.url } });
  }
  const body: Json = { contents: [{ role: 'user', parts }] };
  if (canon.systemText !== '') body.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    body.tools = [{ functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.topP !== null) config.topP = canon.topP;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  void model;
  return body;
}

/** A provider's answer, reduced to text, stop family and usage counts. */
export interface ProviderAnswer {
  text: string;
  /** Stop family: 0 end_turn/stop, 1 max_tokens/length, 2 tool_use, 3 refusal. */
  stopFamily: number;
  /** Tool calls the provider returned, in OpenAI shape (empty in this trial's sim, kept for fidelity). */
  toolCalls: Json[];
  inputTokens: number | null;
  outputTokens: number | null;
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as a stop family. */
function openAIFamily(finish: unknown): number {
  if (finish === 'length') return 1;
  if (finish === 'tool_calls' || finish === 'function_call') return 2;
  if (finish === 'content_filter') return 3;
  return 0;
}

/** Anthropic's stop_reason as a stop family. */
function anthropicFamily(reason: unknown): number {
  if (reason === 'max_tokens') return 1;
  if (reason === 'tool_use') return 2;
  if (reason === 'refusal') return 3;
  return 0;
}

/** Gemini's finishReason as a stop family. */
function geminiFamily(reason: unknown): number {
  if (reason === 'MAX_TOKENS') return 1;
  if (reason === 'SAFETY') return 3;
  return 0;
}

/** Parse an OpenAI Chat Completions JSON answer. */
export function parseOpenAIAnswer(body: Json): ProviderAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls = Array.isArray(message.tool_calls) ? (message.tool_calls.filter(isRecord) as Json[]) : [];
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    stopFamily: openAIFamily(choice.finish_reason),
    toolCalls,
    inputTokens: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : null,
    outputTokens: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : null,
  };
}

/** Parse an Anthropic Messages JSON answer. */
export function parseAnthropicAnswer(body: Json): ProviderAnswer | null {
  if (body.type !== 'message' || !Array.isArray(body.content)) return null;
  const text = textOf(body.content);
  const toolCalls: Json[] = [];
  for (const block of body.content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) } });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    stopFamily: anthropicFamily(body.stop_reason),
    toolCalls,
    inputTokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : null,
    outputTokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : null,
  };
}

/** Parse a Gemini generateContent JSON answer. */
export function parseGeminiAnswer(body: Json): ProviderAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    stopFamily: geminiFamily(candidate.finishReason),
    toolCalls: [],
    inputTokens: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : null,
    outputTokens: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : null,
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split SSE text into frames: each frame's event name (or null) and its data payloads. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** Parse an OpenAI SSE stream into an answer: text, stop family, usage. */
export function parseOpenAIStream(text: string): ProviderAnswer {
  let out = '';
  let stopFamily = 0;
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  const toolCalls: Json[] = [];
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrNull(chunk.usage.prompt_tokens) ?? inputTokens;
      outputTokens = numberOrNull(chunk.usage.completion_tokens) ?? outputTokens;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (Array.isArray(delta.tool_calls)) for (const call of delta.tool_calls) if (isRecord(call)) toolCalls.push(call);
    if (typeof choice.finish_reason === 'string') stopFamily = openAIFamily(choice.finish_reason);
  }
  return { text: out, stopFamily, toolCalls, inputTokens, outputTokens };
}

/** Parse an Anthropic SSE stream into an answer. */
export function parseAnthropicStream(text: string): ProviderAnswer {
  let out = '';
  let stopFamily = 0;
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrNull(data.message.usage.input_tokens) ?? inputTokens;
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopFamily = anthropicFamily(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrNull(data.usage.output_tokens) ?? outputTokens;
    }
  }
  return { text: out, stopFamily, toolCalls: [], inputTokens, outputTokens };
}

/** Parse a Gemini SSE stream into an answer. */
export function parseGeminiStream(text: string): ProviderAnswer {
  let out = '';
  let stopFamily = 0;
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) if (isRecord(part) && typeof part.text === 'string') out += part.text;
    if (typeof candidate.finishReason === 'string') stopFamily = geminiFamily(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrNull(data.usageMetadata.promptTokenCount) ?? inputTokens;
      outputTokens = numberOrNull(data.usageMetadata.candidatesTokenCount) ?? outputTokens;
    }
  }
  return { text: out, stopFamily, toolCalls: [], inputTokens, outputTokens };
}

const FIXED_CHAT_ID = 'chatcmpl-aggregate';
const FIXED_MSG_ID = 'msg_aggregate';

/** A provider answer as an OpenAI Chat Completions JSON object. */
export function answerToOpenAI(answer: ProviderAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) message.tool_calls = answer.toolCalls;
  const finish = answer.stopFamily === 1 ? 'length' : answer.stopFamily === 2 ? 'tool_calls' : answer.stopFamily === 3 ? 'content_filter' : 'stop';
  return {
    id: FIXED_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens ?? 0, completion_tokens: answer.outputTokens ?? 0, total_tokens: (answer.inputTokens ?? 0) + (answer.outputTokens ?? 0) },
  };
}

/** A provider answer as an Anthropic Messages JSON object. */
export function answerToAnthropic(answer: ProviderAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const stop = answer.stopFamily === 1 ? 'max_tokens' : answer.stopFamily === 2 ? 'tool_use' : answer.stopFamily === 3 ? 'refusal' : 'end_turn';
  return {
    id: FIXED_MSG_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: answer.outputTokens ?? 0 },
  };
}

/** A provider answer as an OpenAI SSE stream: role chunk, one content chunk, finish, usage, [DONE]. */
export function answerToOpenAIStream(answer: ProviderAnswer, model: string): string {
  const finish = answer.stopFamily === 1 ? 'length' : answer.stopFamily === 2 ? 'tool_calls' : answer.stopFamily === 3 ? 'content_filter' : 'stop';
  const frames: string[] = [];
  frames.push(sseEvent(null, { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  if (answer.text !== '') {
    frames.push(sseEvent(null, { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: answer.text }, finish_reason: null }] }));
  }
  frames.push(sseEvent(null, { id: FIXED_CHAT_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: finish }] }));
  frames.push(
    sseEvent(null, {
      id: FIXED_CHAT_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens ?? 0, completion_tokens: answer.outputTokens ?? 0, total_tokens: (answer.inputTokens ?? 0) + (answer.outputTokens ?? 0) },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A provider answer as an Anthropic SSE stream: message_start, one text block, message_delta, message_stop. */
export function answerToAnthropicStream(answer: ProviderAnswer, model: string): string {
  const stop = answer.stopFamily === 1 ? 'max_tokens' : answer.stopFamily === 2 ? 'tool_use' : answer.stopFamily === 3 ? 'refusal' : 'end_turn';
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: FIXED_MSG_ID, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens ?? 0, output_tokens: 0 } },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  if (answer.text !== '') {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.outputTokens ?? 0 } }));
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

/**
 * An OpenAI SSE stream as the Anthropic event sequence, streamed chunk by chunk.
 * Used when the upstream provider speaks OpenAI and the client speaks Anthropic.
 */
export async function* anthropicEventsFromOpenAIStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop = 0;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: FIXED_MSG_ID, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const stop = pendingStop === 1 ? 'max_tokens' : pendingStop === 2 ? 'tool_use' : pendingStop === 3 ? 'refusal' : 'end_turn';
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents());
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = openAIFamily(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents()) yield event;
  }
}
