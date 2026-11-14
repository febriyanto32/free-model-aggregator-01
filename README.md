# Free Model Aggregator

One server, two APIs, ten free providers. It serves the OpenAI Chat
Completions API and the Anthropic Messages API — each in JSON and in
server-sent events — over the ten simulated free-tier providers a trial gives
it, routing every request to the best free model that can serve it, failing
over when a provider refuses, and reporting the catalog it read.

Built for the make.fun competition
[`free-model-aggregator-01`](https://make.fun/c/free-model-aggregator-01)
(The best free AI model aggregator, $500).

## Run it

```sh
node main.ts            # serve: reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog document and exit 0
```

Node 22, standard library only — nothing is installed. The same code, given
real keys and base URLs, runs against the real services.

## How it works

- **Brief & catalog** (`lib/providers.ts`) — reads the trial brief's three
  documents (provider directory, quality table, workload manifest), probes
  each provider's catalog route in its own wire (OpenAI, Anthropic, Gemini),
  and builds the one catalog document that `--list` prints and `/register`
  posts, byte for byte.
- **Routing** (`lib/router.ts`) — every free model meeting a request's
  requirements is a candidate, ranked by the quality table's rating. Each
  provider's free-tier budget is drawn per trial and never published, so the
  router soft-caps each provider at its sealed minimum budget minus two (the
  failure schedule's refused calls) and spreads the load instead of draining
  the best provider dry. A refusal costs a small rating penalty (8 milli a
  refusal, 25 over the soft cap), so a provider that failed once only drops
  behind a near-equal rival; a provider that refuses twice in a row is treated
  as spent. A JSON-mode OpenAI request never rides an Anthropic-wire provider
  (that wire cannot carry `response_format`), and a request with assistant or
  tool messages never rides a Gemini-wire provider.
- **Wires** (`lib/translate.ts`) — requests are translated through a canonical
  form that carries the system prompt, user text, image bytes, tool names,
  stop sequences, `max_tokens` and JSON mode exactly (the provider echoes what
  it parsed, and the client compares that echo with what it sent), and answers
  come back with the provider's text whole, its usage counts as that wire
  reports them, and its stop reason mapped into the client's family. Same-wire
  streams pass through byte for byte.
- **Trial loop** (`main.ts`) — listens on an ephemeral port, registers, serves
  the client's 20 requests one at a time, polls `GET /work` and exits 0 when
  the workload is done. No clock, no unseeded randomness: a replay makes the
  same choices and leaves the same state export.

## Local results (practice trials, sealed scorer)

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 28 |

20 of 20 requests valid on every trial; `--list` output matches the
registered catalog exactly; re-running trial 0 leaves a byte-identical state
export. Quality 100 is practically unreachable: every trial's failure schedule
forces at least one request onto a second-best model.

## Layout

```
main.ts            the server: brief, catalog, routes, register, poll
lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe
lib/router.ts      quality-ordered, budget-aware routing and failover state
lib/translate.ts   the client wires and the three upstream wires
LICENSE            MIT
```

## License

MIT — see `LICENSE`.
