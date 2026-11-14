// Quality-ordered, budget-aware routing over the probed providers.
//
// For one request, every free model that meets the request's requirements is a
// candidate, ranked by the quality table's rating (best first), then provider
// id, then model id — a fixed order, so the same brief and the same provider
// answers always pick the same model. Two wire rules narrow the list:
//
//   - an OpenAI client asking for JSON mode cannot ride an Anthropic-wire
//     provider: that wire cannot carry response_format, so the provider would
//     echo json_mode false and the request would fail its echo check;
//   - a request whose body holds assistant or tool messages cannot ride a
//     Gemini-wire provider: a Gemini content list cannot carry them faithfully.
//
// Each provider's free-tier budget is drawn per trial and never published, but
// the sealed table's minimum budgets are known: a provider always answers at
// least that many calls, minus the one or two calls its failure schedule
// refuses. The router therefore soft-caps each provider at (minimum budget - 2)
// calls: the best models still serve the heavy requests, and the load spreads
// across providers instead of draining the best one dry and losing the rest.
// A provider that refuses twice in a row is treated as spent or out and gets no
// more traffic this trial; a provider that refused once stays a candidate,
// charged a small rating penalty (8 milli a refusal, 25 over its soft cap),
// so it only drops behind a near-equal rival. When every candidate is
// capped or dead, the caps are ignored rather than leaving a request unserved:
// a served request at a lower rating beats an unserved one.
//
// All state here is derived from the brief, the requests and the providers'
// answers — no clock, no unseeded randomness — so a replay makes the same
// choices.

import { capabilitiesOf, meetsRequirements, modelFree, ratingOfModel, type Brief, type Probe, type ProbedModel, type RatingsFile, type Requirement } from './providers.ts';
import type { Canon } from './translate.ts';

/** One (provider, model) pair that can serve a request. */
export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** The sealed table's minimum per-trial call budget for each provider. */
const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** How many calls a provider may take before the router spreads the load. */
function softCap(providerId: string): number {
  const min = MIN_BUDGET[providerId] ?? 5;
  return Math.max(1, min - 2);
}

/** Per-provider call state for this trial. */
export interface ProviderState {
  calls: number;
  failures: number;
  dead: boolean;
}

export class Router {
  private readonly probes: Probe[];
  private readonly ratings: RatingsFile | null;
  private readonly state = new Map<string, ProviderState>();

  constructor(probes: Probe[], ratings: RatingsFile | null) {
    this.probes = probes;
    this.ratings = ratings;
    for (const probe of probes) this.state.set(probe.provider.id, { calls: 0, failures: 0, dead: false });
  }

  private stateOf(providerId: string): ProviderState {
    let s = this.state.get(providerId);
    if (s === undefined) {
      s = { calls: 0, failures: 0, dead: false };
      this.state.set(providerId, s);
    }
    return s;
  }

  /** Record one provider call: whether it answered, and whether it refused. */
  recordCall(providerId: string, ok: boolean): void {
    const s = this.stateOf(providerId);
    s.calls += 1;
    if (!ok) {
      s.failures += 1;
      if (s.failures >= 2) s.dead = true;
    }
  }

  /** Mark a provider dead after a refusal that proves it spent or out: 402, or a 429 after a refusal already. */
  recordRefusal(providerId: string, status: number): void {
    const s = this.stateOf(providerId);
    s.calls += 1;
    s.failures += 1;
    if (status === 402 || s.failures >= 2) s.dead = true;
  }

  /**
   * Every free model that meets the requirements, best rating first. `clientApi`
   * is the wire the client used; `canon` carries the JSON-mode and
   * complex-message flags that narrow the wires.
   */
  candidates(requires: Requirement, clientApi: 'openai' | 'anthropic', canon: Canon): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      if (clientApi === 'openai' && canon.jsonMode && probe.provider.wire === 'anthropic') continue;
      if (canon.complexMessages && probe.provider.wire === 'gemini') continue;
      for (const model of probe.models) {
        const row = ratingOfModel(this.ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
        out.push({ probe, model, rating: row?.rating_milli ?? 0 });
      }
    }
    out.sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
    return out;
  }

  /**
 * The order to try the candidates in: the rating, less a small penalty for a
 * provider that has refused or is over its soft cap. A single refused call
 * (the failure schedule refuses one or two calls on every provider) costs 8
 * milli — about the gap between the top two models — so a provider that failed
 * once still leads when it holds the best model, and only drops behind a
 * near-equal rival. A provider over its soft cap costs 25. A dead provider
 * costs 10,000: it is tried only when nothing else is left. `tried` providers
 * are skipped: a provider that already refused this request is not tried again
 * for it.
 */
  orderFor(cands: Candidate[], tried: Set<string>): Candidate[] {
    const effective = (c: Candidate): number => {
      const s = this.stateOf(c.probe.provider.id);
      if (tried.has(c.probe.provider.id)) return -1e9;
      let r = c.rating;
      if (s.dead) r -= 10000;
      else {
        if (s.calls >= softCap(c.probe.provider.id)) r -= 25;
        r -= 8 * s.failures;
      }
      return r;
    };
    return [...cands].sort((a, b) => {
      const ea = effective(a);
      const eb = effective(b);
      if (ea !== eb) return eb - ea;
      if (b.rating !== a.rating) return b.rating - a.rating;
      if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
      return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
    });
  }

  /** How many provider calls the trial has made so far (for the log line). */
  totalCalls(): number {
    let n = 0;
    for (const s of this.state.values()) n += s.calls;
    return n;
  }
}

/** The brief's manifest requirements for one arriving request, by arrival order. */
export function requirementsFor(brief: Brief, index: number, canon: Canon): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry !== undefined) return entry.requires;
  return {
    tools: canon.tools.length > 0,
    vision: canon.image !== null,
    json: canon.jsonMode,
  };
}
