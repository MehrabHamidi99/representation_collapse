# Address acquisition v4 — independent prospective review

2026-10-13. The prepared implementation and prospective contract pass independent
review at the exact hashes below. This accepts engineering readiness for the
separately frozen local discovery experiment. No v4 production result or
scientific mechanism is accepted.

The [protocol](address-acquisition-discovery-v4.md) tests predictable singleton
QUERY addresses 0, 7 and 15 with the original sixteen ordered WRITEs and common
terminal QUERY. Twelve fresh seed-0 fits have fixed 5,000-update endpoints,
60,000 total optimizer updates and a 1,200-second engine deadline. Known future
target, required load, selection uncertainty, address identity and lag differ.
The protocol correctly excludes pure capacity/lag causality and main v2/v3
eligibility.

## Checks and evidence

All **12 focused tests** pass with no skips, using explicit reduced-budget
infrastructure fixtures. The reviewer additionally checked **98** adapter,
oracle, reference and stream conditions over GRU/LSTM, categorical/continuous
tasks and all three addresses. Complete native output/state replay agrees;
LSTM contains hidden and cell state. WRITE payloads/order and the original RNG
advancement are preserved, QUERY carries no payload, and the original adapter
still rejects QUERY 15.

Separate two-update fixtures for GRU categorical QUERY 7 and LSTM continuous
QUERY 7 pass **39** independent checks. Pinned-baseline and endpoint arrays have
identical new validation inputs, targets, selectors and pair IDs. Direct
reduction verifies singleton-minus-baseline native-error sign and fixed
population MSE normalization by 1/3. Both new AdamW states begin empty; baseline
reevaluation performs no optimization. Current endpoints are retained without
selection, and QUERY 15 has no primary baseline evaluation or contrast.

The production entrypoint requires the current implementation/configuration,
clean committed source and an external coordinator receipt binding the
independent review before entering the experiment engine. No implicit fixture
fallback exists. A fixture must reduce an update/fit/wall bound; it cannot
masquerade as the complete production envelope. Failed or censored fits retain
realized update counts and diagnostic current weights with unknown endpoint
competence, rather than inventing gate failure or resume authority.

## Exact acceptance and remaining gates

- Configuration: `55c564c5c607ce3cf298e4daabaeafb2700d50419a4ee859200e0c9bba811c55`.
- Implementation: `cb06f60d45cd26e27f1e8c4959de05d183544568eb19614f60d72d9636376958`.
- Tests: `4b5d52a8c6bb7b951fee7f5ef07bef1dc8279426495a6f17f3714ee1e7b450db`.
- Protocol: `2232689fea45602807914748731fa97c5613ef71c5185e84f8f7b967f745fccd`.

Preparation metadata stays immutable. The coordinator's separate ignored
receipt records source commitment and execution authority. Actual production,
artifact reduction and scientific acceptance are subsequent gates. Existing
v2/v3 runs and their failures remain unchanged.

Supported baseline comparisons are restricted to QUERY 0/7 and use exact new
seed-20003 validation histories. Historical seed-20001 scores are provenance
only. QUERY 15 falls outside the original random-QUERY training support.
Validation draws are shared across address conditions and architectures; they
are not independent training seeds. The study can report scoped acquisition
and paired conditional errors, with no periodicity harm, Floquet certification,
equivalence, architectural impossibility or population-level conclusion.

Ignored receipt:
`runs/review/address-acquisition-v4-implementation-review-20261013.json`.
This reviewer performed no provider, cluster, Git or source action and no
production training.
