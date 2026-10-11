# H63 candidate hypotheses — specified BEFORE the run (2026-10-11, GEMSDOE63)

Lane constraint: this session owns the **robust-selection lane** — the emission is
chosen and shaped by CVaR across plausible label worlds (Rockafellar & Uryasev,
2000). The hypotheses below therefore differ in the *geological content fed into
the fixed CVaR engine*, not in the selection rule. Each names layers, physical
signature, why it should catch a catalogue-missing fault, and how it differs from
everything already implemented across the family repos.

## H63-A — Corroboration-weighted multi-physics posterior, posterior-ordered thinning (IMPLEMENT, top ranked)

- **Layers:** four independently-scored prior surfaces already in the family
  registry: h19-5 (multiline corroborated geophysics, live 0.1922), h19-4
  (openness + thermal population blend, 0.1894), h16-1 (topo-geophys baseline
  ridges, 0.1855), ens12 (ensemble-of-12, 0.1563).
- **Physical signature:** *coincidence* — a pixel scores only where
  geophysically independent lineament detectors agree (edge/gradient ridges of
  magnetic, gravity, topographic families), i.e. a structure boundary, not a
  single-instrument artefact.
- **Why it should catch a missing fault:** the GEMSDOE32 d28 autopsy measured
  two defects in the live-0.26 emission: 1,732 *uncorroborated* speckle dots,
  and row-major thinning letting low-posterior endpoints suppress ridge crests.
  Independent-physics agreement hit-rates on hidden faults beat single-surface
  speckle; posterior-ordered greedy thinning guarantees crest dots are never
  suppressed by weaker neighbours.
- **Differs from prior art:** every prior dotted emission (d1.5, d2.8, d28,
  h27-4, h33-2-b2) thinned a single surface in row-major or lattice order;
  none used a corroboration-multiplied posterior with ordered thinning.
- **Expected DTI gain:** +0.005–0.015 vs the d28/h33 line on the lane
  instrument. **Cost: low** (all rasters pinned in the shared data manifest).

## H63-B — SGMC-corroborated augmentation (IMPLEMENT as candidate variant)

- **Layers:** `derived_sgmc_faults_100m_u8.tif` (off-catalogue surrogate fault
  traces the group built earlier), gated by `P > 0` evidence.
- **Physical signature:** proximity to surrogate-mapped but catalogue-absent
  structures — the population the private round-2 labels are argued to add.
- **Why:** the template's SGMC prevalence instrument is calibrated to the
  hidden-truth population size (12,691 px); mass near surrogate traces should
  have higher hit probability *if* the surrogate reflects the private truth.
- **Differs from prior art:** GEMSDOE32 used SGMC only as a scoring surrogate,
  never as an emission prior gated by multi-physics evidence.
- **Expected DTI gain:** 0 to +0.01 (risk: surrogate = circular). **Cost: low.**
  Scored head-to-head with H63-A variants inside the same CVaR ensemble — the
  engine, not intuition, decides.

## H63-C — CVaR-optimal spacing/budget scan on the label-world ensemble (IMPLEMENT — the lane's core statistic)

- Not a new surface; the emission **policy** itself is the hypothesis: the
  spacing/budget that maximizes mean−tail DTI robustly across shifted/dropped/
  decoyed worlds differs from the spacing that maximizes the unperturbed
  holdout (catalogue offsets up to 400 m, Hermant et al., smear the 300 m
  kernel, favouring marginally different dot density than the nominal optimum).
- **Expected DTI gain:** +0.002–0.008 robustness. **Cost: low.**

## H63-D — Drainage/valley-chain structural-control field (NOT implemented this session)

- GEMSDOE32's h33 probe found this label-free field scored 0.09072 on the SGMC
  off-catalogue instrument vs 0.06059 incumbent, but the catalogue-hidden
  instrument disagreed in sign; promotion stalled on instrument disagreement.
- **Needs:** 1 m/10 m DEM derivatives (valley network extraction). The 419 MB
  19-band stack is mirror-pinned and obtainable, but valley extraction plus a
  new disagreement analysis does not fit the 3-experiment budget.
- **Verdict:** deferred; named with its blocker, not faked.

## H63-E — USGS FDSN seismicity-anchored swarm conduits (DATA-BLOCKED, named for completeness)

- GEMSDOE32 measured that the supplied seismic band is spatially smooth at
  fault scale (autocorrelation 0.9986 at 1 km) and ships a fetcher for the raw
  FDSN catalogue; `earthquake.usgs.gov` is unreachable from this sandbox
  (same egress class as there), so obtainability is NOT claimed. Requires the
  official source: https://earthquake.usgs.gov/fdsnws/event/1/ (USGS FDSN).
- **Verdict:** data-blocked; not proposed as viable in this run.

## Ranking (expected DTI gain ÷ cost)

1. **H63-A** — implement (posterior + ordered thinning + speckle filter)
2. **H63-C** — implement (it's the lane's mandated selection rule)
3. **H63-B** — implement as variant inside the same candidate scan
4. **H63-D** — deferred (instrument disagreement unresolved)
5. **H63-E** — data-blocked (official source identified, not reachable here)

## Validation gate

Nothing here touches a submission slot unless the selected artifact beats the
family anchors (d28 0.2600, h33-2-b2 0.2778) on the session's declared lane
instrument (CVaR J across the label-world ensemble), carries all validator
checks green, and passes the parallel-run uniqueness limits
(rank-corr ≤ 0.90, dot overlap < 70% vs every registry raster).
