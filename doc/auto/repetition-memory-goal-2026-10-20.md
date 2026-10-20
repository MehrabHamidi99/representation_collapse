# Immediate research goal: repetition and useful memory

Owner instruction, 2026-10-20: study representation collapse under repetition
and memorization tasks, run longer A/B repetition experiments, and report results.
This is the immediate subgoal of the existing novelty-adjusted research goal.
The tool-level existing goal is paused and cannot be replaced while unfinished;
this document records the current requested work without falsely marking the
broader study complete or claiming a tool-level status change.

1. Run the frozen fixed-checkpoint GC-LSTM dwell sweep in
   [the repetition protocol](repetition-dwell-protocol-2026-10-20.md), independently
   verify its results, and plot all prespecified conditions.
2. Distinguish elapsed horizon, motif dwell, transition frequency, and startup.
   Report performance and complete-state geometry separately, including negative
   or nonmonotone effects. This checkpoint was trained at dwell32; inference at
   other dwell lengths is a temporal distribution shift.
3. Prepare a competence-first independently written memory task with matched
   retention horizons, periodic/nonperiodic order controls, native query accuracy,
   held-out cue recovery, and causal state interventions with shams. Freeze its
   full task and statistical protocol before training/held-out interpretation.
4. Test whether extra repetition specifically destroys useful history, whether
   readout timing explains errors, or whether cue-dependent attractor states
   preserve memory despite within-cue contraction. Replicate across seeds and
   modalities only after native competence and measurement validation.

Success is a reproducible answer about when repetition harms required memory,
including a valid preservation or null finding. It does not require a positive
collapse result. The graph dwell sweep alone does not measure memory capacity.
