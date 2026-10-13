# Reliable dictation and meetings

Branch: `feat/reliable-dictation-meetings`; review base: `55b75ca`.

## Delivery gates

Each implementation milestone has a regression test (observed failing before
the fix), focused validation, and a separate commit. Production recordings and
the running daemon are preserved until an explicit integration check.

| Milestone | Scope | State |
| --- | --- | --- |
| M1 | B01 double PCM; B02 merge loss; B06 empty CLI; B07 dictation tail; B08 clipboard; B09 extension races | pending |
| M2 | B03 continuous source audio; B04 crash recovery; B05 session ownership; bounded queues, pause/resume, 110-minute capture | pending |
| M3 | Paired/authenticated bridge, validated sequenced audio; offline models and diarization preflight | pending |
| M4 | Dictation preview/cancel, configurable shortcuts and silence, languages, vocabulary, persistent history, focus protection | pending |
| M5 | Source health indicators, searchable playback, speaker editing, local summaries and action items | pending |
| M6 | PR, independent subagent review, corrections, final end-to-end verification | pending |

## Acceptance evidence

- No missing tail, no cross-source deletion, no duplicate browser PCM.
- Continuous per-source audio survives a killed writer; source gaps are explicit.
- 110 minutes of synthetic samples retain exact duration without RAM growth
  proportional to meeting length. Accelerated testing is distinguished from
  wall-clock capture and real microphone/browser checks.
- Controls are session-scoped; finalization, file jobs and reconnects cannot
  stop or overwrite another session. A requested stop cannot start recording.
- Unpaired clients cannot read results, alter settings, or start recording.
- Offline mode prevents implicit downloads and disables dependency telemetry;
  missing diarization models are visible before capture, with plain recording
  remaining available.
- Dictation delivery preserves the clipboard and checks the intended target.
- Meeting review works locally, with timestamped playback and editable speakers.
- Existing recordings are never migrated or overwritten implicitly.

## Audit findings

All B01–B09 and the three additional findings are open until their changes and
regression checks land. External model-account permissions cannot be fabricated;
their actual verification state must remain explicit.
