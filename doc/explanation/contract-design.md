# Contract Design Decisions

Why the contract behaves the way the [reference](../reference/contract.md) describes.

## Why validation happens at load

A contract error discovered mid-episode wastes robot time and corrupts data. A contract error discovered at deploy time endangers hardware. So the loader front-loads everything checkable: shape parsing, timeline attestation against the channel's message type, codec and operator resolvability, kind dimension counts, round-trip verification of bijective operators. `load_contract` guarantees a returned `Contract` is fully valid, and nothing downstream re-validates. The cost is a stricter schema. The payoff is that the failure surface lives in one place, before any message flows.

## Why align is mandatory and nothing has a default

Every implicit default is a decision the contract no longer records. A reader of the YAML should reconstruct the exact frame semantics without consulting code. Alignment strategy and timeline choice change what the policy trains on, so the contract forces both to be written down.

## Why every entry reads as one pipeline

Each frame entry follows the same order: `channel` provides, `align` lands samples on the clock, `select` projects fields, `apply` transforms them, and the mapping key names the result. One shape for every entry keeps contracts skimmable. The pipeline reads right-to-left for actions: recording decodes from the channel, serving encodes back to it.

## Why layout follows declaration order

Multi-source keys concatenate in the order sources appear. There is no separate ordering field (e.g. `position: 2`) because a YAML list already expresses order fully: any position for any source, reordered with a one-line diff. An explicit ordering field would duplicate the same information in a second place and let the two drift apart.

## Why `safety` defaults to `none`

A fabricated zero command is only a safe stop under velocity control. Under position control, the common case, zero commands a slam to the zero pose. A default of `zeros` would make the dangerous case the silent one. So the watchdog publishes nothing unless the contract author opts in per channel, choosing `zeros` where velocity semantics make zero safe, or `hold` where re-sending the last command is safe. `zeros` means zeros in action space, run through the inverse `apply` pipeline before encoding, so a declared `clamp` still bounds the fabricated command.

## Why gaps zero-fill instead of skipping frames

After warmup, a stream with no sample at a tick zero-fills at its static dim, and this is not configurable. Live inference never skips a tick: the policy needs an observation every step, and staleness is handled by missing-stream logging and the safety watchdog, not by the frame shape. Offline, dropped frames would silently break the `fps` grid the dataset declares. Bag conversion and the live bridge share the behavior, so a gap looks identical in training data and at inference. Zero-fill is a visible, learnable artifact instead of a silent timing lie.

## Why operators refuse to repair values

`clamp` passes NaN through rather than mapping NaN to a bound. Repairing a non-finite value would hide policy divergence inside a plausible-looking command. Instead the encode path refuses the whole frame (no partial frame across a multi-channel action) and the watchdog applies each channel's declared `safety` if the condition persists. Bounding is a declared transform. Repair would be a silent one.

## Why action operators must invert

Recording runs `apply` forward. Serving runs the same pipeline backward. An operator without a true inverse would make the policy's output drift from what the robot receives, exactly the skew the contract exists to eliminate. The invertibility tiers make the rule checkable: forward-only operators are rejected on actions at load, and bijective operators are round-trip verified at load, so a wrong inverse fails before deployment instead of corrupting actions silently.

## Why the contract embeds into bags and datasets

The recorder embeds the exact contract text into bag metadata, and the porter embeds the contract into the dataset. Provenance beats convention: the artifact carries the translation, so a checkpoint resolves its own contract at deploy time and a bag warns when ported with a semantically different contract. The recorder validates the same bytes the bag stores (`parse_contract` on the read text), so the embedded contract is the validated contract, not a copy of a file that may have changed since.
