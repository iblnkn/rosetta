# Code Review Findings — demo/202605-madison PR(s)

> Generated 2026-06-15 from high-effort recall reviews of the `rosetta` and
> `sns_robot_learning` branches (`demo/202605-madison-clean-up`). Each item was
> independently verified (CONFIRMED/PLAUSIBLE). This is a backlog to triage
> **after** the working finalize — not a blocker list.

**Status legend:** ✅ fixed · 🟡 partially fixed · ⬜ open · 🚫 won't-fix-now (intentional)

Severity: 🔴 breaks documented functionality / silent wrong robot behavior ·
🟠 breaks pipeline outside happy path · 🟡 latent / quality / efficiency.

---

## rosetta package

### 🔴 ⬜ 1. RTC server breaks every policy type except `sns_diffusion`
- **Where:** `rosetta/common/rtc_policy_server.py:321` (`_build_stacked_observations`), entered via `rosetta/common/policy_server.py` (now imports `serve` from `rtc_policy_server`).
- **What:** The new server is the serving path for *all* policies but whitelists only `observation.state` / `observation.environment_state` / `OBS_IMAGES` and force-adds an `n_obs_steps` temporal dim. ACT (the node default) → `KeyError` on per-camera keys; upstream diffusion → `torch.stack` on empty `_queues` (deleted monkey-patch used to populate them); smolvla/pi0 → lose the `task` key. README/CLAUDE.md still document deploying `act`/`diffusion`.
- **Fix direction:** Gate the stacking on policies that actually want `(B, n_obs_steps, …)`; pass the untouched observation dict through for everything else (make the RTC override a superset of upstream, not a replacement).

### 🟠 ✅ 2. Decoder resize removed → `port_bags` fails for shape-declaring contracts
- **Where:** `rosetta/common/decoders.py:240,258`; `rosetta/port_bags.py`.
- **What:** Decoders now return native resolution; `build_feature` still declares the dataset feature from contract `image.shape`, so the default (no-processor) port path wrote mismatched frames → `add_frame` rejects them. All shipped contracts declare non-native shapes (224×224, 512×512).
- **Fixed:** `port_bags` now resizes image frames to the contract `image_shape` (nearest-neighbor, no crop) when no observation processor is given — `_conform_images_to_specs` in `rosetta/port_bags.py`. **Working-tree change, not yet committed.** Also covers the action-processor-only path.
- **Still deeper (open):** image-shape responsibility is split across contract `image.shape`, decoder, processor `resize_size`/crop, and policy config — kept in sync by hand. Consider deriving the processor resize from the contract spec, or one validation that decoded frames match the declared feature.

### 🔴 ⬜ 3. Failed contract swap → silent standalone mode on wrong topics
- **Where:** `rosetta/rosetta_client_node.py:807` (`_swap_contract_if_needed`).
- **What:** On a failed swap the old bridge is torn down but `self._contract_path` still points at the old contract. A later goal for that contract no-ops the swap check with `self._bridge is None`; `_build_config` injects `_external_bridge=None`, and `Rosetta.connect()` silently falls into standalone mode (own node, root namespace, no launch remappings) → obs/actions on wrong topics, no error.
- **Fix direction:** clear `self._contract_path = None` on the failure path, **or** require `self._bridge is not None` in the early-return guard.

### 🔴 ⬜ 4. Launch dropped `pretrained_name_or_path` / `policy_type` args; docs still use them
- **Where:** `launch/rosetta_client_launch.py:178` (arg declarations).
- **What:** The two args were removed from the launch file, but README (same PR) and `CLAUDE.md` still document them. `ros2 launch` silently ignores unknown `name:=value` pairs, so the documented deploy command discards the user's checkpoint — goal rejected ("No pretrained_name_or_path"), or a registry default checkpoint runs on the robot instead of the requested one.
- **Fix direction:** re-declare + forward the args (cheap one-off deploy path), and/or update README + CLAUDE.md to the registry flow in the same change.

### 🔴 ⬜ 5. Registry: one stale entry bricks the node; typo'd keys vanish silently
- **Where:** `rosetta/common/policy_registry.py:89` (`_parse_entry` → `validate_pretrained`), `rosetta_client_node.py` `on_configure`.
- **What (a):** `load_registry` eagerly path-validates *every* entry at configure; one missing `/root/ws_rl/src/models/...` checkpoint → `on_configure` FAILURE, node never activates — even for goals passing an explicit pretrained path. This is the *default* launch experience (default params wire the registry on).
- **What (b):** `_parse_entry` iterates `_OPTIONAL_FIELDS` only; unknown/typo'd keys (e.g. `action_per_chunk:`) are dropped with no log → node default silently applies.
- **Fix direction:** validate lazily per requested entry (or warn-and-skip bad entries at load); add an unknown-key check that raises/warns.

### 🟠 ⬜ 6. Launch defaults hardcode a container-only path into a sibling repo
- **Where:** `launch/rosetta_client_launch.py:169`, `launch/rosetta_hil_launch.py:80`. This repo's own `params/rosetta_client.yaml` was deleted.
- **What:** `default_params = "/root/ws_rl/src/sns_robot_learning/params/rosetta_client.yaml"`. Fails anywhere not mounted at `/root/ws_rl` (host checkout, CI, standalone). HIL launch is worse — it `open()`s the path inside `generate_launch_description()`, so it raises at description-build time and can't be rescued by `params_file:=`.
- **Fix direction:** ship a minimal default params file in rosetta's `share/` (restore `get_package_share_directory`); let `sns_robot_learning` override **via** `params_file:=`.

### 🟡 ⬜ 7. Policy server launches at fps=30 before the contract is known
- **Where:** `rosetta/rosetta_client_node.py:416` (`_start_policy_server`, runs in `on_activate`).
- **What:** In registry-owned mode (`contract_path` launch default empty) `_rosetta_config` is `None` at activate → server gets `--fps=30`. `_swap_contract_if_needed` never restarts the server. Client uses the correct per-goal fps, so for a non-30-fps contract `environment_dt` is wrong, skewing `inference_delay`, consumption simulation, and `TimedAction` timestamps. **Latent today** (all inference contracts are fps 30; only `record/emily_right_arm_cart.yaml` is 10).
- **Fix direction:** (re)start the server at goal time once the contract resolves, or pass fps per-connection.

### 🟡 ⬜ 8. Unbounded ActionQueue growth for non-RTC policies
- **Where:** `rosetta/common/rtc_policy_server.py:206` (merge) vs `_rtc_enabled()`-gated consumption.
- **What:** With RTC disabled, a fallback `ActionQueue` is still created and `merge()`d every inference, but nothing consumes it (`last_index` stays 0; trim is `[0:]`). Queue tensors grow by `actions_per_chunk` rows per inference for the connection lifetime (+2 `.clone()`s/chunk) → creeping OOM on long deployments.
- **Fix direction:** skip ActionQueue creation + merge entirely when RTC is disabled.

### 🟡 ⬜ 9. RTC bookkeeping estimates what the protocol already reports exactly
- **Where:** `rosetta/common/rtc_policy_server.py:351` (`_simulate_client_consumption`) and `inference_delay` from `self._last_inference_time`.
- **What (a):** Server advances `last_index` from `int(elapsed/environment_dt)` although the client already stamps each observation with `timestep = max(latest_action, 0)` (exact consumed index). Drifts under pauses/hiccups/starvation → RTC blends against the wrong prefix.
- **What (b):** `inference_delay` uses the previous single-sample latency; bimodal latency underestimates the frozen prefix. Upstream uses `LatencyTracker.max()` (already vendored).
- **Fix direction:** advance `last_index` from the client-reported timestep; adopt `LatencyTracker`.

### 🟡 ⬜ 10. `_OBS_HISTORY_MAXLEN = 4` is a comment-enforced invariant
- **Where:** `rosetta/common/robot_client.py:37`; server pad at `rtc_policy_server.py:304`.
- **What:** Client history is hardcoded to 4; `n_obs_steps > 4` silently pads with the oldest frame forever (log prints `used=n_obs_steps`, masking it) → OOD temporal context, no diagnostic. Also ships 4 obs/request when policy uses 2.
- **Fix direction:** make the window per-policy config (PolicyBundle/registry, next to `actions_per_chunk`); validate `n_obs_steps <= maxlen`; warn on post-rampup padding.

### Cut / lower-priority (rosetta) — all CONFIRMED
- 🟡 ⬜ **Debug leftover spam:** `robot_client.py:408` injects `_capture_time` into every obs; `rtc_policy_server.py:328-348` logs ~20 lines of spacing diagnostics at INFO per inference, ungated. Delete or demote to `logger.debug`. The smuggled key survives only because helpers drop unknown keys.
- 🟡 ⬜ **Duplication:** `policy_server.py` is a misdirecting shim (docstring/comment still claim it delegates to upstream lerobot `serve()`); `_get_action_chunk_with_kwargs`, ~40 lines of `_predict_action_chunk`, and `serve()` are verbatim upstream copies; generic server hardcodes `import lerobot_policy_sns_diffusion` (×2) + one-name `SUPPORTED_POLICIES` patch → ImportError without that plugin even when serving ACT. Use upstream `register_third_party_plugins()`.
- 🟡 ⬜ **Hot-path efficiency:** patched control loop runs the full cv2 crop/resize processor at 30 Hz on the control thread (~85% of frames never sent; upstream gates on send ticks); server preprocesses history frames one-at-a-time at batch=1 and postprocesses chunks in a ~50-iter Python loop where one batched call would do.

---

## sns_robot_learning

### 🔴 ⬜ 1. Default `policy_registry.yaml` bricks `rosetta_client` on missing checkpoints
- **Where:** `params/policy_registry.yaml` (3 entries → `/root/ws_rl/src/models/...`), made default by `params/rosetta_client.yaml`.
- **Mirror of rosetta finding #5(a).** One missing checkpoint dir fails the node's configure transition. Fix belongs on the rosetta side (lazy validation) and/or here (don't ship machine-local absolute paths as the default).

### 🔴 ⬜ 2. RTC prefix misaligned by `n_obs_steps-1` in the diffusion-RTC integration
- **Where:** `policies/lerobot_policy_sns_diffusion/src/lerobot_policy_sns_diffusion/modeling_sns_diffusion.py:409` + `modeling_rtc.py` `denoise_step`.
- **What:** `generate_actions` slices the chunk at horizon index `n_obs_steps-1` ("now"), but the leftover prefix and `get_prefix_weights` are anchored at horizon index 0. With default `n_obs_steps=2`, every RTC constraint applies one frame early → systematic ~1-frame discontinuity at every chunk seam (the artifact RTC exists to remove). Vendored tests only exercise `denoise_step` in isolation, so nothing catches it.
- **Fix direction:** pad the leftover and build prefix weights anchored at horizon index `n_obs_steps-1`, or run guidance on the sliced `[n_obs_steps-1 : …]` window.

### 🟠 🚫 3. `record/emily_left_only.yaml` feature-incompatible with inference counterpart
- **Where:** `rosetta_contracts/record/emily_left_only.yaml` vs `inference/emily_left_only.yaml`.
- **What:** Different image key (`left_wrist` vs `left_arm_d405`), right-arm state selectors copy-pasted from `emily_isaac` into a left-only contract, 6-dim JointTrajectory action vs 8-dim DMP Float64MultiArray. Recording with the record contract yields a dataset that can't deploy via the inference contract.
- **Status:** user asked to leave YAML unchanged for now. Revisit when touching contracts.

### 🔴 ⬜ 4. `rosetta_client_openarm.yaml` not migrated to registry-style config
- **Where:** `params/rosetta_client_openarm.yaml`.
- **What:** Lacks `observation_processor_path` and `policy_registry_path`; the new node rejects goals when no processor resolves (old "empty = identity" behavior removed). Launching openarm → every goal rejected. Param descriptor still falsely claims "empty = identity processor".
- **Fix direction:** migrate the file to the new keys (or restore an identity-processor fallback in the node).

### eval_offline.py — ✅ FIXED (committed as `fixup! Add eval_offline script`)
- ✅ **RTC leftover space mismatch** (`:265`): tail now taken from the model-space chunk before postprocessing.
- ✅ **Cross-episode history leak** (`:162`): history clamped to `ep_from`.
- ✅ **Dead `inference_delay` lookup** (`:573`): removed the nonexistent `RTCConfig.inference_delay` read; banner reports RTC from `enabled`; honest `--inference-delay` default.
- ✅ **Efficiency:** read action column instead of full `__getitem__` video decode; compute `aggregate_chunks` once per episode; memoize duplicate frame fetches.

### 🟡 ⬜ 5. `EVAL_EPISODES=5` means "episode index 5", not "5 episodes"
- **Where:** `scripts/train_queue/run_experiment.sh:137` → `eval_offline.py` `parse_episodes`.
- **What:** Default evaluates only episode #5 and crashes (under `set -e`, aborting before upload) on datasets with ≤5 episodes.
- **Fix direction:** decide the intended semantics; if "count", convert to indices `0..N-1`.

### 🟡 ⬜ 6. `run_experiment.sh` clobbers `JOB_NAME` / `POLICY_REPO_ID` after sourcing config
- **Where:** `scripts/train_queue/run_experiment.sh:108-109` (plain assignment, not `${VAR:-…}`).
- **What:** Config overrides are silently discarded; training/upload always uses hardcoded `Emily/${EXP_NAME}`. Example config documents these as overridable. Doc drift on R2 prefix too.
- **Fix direction:** use `${JOB_NAME:-${EXP_NAME}}` / `${POLICY_REPO_ID:-…}`; reconcile docs.

### worker.sh queue — 🟡 partially fixed
- ✅ **Stale lock** (`scripts/train_queue/worker.sh`): `EXIT/INT/TERM` trap releases the lock (committed as `fixup! Add script to simplify workflow`).
- ⬜ **Lost-append race:** `submit.sh` still appends with no lock while `pop_job` rewrites via `tail`+`mv`; an append landing across the rewrite is destroyed. Fix: take the same lock in `submit.sh`, or use an atomic-append-safe pop.

### modeling_rtc.py allocation — 🟡 partially fixed
- ✅ Per-step pad/weights now allocate with `device=`/`dtype=` directly (committed as `fixup! Add RTC`).
- ⬜ Still rebuilt inside the denoise-step loop though `(delay, horizon, shape)` is constant per `conditional_sample`; hoist out of the loop.

### Cut / lower-priority (sns_robot_learning) — all CONFIRMED
- 🟡 ⬜ **`inference/emily_right_arm.yaml:44`**: `/joint_states` flipped to `reliable` + `stamp: receive` while record uses `best_effort` + `header` → DDS-incompatible if publisher is best-effort (policy runs on zero-filled state); `robot_type` rename mismatch.
- 🟡 ⬜ **`params/rosetta_client.yaml:51`**: node-default fallback processor `default_resize_224` (no crop) silently substitutes a different transform than crop-trained policies saw; ad-hoc goals always hit it.
- 🟡 ⬜ **`scripts/lerobot_train.py`**: 598-line fork of vendored upstream with a ~32-line real delta (GPU image transforms) → drifts on next lerobot bump. Prefer an upstream patch / thin wrapper.
- 🟡 ⬜ **`modeling_rtc.py` / `modeling_sns_diffusion.py`**: ~250 + ~380 lines near-verbatim upstream copies + ~840 lines of copied tests. A subclass overriding `denoise_step` with `predict_x0_fn` would do.
- 🟡 ⬜ **`scripts/train_queue/patch_inference_config.py`**: destructively rewrites checkpoint `config.json` in place (no backup) before upload, baking deploy-time DDIM/RTC settings into the artifact. Plumb overrides through the `PolicyBundle` registry instead.

---

## Suggested order when you return to this
1. **rosetta #1** (ACT/diffusion deploy is broken) and **#3** (silent wrong-topic mode) — correctness, small fixes.
2. **rosetta #4/#5/#6** + **sns #1/#4** — the "default launch experience" cluster (registry/params/launch coupling across both repos).
3. **sns #2** RTC alignment — affects on-robot RTC quality.
4. Commit the **rosetta #2** working-tree port_bags fix.
5. Everything else 🟡 (efficiency, dedup, doc drift, debug leftovers).
