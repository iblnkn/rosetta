^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rerun_rosetta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* ``RerunReader``, the ``rerun`` frame reader: ``rosetta-lerobot-train
  --rosetta.reader=rerun`` trains any LeRobot policy on a Rerun catalog. It
  serves samples by index through Rerun's map-style dataset, one catalog query
  per batch, or streams them in the order of its ``shuffle`` (``sample``,
  ``block`` with ``shuffle_buffer``, or ``none``). ``frame_index`` and
  ``timestamp`` count from an episode's first frame after warmup, as the port
  numbers them. ``index`` counts only the sample starts Rerun keeps, so where
  windows trim it runs behind a ported dataset's. Each episode's task is the
  operator prompt stored in the recording. Windows are trimmed at episode
  edges, so ``<key>_is_pad`` is always false, and a recording too short for
  the windows isn't an episode. A contract with a ``tasks`` section is
  refused.
* Window offsets go to Rerun in whole steps of ``round(1e9 / fps)``. As
  ``k / fps`` seconds, an observation history at a rate such as 15 fps put
  every sample a nanosecond before its tick, so it was numbered one frame low
  and missed a message stamped on the tick. Breaking: a saved manifest for a
  windowed contract at 15, 30 or 60 fps no longer matches. Delete it to
  rebuild.
* ``RerunReader``, ``write_stats_layer`` and ``load_stats`` skip a segment
  with no data for an observation stream, with a warning, as the porter skips
  such an episode. ``layout.observation_starts`` does the check.
  ``load_stats`` takes ``segments``, by default those segments.
* ``ContractFrames`` gains ``read(rows)``, ``positions()``, ``reseed(seed)``
  and ``index_name``. ``ContractFrames(positions=True)`` adds each sample's
  segment and tick.
* A windowed field's decoded values are cached per window tick, up to
  ``DEFAULT_CACHE_BYTES`` per stream and worker, so a tick is decoded once
  rather than once for every sample whose window covers it. Every sample gets
  its own copy of a cached value.
* Breaking: ``rerun_rosetta.stats`` no longer defines ``QUANTILES`` and
  ``quantile_name``. Import them from ``rosetta.data.stats``, which now holds
  the stats math.
* Breaking: ``examples/train_act.py`` is removed. ``rosetta-lerobot-train``
  replaces it for every LeRobot policy.
