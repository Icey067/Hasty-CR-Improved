# Portrait Studio

Open `studio.cmd` or run `./studio.ps1`. The portrait showcase uses dedicated
columns: uncropped gameplay on the left and the independent 3x simulator on the
right. Neither simulator content nor its label/border/shadow covers gameplay.
The game is centered vertically, leaving top and bottom space for captions.
No branding, headings or decision cards are included. Overlay drag/resize was
removed so it cannot accidentally obscure gameplay again.

```powershell
.\studio.ps1 -Record -Presentation
.\studio.ps1 -Layout sim -SimSpeed 3
```

`P` toggles the clean portrait preview and restores operator controls. `R`
starts/stops recording, `S` saves a full-resolution PNG, and `5` selects the
showcase. The Logs button opens the optional desktop console. Recording always
uses the original **1080x1920** canvas, not the resized preview, controls or
console. No manual crop or desktop screen recording is needed. Clips are saved
to `tmp/live/studio/`.

Exports use H.264/yuv420p, 60 fps, CRF 16, with fast-start MP4 metadata. The
encoder uses `veryfast` to preserve headroom for simulation and capture. The
operator status reports actual capture resolution and encoder lag. To reduce
encoding load, the Python launcher accepts `--record-fps 30`.

Window capture on a background worker is the default. Keep the emulator
unobscured: an occluded window can stop presenting new frames. `-HQ` opts into
native device-stream capture; `-WindowCapture` explicitly selects the default.
ADB screenshot fallback is also opt-in (`--adb-fallback` in the Python launcher),
so the mirror does not silently switch to a slow source. A stalled native
stream falls back to window capture. Native capture has buffering latency and shares ADB resources. It does
not change the bot's decision capture path. Launching a bot from Studio passes
`-FullRes`, so the bot launcher preserves native device resolution. An already
running bot/device may still be at a reduced resolution; Studio reports the
source size and cannot reconstruct details absent from that source.

The simulator starts when Studio opens a simulator layout, runs independently,
and stops when Studio closes. The simulator is a demonstration match, not a
claim that model training is occurring. Live-log information stays in the optional operator console.

Validation: offscreen portrait composition with a real saved game frame and
running simulator; MP4 decoded as 1080x1920, H.264/yuv420p at 60 fps. This is a
layout/encoding smoke test, not a new live ladder recording. Native capture
quality and throughput still depend on the emulator and concurrent workload.
