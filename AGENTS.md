# AGENTS.md

Tauri 2 desktop app: Rust backend + React 19/TypeScript/Vite frontend that renders a bit grid ("life") on a canvas.

## Commands

- `npm install` — install frontend deps (lockfile is present).
- `npm run tauri dev` — run the real app. Use this for anything that calls `invoke`: a plain `npm run dev` browser tab cannot reach the Rust command and only shows the error banner.
- `npm run tauri dev -- -- 400` — same, with the field size as the first argument (default `144`). Only perfect squares from `4` to `4096` are accepted; anything else falls back to `144` with a stderr note.
- `npx tsc` — frontend typecheck. The root `tsconfig.json` includes only `src`; `vite.config.ts` is checked separately with `npx tsc -p tsconfig.node.json --noEmit` (`npm run build` does not check it either).
- `npm run build` — `tsc && vite build` (no lint or test step exists).
- `cargo check --manifest-path src-tauri/Cargo.toml` — Rust typecheck. `npm run tauri build` for bundles.
- `cargo test --manifest-path src-tauri/Cargo.toml --lib` — the simulation/LOD unit tests live in `lib.rs`; this is the only test suite.
- Vite is pinned to port 1420 with `strictPort`; Tauri's `devUrl` depends on it. If it is busy, stop the stale server instead of changing the port.

## Structure

- `src/App.tsx` — the entire UI: chunk texture cache, LOD crossfade, step controls, camera in refs, canvas rendering (rAF-coalesced), wheel zoom and pointer pan.
- `src-tauri/src/lib.rs` — `World` (chunked bit storage + simulation), chunk/LOD density aggregation, and the commands `world_info` / `next_step` / `render_chunks` / `flip_cell` / `reset_world` / `restart_with_size`. `src-tauri/src/main.rs` just calls `life_lib::run()`; keep the `life_lib` lib name (Windows requires it to differ from the binary name).
- `src-tauri/tauri.conf.json` — dev URL, frontend dist, bundle icons. Plugin permissions go in `src-tauri/capabilities/default.json`.

## Data contract (easy to break)

- The field side comes from the first process argument (`parse_initial_side`) and defaults to `DEFAULT_SIDE = 144`. The grid is sized once per process on purpose: changing the size restarts the executable with the new argument instead of resizing chunks while the simulation runs.
- `side` must be a perfect square in `MIN_SIDE..=MAX_SIDE` (`4..=4096`); `validate_side` enforces it in both `restart_with_size` and the startup argument.
- The world is a torus of `side` x `side` cells stored as equal chunks of `chunk_cells` x `chunk_cells`, where `chunk_cells_for(side)` is the largest divisor of `side` up to `CHUNK_MAX = 128` (144 -> 72, 400 -> 100, 512 -> 128). `Chunk { rows: [u128; 128] }` holds one row per line, only the low `chunk_cells` bits are used and the rest stay clear. `fastbit` is no longer used.
- `world_info` returns `{ side, chunk_cells, step }`. `chunk_cells` is the single source of truth; the frontend adopts it instead of hardcoding.
- `next_step(steps)` advances the simulation (`MAX_STEPS_PER_CALL = 10_000`) and returns `{ step, changed: [chunk_index] }` where `chunk_index = chunk_y * chunks_per_side + chunk_x`.
- `reset_world` empties the field, keeps the size and puts `step` back to 0; `restart_with_size(side)` spawns the executable with the new size and exits the current process (2 s grace in debug builds so Vite has time to serve the new window).
- Simulation is bit-parallel: per row, `life_step` builds the eight shifted neighbour rows and sums them with the carry-save tree in `sum8`; all edges wrap around the world. Rules: born on exactly 3, survives on 2 or 3.
- Chunk hashing: every chunk has an FNV-1a hash. A chunk is recomputed only when it or one of its 8 neighbours changed in the previous generation; otherwise it is copied verbatim. If no chunk changed at all, `advance` treats the world as a fixed point and jumps the remaining steps.
- `render_chunks(chunks: [[chunk_x, chunk_y, lod], ...])` returns raw bytes via `tauri::ipc::Response`, so JS receives an `ArrayBuffer` (no JSON, no `JSON.parse`).
- Binary layout, all `u32` little-endian: header `count, side`, then per chunk `chunk_x, chunk_y, lod, factor, cells_x, cells_y, tex_w, tex_h` (32 bytes) followed by `tex_w * tex_h` density bytes.
- LOD factors are `[1, 8, 64]` cells per texel (`LOD_FACTORS` in Rust, `lodPlan` in `App.tsx` picks the level from *physical* px/cell `cam.scale * devicePixelRatio`). Adjacent levels are crossfaded, so `lodPlan` can return two layers.
- Density bytes: `0` means an empty block, otherwise `40 + 215 * alive / total` (`MIN_DENSITY`), clamped to 255. `App.tsx` maps them through `CELL_LUT`; `0` is fully transparent because the world sheet is painted separately.
- The frontend caches one canvas per `lod:chunk_x:chunk_y` (bounded by `MAX_CACHED_CHUNKS`) and draws cached chunks with `drawImage`. Changed chunks are marked `stale` (their canvas is reused) and refetched; at most one batch is in flight, and `generationRef` discards batches computed before a flip or step.
- Camera changes never resend already-cached chunks; minimum fit scale is `0.000001`.
- A left click calls `flip_cell(index)` with `index = y * side + x`; moves of 4 px or more are a pan instead. `flip_cell` returns the affected chunk index and updates that chunk's hash.
- Space advances one step; holding it follows `speed` steps/second (`queueSteps` coalesces and `next_step` runs the batch). The panel's skip button queues `skipCount` steps.

## Notes

- User-visible strings are Russian.
- The WebView2 chrome is switched off in `harden_webview` (`setup`): no right-click menu, no F12/devtools. `tauri.conf.json` also sets `devtools: false`, and `App.tsx` prevents the page-level `contextmenu` as a fallback.
- `src-tauri/gen/schemas` and `src-tauri/target` are generated/ignored; do not edit or commit them.
- `vite.config.ts` imports `node:process` with `@ts-expect-error` because `@types/node` is not installed; removing the directive requires adding that dependency.
- No ESLint, Prettier, or test runner is configured. Do not invent test commands.
