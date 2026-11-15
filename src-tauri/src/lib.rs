use std::sync::Mutex;
use tauri::{State, ipc::Response};

/// Widest chunk a single `u128` row can hold.
const CHUNK_MAX: usize = 128;

/// Field used when the program is started without a size argument.
const DEFAULT_SIDE: usize = 144;

/// Accepted field sizes; the side must also be a perfect square.
const MIN_SIDE: usize = 4;
const MAX_SIDE: usize = 4096;

/// How many cells one texture texel covers at each LOD level.
const LOD_FACTORS: [usize; 3] = [1, 8, 64];

/// A block that contains at least one live cell never renders darker than
/// this, otherwise sparse patterns would vanish once they get aggregated.
const MIN_DENSITY: u32 = 40;

const MAX_CHUNKS_PER_REQUEST: usize = 4096;

/// Hard cap so one `next_step` call cannot block the backend for long.
const MAX_STEPS_PER_CALL: usize = 10_000;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01b3;

/// One block of cells, stored as rows of at most 128 bits.
///
/// A row is a single `u128` so horizontal neighbours are a shift plus one
/// carry bit from the left/right chunk, and the whole chunk is contiguous.
#[derive(Clone, Copy)]
struct Chunk {
    rows: [u128; CHUNK_MAX],
}

impl Chunk {
    fn empty() -> Self {
        Self {
            rows: [0; CHUNK_MAX],
        }
    }

    /// FNV-1a over the raw rows. Two equal hashes mean an identical chunk.
    fn hash(&self) -> u64 {
        let mut hash = FNV_OFFSET;
        for row in self.rows.iter() {
            hash = (hash ^ (*row as u64)).wrapping_mul(FNV_PRIME);
            hash = (hash ^ ((*row >> 64) as u64)).wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

/// The live field plus the bookkeeping that makes repeating regions skippable.
struct World {
    side: usize,
    /// Chunk side in cells; always divides `side` and is at most `CHUNK_MAX`.
    chunk_cells: usize,
    chunks_per_side: usize,
    chunks: Vec<Chunk>,
    /// Chunk hashes before the previous step.
    hash_prev: Vec<u64>,
    /// Chunk hashes of the current state.
    hash_cur: Vec<u64>,
    step: u64,
}

/// Adds three one-bit-per-cell values: `(sum, carry)`, weights 1 and 2.
#[inline]
fn add3(a: u128, b: u128, c: u128) -> (u128, u128) {
    (a ^ b ^ c, (a & b) | (a & c) | (b & c))
}

#[inline]
fn half_add(a: u128, b: u128) -> (u128, u128) {
    (a ^ b, a & b)
}

/// Bit-sliced sum of eight per-cell bits into four bit planes (0..=8).
#[inline]
fn sum8(n: [u128; 8]) -> [u128; 4] {
    let (a0, a1) = half_add(n[0], n[1]);
    let (b0, b1) = half_add(n[2], n[3]);
    let (c0, c1) = half_add(n[4], n[5]);
    let (d0, d1) = half_add(n[6], n[7]);

    let (e0, e1) = half_add(a0, b0);
    let (f0, f1) = half_add(c0, d0);
    let (g0, g1) = half_add(e0, f0);

    let (h0, h1) = add3(a1, b1, c1);
    let (i0, i1) = add3(d1, e1, f1);
    let (j0, j1) = add3(h0, i0, g1);
    let (k0, k1) = add3(h1, i1, j1);

    [g0, j0, k0, k1]
}

/// Game of Life rule for one row: `s` is the current state, the rest are the
/// eight neighbour rows already shifted into place.
#[inline]
fn life_step(
    s: u128,
    u: u128,
    d: u128,
    ul: u128,
    ur: u128,
    sl: u128,
    sr: u128,
    dl: u128,
    dr: u128,
) -> u128 {
    let [b0, b1, b2, _] = sum8([ul, u, ur, sl, sr, dl, d, dr]);
    // b1 = 1 and b2 = 0 can only mean a neighbour count of 2 or 3.
    let mid = b1 & !b2;
    (mid & b0) | (mid & !b0 & s)
}

/// Largest chunk side in cells that divides `side` and still fits a `u128`.
///
/// Keeping the world a whole number of equal chunks means every chunk is a
/// plain square: no partial edges, so the torus wrap stays a single shift.
fn chunk_cells_for(side: usize) -> usize {
    (1..=CHUNK_MAX)
        .rev()
        .find(|cells| side % cells == 0)
        .unwrap_or(1)
}

/// Checks a field size: a perfect square inside the supported range.
fn validate_side(side: usize) -> Result<(), String> {
    if !(MIN_SIDE..=MAX_SIDE).contains(&side) {
        return Err(format!(
            "размер поля должен быть от {MIN_SIDE} до {MAX_SIDE}, получено {side}"
        ));
    }
    let root = (side as f64).sqrt().round() as usize;
    if root * root != side {
        return Err(format!("{side} не является квадратом целого числа"));
    }
    Ok(())
}

/// Reads the size from the command line; anything unusable falls back to 144.
fn parse_initial_side(argument: Option<String>) -> usize {
    let Some(raw) = argument else {
        return DEFAULT_SIDE;
    };
    match raw.trim().parse::<usize>() {
        Ok(side) => match validate_side(side) {
            Ok(()) => side,
            Err(error) => {
                eprintln!("{} — используется {DEFAULT_SIDE}", error);
                DEFAULT_SIDE
            }
        },
        Err(_) => {
            eprintln!(
                "не удалось прочитать размер поля из аргумента «{}» — используется {DEFAULT_SIDE}",
                raw.trim()
            );
            DEFAULT_SIDE
        }
    }
}

impl World {
    fn new(side: usize) -> Self {
        let chunk_cells = chunk_cells_for(side);
        let chunks_per_side = side / chunk_cells;
        let total = chunks_per_side * chunks_per_side;
        let chunks = vec![Chunk::empty(); total];
        let hash_cur = chunks.iter().map(Chunk::hash).collect();

        Self {
            side,
            chunk_cells,
            chunks_per_side,
            chunks,
            // Mismatching the current hashes forces a full first step.
            hash_prev: vec![0; total],
            hash_cur,
            step: 0,
        }
    }

    /// Bits of a chunk row that belong to the world; the rest stay clear.
    #[inline]
    fn row_mask(&self) -> u128 {
        if self.chunk_cells >= CHUNK_MAX {
            u128::MAX
        } else {
            (1u128 << self.chunk_cells) - 1
        }
    }

    #[inline]
    fn index(&self, chunk_x: usize, chunk_y: usize) -> usize {
        (chunk_y % self.chunks_per_side) * self.chunks_per_side + (chunk_x % self.chunks_per_side)
    }

    fn cell(&self, x: usize, y: usize) -> bool {
        let chunk = &self.chunks[self.index(x / self.chunk_cells, y / self.chunk_cells)];
        (chunk.rows[y % self.chunk_cells] >> (x % self.chunk_cells)) & 1 != 0
    }

    fn set_cell(&mut self, x: usize, y: usize, alive: bool) {
        let index = self.index(x / self.chunk_cells, y / self.chunk_cells);
        let bit = 1u128 << (x % self.chunk_cells);
        let row = &mut self.chunks[index].rows[y % self.chunk_cells];
        if alive {
            *row |= bit;
        } else {
            *row &= !bit;
        }
        self.hash_cur[index] = self.chunks[index].hash();
    }

    /// Recomputes the one chunk that contains `(x, y)`.
    fn generate_chunk(&self, chunk_x: usize, chunk_y: usize) -> Chunk {
        let per_side = self.chunks_per_side;
        let cells = self.chunk_cells;
        let mask = self.row_mask();
        // Shifting left by `cells` would leave the chunk, so the carry bit from
        // the neighbour arrives at the highest cell instead of bit 127.
        let carry = cells - 1;

        let self_chunk = &self.chunks[self.index(chunk_x, chunk_y)];
        let left = &self.chunks[self.index(chunk_x + per_side - 1, chunk_y)];
        let right = &self.chunks[self.index(chunk_x + 1, chunk_y)];
        let up = &self.chunks[self.index(chunk_x, chunk_y + per_side - 1)];
        let down = &self.chunks[self.index(chunk_x, chunk_y + 1)];

        let mut out = Chunk::empty();
        for y in 0..cells {
            let (up_chunk, up_y, down_chunk, down_y) = if y == 0 {
                (up, cells - 1, self_chunk, 1)
            } else if y == cells - 1 {
                (self_chunk, cells - 2, down, 0)
            } else {
                (self_chunk, y - 1, self_chunk, y + 1)
            };

            let s = self_chunk.rows[y];
            let u = up_chunk.rows[up_y];
            let d = down_chunk.rows[down_y];
            let ul = ((u << 1) | (left.rows[up_y] >> carry)) & mask;
            let ur = ((u >> 1) | ((right.rows[up_y] & 1) << carry)) & mask;
            let sl = ((s << 1) | (left.rows[y] >> carry)) & mask;
            let sr = ((s >> 1) | ((right.rows[y] & 1) << carry)) & mask;
            let dl = ((d << 1) | (left.rows[down_y] >> carry)) & mask;
            let dr = ((d >> 1) | ((right.rows[down_y] & 1) << carry)) & mask;

            out.rows[y] = life_step(s, u, d, ul, ur, sl, sr, dl, dr) & mask;
        }
        out
    }

    /// Advances one generation and returns the chunks whose bits changed.
    fn step_once(&mut self) -> Vec<u32> {
        let per_side = self.chunks_per_side;
        let total = per_side * per_side;

        let mut changed = vec![false; total];
        let mut any_changed = false;
        for index in 0..total {
            if self.hash_cur[index] != self.hash_prev[index] {
                changed[index] = true;
                any_changed = true;
            }
        }

        self.step += 1;
        if !any_changed {
            // Nothing moved in the last generation, so this is a fixed point.
            return Vec::new();
        }

        // A chunk only needs work if one of the 3x3 chunks around it changed;
        // everything else is copied verbatim, which is what makes large empty
        // regions free.
        let mut busy = vec![false; total];
        for chunk_y in 0..per_side {
            for chunk_x in 0..per_side {
                if !changed[chunk_y * per_side + chunk_x] {
                    continue;
                }
                for dy in [per_side - 1, 0, 1] {
                    for dx in [per_side - 1, 0, 1] {
                        busy[self.index(chunk_x + dx, chunk_y + dy)] = true;
                    }
                }
            }
        }

        let mut next = Vec::with_capacity(total);
        let mut next_hash = Vec::with_capacity(total);
        let mut delta = Vec::new();

        for chunk_y in 0..per_side {
            for chunk_x in 0..per_side {
                let index = chunk_y * per_side + chunk_x;
                if !busy[index] {
                    next.push(self.chunks[index]);
                    next_hash.push(self.hash_cur[index]);
                    continue;
                }
                let chunk = self.generate_chunk(chunk_x, chunk_y);
                let hash = chunk.hash();
                if hash != self.hash_cur[index] {
                    delta.push(index as u32);
                }
                next.push(chunk);
                next_hash.push(hash);
            }
        }

        self.chunks = next;
        self.hash_prev = std::mem::take(&mut self.hash_cur);
        self.hash_cur = next_hash;
        delta
    }

    /// Runs `steps` generations, stopping early once the world is static.
    fn advance(&mut self, steps: usize) -> Vec<u32> {
        let mut changed: Vec<u32> = Vec::new();
        let mut done = 0;
        while done < steps {
            let delta = self.step_once();
            done += 1;
            if delta.is_empty() {
                // A fixed point: the remaining steps cannot change anything.
                self.step += (steps - done) as u64;
                break;
            }
            for index in delta {
                if !changed.contains(&index) {
                    changed.push(index);
                }
            }
        }
        changed
    }

    fn info(&self) -> WorldInfo {
        WorldInfo {
            side: self.side,
            chunk_cells: self.chunk_cells,
            step: self.step,
        }
    }
}

/// One rendered chunk: `pixels` holds one density byte per texel.
struct ChunkData {
    cells_x: usize,
    cells_y: usize,
    tex_w: usize,
    tex_h: usize,
    pixels: Vec<u8>,
}

/// Aggregates a chunk into density texels; `factor` cells per texel per axis.
fn chunk_density(
    world: &World,
    chunk_x: usize,
    chunk_y: usize,
    factor: usize,
) -> Option<ChunkData> {
    if chunk_x >= world.chunks_per_side || chunk_y >= world.chunks_per_side {
        return None;
    }
    let chunk = &world.chunks[chunk_y * world.chunks_per_side + chunk_x];
    let cells = world.chunk_cells;

    // A factor that does not divide the chunk side still spreads evenly: the
    // texel count rounds up and the edge texels simply cover fewer cells.
    let tex_w = (cells + factor - 1) / factor;
    let tex_h = tex_w;
    let mut pixels = vec![0u8; tex_w * tex_h];

    for ty in 0..tex_h {
        let y0 = ty * factor;
        let y1 = (y0 + factor).min(cells);
        for tx in 0..tex_w {
            let x0 = tx * factor;
            let x1 = (x0 + factor).min(cells);
            let width = x1 - x0;
            let mask = if width >= CHUNK_MAX {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };

            let mut alive = 0u32;
            for y in y0..y1 {
                alive += ((chunk.rows[y] >> x0) & mask).count_ones();
            }

            let total = (width * (y1 - y0)) as u32;
            pixels[ty * tex_w + tx] = if alive == 0 {
                0
            } else {
                (MIN_DENSITY + (255 - MIN_DENSITY) * alive / total).min(255) as u8
            };
        }
    }

    Some(ChunkData {
        cells_x: cells,
        cells_y: cells,
        tex_w,
        tex_h,
        pixels,
    })
}

/// The grid geometry the frontend needs before it can ask for chunks.
#[derive(serde::Serialize)]
struct WorldInfo {
    side: usize,
    chunk_cells: usize,
    step: u64,
}

#[derive(serde::Serialize)]
struct StepResult {
    step: u64,
    changed: Vec<u32>,
}

#[tauri::command]
fn world_info(state: State<AppState>) -> Result<WorldInfo, String> {
    let world = state
        .world
        .lock()
        .map_err(|_| "grid state is poisoned".to_string())?;

    Ok(world.info())
}

/// Advances the simulation and reports which chunks now need new textures.
#[tauri::command]
fn next_step(steps: usize, state: State<AppState>) -> Result<StepResult, String> {
    let mut world = state
        .world
        .lock()
        .map_err(|_| "grid state is poisoned".to_string())?;

    let changed = world.advance(steps.min(MAX_STEPS_PER_CALL));

    Ok(StepResult {
        step: world.step,
        changed,
    })
}

/// Empties the field and puts the step counter back to zero.
#[tauri::command]
fn reset_world(state: State<AppState>) -> Result<WorldInfo, String> {
    let mut world = state
        .world
        .lock()
        .map_err(|_| "grid state is poisoned".to_string())?;

    *world = World::new(world.side);
    Ok(world.info())
}

/// Restarts the executable with a new field size.
///
/// The grid is sized when the process starts, so changing it means a clean
/// launch rather than resizing chunks under the running simulation.
#[tauri::command]
fn restart_with_size(side: usize, app: tauri::AppHandle) -> Result<(), String> {
    validate_side(side)?;

    let exe = std::env::current_exe()
        .map_err(|error| format!("не удалось найти исполняемый файл: {error}"))?;
    std::process::Command::new(exe)
        .arg(side.to_string())
        .spawn()
        .map_err(|error| format!("не удалось перезапустить приложение: {error}"))?;

    // In a dev build the frontend comes from Vite, and `tauri dev` stops that
    // server as soon as this process ends; give the new window time to load.
    if cfg!(debug_assertions) {
        std::thread::sleep(std::time::Duration::from_millis(2_000));
    }

    app.exit(0);
    Ok(())
}

/// Renders the requested chunks into one binary payload.
///
/// Layout (all integers are little-endian `u32`):
///
/// ```text
/// header:  count, side
/// chunk:   chunk_x, chunk_y, lod, factor, cells_x, cells_y, tex_w, tex_h
///          followed by tex_w * tex_h density bytes
/// ```
#[tauri::command]
fn render_chunks(chunks: Vec<[u32; 3]>, state: State<AppState>) -> Result<Response, String> {
    if chunks.len() > MAX_CHUNKS_PER_REQUEST {
        return Err(format!(
            "too many chunks requested: {} > {MAX_CHUNKS_PER_REQUEST}",
            chunks.len()
        ));
    }

    let world = state
        .world
        .lock()
        .map_err(|_| "grid state is poisoned".to_string())?;

    let mut out = Vec::with_capacity(8 + chunks.len() * 64);
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    out.extend_from_slice(&(world.side as u32).to_le_bytes());

    for [chunk_x, chunk_y, lod] in chunks {
        let lod = (lod as usize).min(LOD_FACTORS.len() - 1);
        let factor = LOD_FACTORS[lod];
        let chunk = chunk_density(&world, chunk_x as usize, chunk_y as usize, factor);

        let empty = ChunkData {
            cells_x: 0,
            cells_y: 0,
            tex_w: 0,
            tex_h: 0,
            pixels: Vec::new(),
        };
        let chunk = chunk.as_ref().unwrap_or(&empty);

        for value in [
            chunk_x,
            chunk_y,
            lod as u32,
            factor as u32,
            chunk.cells_x as u32,
            chunk.cells_y as u32,
            chunk.tex_w as u32,
            chunk.tex_h as u32,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&chunk.pixels);
    }

    Ok(Response::new(out))
}

/// Flips one cell and returns the chunk that must be re-rendered.
#[tauri::command]
fn flip_cell(index: usize, state: State<AppState>) -> Result<u32, String> {
    let mut world = state
        .world
        .lock()
        .map_err(|_| "grid state is poisoned".to_string())?;

    let side = world.side;
    if index >= side * side {
        return Err(format!("cell {index} is outside the grid"));
    }

    let x = index % side;
    let y = index / side;
    let alive = !world.cell(x, y);
    world.set_cell(x, y, alive);

    Ok(((y / world.chunk_cells) * world.chunks_per_side + x / world.chunk_cells) as u32)
}

struct AppState {
    world: Mutex<World>,
}

/// Turns off the WebView2 chrome: the right-click menu (share/save/reload) and
/// the browser shortcuts, including F12.
#[cfg(windows)]
fn harden_webview(window: &tauri::WebviewWindow) {
    let _ = window.with_webview(|webview| unsafe {
        let Ok(core) = webview.controller().CoreWebView2() else {
            return;
        };
        let Ok(settings) = core.Settings() else {
            return;
        };
        let _ = settings.SetAreDefaultContextMenusEnabled(false);
        let _ = settings.SetAreDevToolsEnabled(false);
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let side = parse_initial_side(std::env::args().nth(1));

    tauri::Builder::default()
        .manage(AppState {
            world: Mutex::new(World::new(side)),
        })
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            #[cfg(windows)]
            {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    harden_webview(&window);
                }
            }
            let _ = app;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            world_info,
            next_step,
            render_chunks,
            flip_cell,
            reset_world,
            restart_with_size
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn life_rules_per_bit() {
        let cell = 1u128 << 5;

        // Two neighbours: a live cell survives, a dead one stays dead.
        assert_eq!(life_step(cell, 0, 0, 0, 0, cell, cell, 0, 0) & cell, cell);
        assert_eq!(life_step(0, 0, 0, 0, 0, cell, cell, 0, 0) & cell, 0);

        // Three neighbours: a dead cell is born.
        assert_eq!(life_step(0, cell, cell, 0, 0, cell, 0, 0, 0) & cell, cell);

        // Four neighbours: a live cell dies.
        assert_eq!(
            life_step(cell, cell, cell, 0, 0, cell, cell, 0, 0) & cell,
            0
        );
    }

    #[test]
    fn default_field_is_144_and_splits_into_whole_chunks() {
        let world = World::new(DEFAULT_SIDE);

        assert_eq!(world.side, 144);
        assert_eq!(world.chunk_cells, 72);
        assert_eq!(world.chunks_per_side, 2);
        assert_eq!(world.chunks.len(), 4);
    }

    #[test]
    fn any_square_side_divides_into_whole_chunks() {
        for side in [4, 100, 144, 400, 512, 900, 1024, 2500] {
            let world = World::new(side);
            assert_eq!(world.side, side);
            assert!(world.chunk_cells <= CHUNK_MAX, "side {side} is too wide");
            assert_eq!(world.side % world.chunk_cells, 0);
            assert_eq!(
                world.chunks_per_side * world.chunk_cells,
                world.side,
                "chunks must cover the whole field for side {side}"
            );
            assert_eq!(world.chunks.len(), world.chunks_per_side.pow(2));
        }
    }

    #[test]
    fn only_squares_inside_the_range_are_accepted() {
        assert!(validate_side(144).is_ok());
        assert!(validate_side(4).is_ok());
        assert!(validate_side(4096).is_ok());
        assert!(validate_side(145).is_err());
        assert!(validate_side(3).is_err());
        assert!(validate_side(4097).is_err());
        assert!(validate_side(0).is_err());
    }

    #[test]
    fn the_size_argument_wins_over_the_default() {
        assert_eq!(parse_initial_side(None), DEFAULT_SIDE);
        assert_eq!(parse_initial_side(Some("400".into())), 400);
        assert_eq!(parse_initial_side(Some(" 144 ".into())), 144);
        // Invalid arguments fall back instead of crashing on startup.
        assert_eq!(parse_initial_side(Some("145".into())), DEFAULT_SIDE);
        assert_eq!(parse_initial_side(Some("-4".into())), DEFAULT_SIDE);
        assert_eq!(parse_initial_side(Some("abc".into())), DEFAULT_SIDE);
    }

    #[test]
    fn blinkers_wrap_around_the_world() {
        let mut world = World::new(512);
        // Vertical blinker on the seam between the last and first column.
        world.set_cell(world.side - 1, 1, true);
        world.set_cell(0, 1, true);
        world.set_cell(1, 1, true);
        world.step_once();

        // It must blink into the vertical orientation on the same columns.
        assert!(world.cell(0, 0));
        assert!(world.cell(0, 1));
        assert!(world.cell(0, 2));
        assert!(!world.cell(1, 1));
    }

    #[test]
    fn blinkers_wrap_inside_72_cell_chunks() {
        // 144 is the default field; its chunks are 72 cells wide, so the shift
        // carry lands on bit 71 instead of bit 127.
        let mut world = World::new(144);
        world.set_cell(world.side - 1, 1, true);
        world.set_cell(0, 1, true);
        world.set_cell(1, 1, true);
        world.step_once();

        assert!(world.cell(0, 0));
        assert!(world.cell(0, 1));
        assert!(world.cell(0, 2));
        assert!(!world.cell(1, 1));
    }

    #[test]
    fn glider_moves_one_cell_per_four_steps() {
        let mut world = World::new(512);
        for (dx, dy) in [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)] {
            world.set_cell(10 + dx, 200 + dy, true);
        }

        world.advance(4);

        let mut live = 0;
        for y in 195..210 {
            for x in 5..20 {
                if world.cell(x, y) {
                    live += 1;
                }
            }
        }
        assert_eq!(live, 5);
        for (dx, dy) in [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)] {
            assert!(world.cell(11 + dx, 201 + dy));
        }
        assert_eq!(world.step, 4);
    }

    #[test]
    fn blocks_are_still_lifes() {
        let mut world = World::new(144);
        let side = world.side;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            world.set_cell(side - 2 + dx, 5 + dy, true);
        }
        world.advance(1);
        world.advance(25);

        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            assert!(world.cell(side - 2 + dx, 5 + dy));
        }
        // A still life stops generating work after the first generation.
        assert!(world.advance(1).is_empty());
        assert_eq!(world.step, 27);
    }

    #[test]
    fn density_aggregates_each_lod() {
        let mut world = World::new(144);
        let cells = world.chunk_cells;
        // One live cell in chunk (1, 1): it sits in the first texel of every LOD.
        world.set_cell(cells, cells, true);

        for (lod, factor) in LOD_FACTORS.iter().copied().enumerate() {
            let data = chunk_density(&world, 1, 1, factor).expect("chunk exists");
            let tex_w = (cells + factor - 1) / factor;
            assert_eq!(data.tex_w, tex_w);
            assert_eq!(data.tex_h, tex_w);
            assert_eq!(data.cells_x, cells);
            assert_eq!(data.cells_y, cells);
            assert_eq!(data.pixels.len(), tex_w * tex_w);

            // A single cell only brightens its own texel; the rest stay clear.
            let expected = MIN_DENSITY + (255 - MIN_DENSITY) / (factor * factor) as u32;
            assert_eq!(
                data.pixels[0], expected as u8,
                "one alive cell sets the minimum visible density at lod {lod}"
            );
            assert_eq!(data.pixels[1], 0, "empty texels stay transparent");
        }

        // Out-of-range chunks are reported as absent instead of panicking.
        assert!(chunk_density(&world, world.chunks_per_side, 0, 1).is_none());

        // A full block saturates at 255.
        for y in 0..cells {
            for x in 0..cells {
                world.set_cell(cells + x, cells + y, true);
            }
        }
        let data = chunk_density(&world, 1, 1, 8).expect("chunk exists");
        assert!(data.pixels.iter().all(|byte| *byte == 255));
    }

    #[test]
    fn static_regions_are_skipped() {
        let mut world = World::new(144);
        world.set_cell(10, 10, true);
        let changed = world.advance(1);
        assert_eq!(changed, vec![0]);
        // The single cell dies, then nothing ever changes again.
        let changed = world.advance(50);
        assert!(changed.is_empty());
        assert_eq!(world.step, 51);
    }
}
