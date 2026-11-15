import { invoke } from "@tauri-apps/api/core";
import { useEffect, useRef, useState } from "react";
import "./App.css";

/** Must match `CHUNK_MAX` in `src-tauri/src/lib.rs`; `world_info` confirms it. */
const DEFAULT_CHUNK_CELLS = 128;

/** Field sizes the backend accepts; the side must be a perfect square. */
const MIN_SIDE = 4;
const MAX_SIDE = 4096;

/** Extra cells around the viewport that are fetched ahead of the pan. */
const PREFETCH_CELLS = 32;

/** Camera/DOM churn is coalesced into at most one chunk request per delay. */
const FETCH_DELAY_MS = 40;

/** Dead cells stay faintly visible, exactly like the old per-cell renderer. */
const SHEET_ALPHA = 28 / 255;

/** Chunk textures are ~64 KB each, so the working set is kept bounded. */
const MAX_CACHED_CHUNKS = 192;

interface Camera {
  x: number;
  y: number;
  scale: number;
}

interface Drag {
  x: number;
  y: number;
  camX: number;
  camY: number;
}

interface WorldInfo {
  side: number;
  chunk_cells: number;
  step: number;
}

interface StepResult {
  step: number;
  changed: number[];
}

/** One cached chunk: a canvas that `drawImage` can blit without touching bits. */
interface ChunkTexture {
  canvas: HTMLCanvasElement;
  cellsX: number;
  cellsY: number;
  /** Set when the simulation changed the chunk; it must be fetched again. */
  stale: boolean;
}

interface LodLayer {
  lod: number;
  alpha: number;
}

/** Builds a 256-entry pack of RGBA colors, indexed by a density byte. */
function buildColorLut(shade: (density: number) => [number, number, number, number]) {
  const lut = new Uint32Array(256);
  const word = new Uint32Array(1);
  const bytes = new Uint8ClampedArray(word.buffer);
  for (let density = 0; density < 256; density += 1) {
    const [r, g, b, a] = shade(density / 255);
    bytes[0] = r;
    bytes[1] = g;
    bytes[2] = b;
    bytes[3] = a;
    lut[density] = word[0];
  }
  return lut;
}

/**
 * `0` density is fully transparent (the world sheet provides the faint
 * backdrop), `255` is exactly the old lit cell. The square root keeps a lone
 * live cell inside a large aggregated block visible.
 */
const CELL_LUT = buildColorLut((density) => {
  const t = Math.sqrt(density);
  return [255, Math.round(255 - 5 * t), Math.round(255 - 20 * t), Math.round(240 * t)];
});

/**
 * Maps physical pixels per cell onto a continuous LOD position, so zooming
 * fades one level into the next instead of popping.
 *
 * LOD 0 is one cell per texel, LOD 1 aggregates 8x8 cells and LOD 2 aggregates
 * 64x64 cells, which is the pure density view.
 */
function lodPlan(pxPerCell: number): LodLayer[] {
  const octave = Math.log2(Math.max(pxPerCell, 1e-9));

  let level: number;
  if (octave >= 0.5) level = 0;
  else if (octave >= -2.5) level = (0.5 - octave) / 3;
  else if (octave >= -3.5) level = 1 + (-2.5 - octave);
  else level = 2;
  level = Math.min(2, Math.max(0, level));

  const base = Math.floor(level);
  const coarse = Math.min(2, base + 1);
  const alpha = level - base;
  const layers: LodLayer[] = [{ lod: base, alpha: 1 }];
  if (coarse !== base && alpha > 0.01) layers.push({ lod: coarse, alpha });
  return layers;
}

function chunkKey(lod: number, chunkX: number, chunkY: number) {
  return `${lod}:${chunkX}:${chunkY}`;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

/** Mirrors the backend rule: a perfect square inside the supported range. */
function parseFieldSide(text: string): number | null {
  const side = Number(text.trim());
  if (!Number.isInteger(side) || side < MIN_SIDE || side > MAX_SIDE) return null;
  const root = Math.round(Math.sqrt(side));
  return root * root === side ? side : null;
}

/** Normalizes the raw IPC payload, which is an `ArrayBuffer` on desktop. */
function toArrayBuffer(input: unknown): ArrayBuffer {
  if (input instanceof ArrayBuffer) return input;
  if (ArrayBuffer.isView(input)) {
    const view = input as ArrayBufferView;
    return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength) as ArrayBuffer;
  }
  if (Array.isArray(input)) return new Uint8Array(input).buffer;
  throw new Error("unexpected chunk payload");
}

/** Turns density bytes into a canvas texture; done once per loaded chunk. */
function makeChunkTexture(
  cellsX: number,
  cellsY: number,
  texW: number,
  texH: number,
  density: Uint8Array,
  reuse?: HTMLCanvasElement,
): ChunkTexture | null {
  if (texW === 0 || texH === 0 || cellsX === 0 || cellsY === 0) return null;

  const canvas = reuse ?? document.createElement("canvas");
  if (canvas.width !== texW) canvas.width = texW;
  if (canvas.height !== texH) canvas.height = texH;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const image = ctx.createImageData(texW, texH);
  const pixels = new Uint32Array(image.data.buffer);
  const count = Math.min(density.length, pixels.length);
  for (let i = 0; i < count; i += 1) {
    pixels[i] = CELL_LUT[density[i]];
  }
  ctx.putImageData(image, 0, 0);

  return { canvas, cellsX, cellsY, stale: false };
}

export default function App() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glowLayerRef = useRef<HTMLCanvasElement | null>(null);

  const sideRef = useRef(0);
  const chunkCellsRef = useRef(DEFAULT_CHUNK_CELLS);
  const chunkCacheRef = useRef(new Map<string, ChunkTexture>());
  const inflightRef = useRef(false);
  const queuedRef = useRef(false);
  const generationRef = useRef(0);
  const fetchTimerRef = useRef(0);
  const stepBusyRef = useRef(false);
  const pendingStepsRef = useRef(0);
  const stepEpochRef = useRef(0);
  const flushStepsRef = useRef<() => void>(() => {});
  const sizeInitialisedRef = useRef(false);
  const busyRef = useRef(false);

  const camRef = useRef<Camera>({ x: 0, y: 0, scale: 1 });
  const dragRef = useRef<Drag | null>(null);
  const movedRef = useRef(false);
  const frameRef = useRef(0);
  const redrawRef = useRef<() => void>(() => {});
  const requestDrawRef = useRef<() => void>(() => {});
  const fitRef = useRef<() => void>(() => {});
  const pumpRef = useRef<() => void>(() => {});

  const [dragging, setDragging] = useState(false);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [message, setMessage] = useState("Загрузка сетки…");
  const [step, setStep] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [skipCount, setSkipCount] = useState(100);
  const [sizeText, setSizeText] = useState("144");
  const [busy, setBusy] = useState(false);

  // The hold-space loop reads these from refs so it never uses a stale value.
  const speedRef = useRef(speed);
  speedRef.current = speed;

  fitRef.current = () => {
    const viewport = viewportRef.current;
    const side = sideRef.current;
    if (!viewport || side === 0) return;

    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    if (width === 0 || height === 0) return;

    const margin = Math.min(48, width * 0.08, height * 0.08);
    const scale = Math.max(
      0.000001,
      Math.min((width - margin * 2) / side, (height - margin * 2) / side),
    );
    camRef.current = {
      scale,
      x: (width - side * scale) / 2,
      y: (height - side * scale) / 2,
    };
  };

  redrawRef.current = () => {
    const viewport = viewportRef.current;
    const canvas = canvasRef.current;
    if (!viewport || !canvas) return;

    const cssW = viewport.clientWidth;
    const cssH = viewport.clientHeight;
    if (cssW === 0 || cssH === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const bitmapW = Math.round(cssW * dpr);
    const bitmapH = Math.round(cssH * dpr);
    if (canvas.width !== bitmapW || canvas.height !== bitmapH) {
      canvas.width = bitmapW;
      canvas.height = bitmapH;
    }

    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cssW, cssH);

    const side = sideRef.current;
    const cam = camRef.current;
    if (side === 0 || cam.scale <= 0) return;

    // Faint sheet under the whole grid, so transparent dead cells still read
    // as a surface instead of the page background.
    const worldSize = side * cam.scale;
    const sheetLeft = Math.max(0, cam.x);
    const sheetTop = Math.max(0, cam.y);
    const sheetRight = Math.min(cssW, cam.x + worldSize);
    const sheetBottom = Math.min(cssH, cam.y + worldSize);
    if (sheetRight > sheetLeft && sheetBottom > sheetTop) {
      ctx.fillStyle = `rgba(255, 255, 255, ${SHEET_ALPHA})`;
      ctx.fillRect(sheetLeft, sheetTop, sheetRight - sheetLeft, sheetBottom - sheetTop);
    }

    const c0 = Math.max(0, Math.floor(-cam.x / cam.scale));
    const r0 = Math.max(0, Math.floor(-cam.y / cam.scale));
    const c1 = Math.min(side, Math.ceil((cssW - cam.x) / cam.scale));
    const r1 = Math.min(side, Math.ceil((cssH - cam.y) / cam.scale));
    if (c1 <= c0 || r1 <= r0) return;

    const chunkCells = chunkCellsRef.current;
    const chunkX0 = Math.floor(c0 / chunkCells);
    const chunkY0 = Math.floor(r0 / chunkCells);
    const chunkX1 = Math.ceil(c1 / chunkCells);
    const chunkY1 = Math.ceil(r1 / chunkCells);

    const pxPerCell = cam.scale * dpr;
    const layers = lodPlan(pxPerCell);
    const smoothing = pxPerCell < 1;
    const cache = chunkCacheRef.current;

    const drawLayer = (
      target: CanvasRenderingContext2D,
      lod: number,
      alpha: number,
      smoothingEnabled: boolean,
    ) => {
      target.globalAlpha = alpha;
      target.imageSmoothingEnabled = smoothingEnabled;
      for (let chunkY = chunkY0; chunkY < chunkY1; chunkY += 1) {
        const worldY = chunkY * chunkCells * cam.scale;
        for (let chunkX = chunkX0; chunkX < chunkX1; chunkX += 1) {
          const texture = cache.get(chunkKey(lod, chunkX, chunkY));
          if (!texture) continue;
          target.drawImage(
            texture.canvas,
            chunkX * chunkCells * cam.scale + cam.x,
            worldY + cam.y,
            texture.cellsX * cam.scale,
            texture.cellsY * cam.scale,
          );
        }
      }
    };

    // The glow is accumulated into one viewport-sized layer and blurred once,
    // so the cost per frame stays close to one fullscreen blit no matter how
    // many chunks are visible.
    let glowLayer = glowLayerRef.current;
    if (!glowLayer) {
      glowLayer = document.createElement("canvas");
      glowLayerRef.current = glowLayer;
    }
    if (glowLayer.width !== bitmapW || glowLayer.height !== bitmapH) {
      glowLayer.width = bitmapW;
      glowLayer.height = bitmapH;
    }
    const glowCtx = glowLayer.getContext("2d", { alpha: true });
    if (glowCtx) {
      glowCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      glowCtx.clearRect(0, 0, cssW, cssH);
      glowCtx.globalCompositeOperation = "lighter";
      for (const layer of layers) {
        drawLayer(glowCtx, layer.lod, layer.alpha, smoothing);
      }
      glowCtx.globalAlpha = 1;

      const glow = Math.min(18, Math.max(2, cam.scale * 0.35));
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.filter = `blur(${glow}px)`;
      ctx.globalAlpha = 0.65;
      ctx.drawImage(glowLayer, 0, 0, cssW, cssH);
      ctx.restore();
    }

    for (const layer of layers) {
      ctx.save();
      drawLayer(ctx, layer.lod, layer.alpha, smoothing);
      ctx.restore();
    }

    // Faint grid so transparent black cells stay readable.
    if (layers[0].lod === 0 && cam.scale >= 8 && c1 - c0 <= 500 && r1 - r0 <= 500) {
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx.lineWidth = 1;
      const left = c0 * cam.scale + cam.x;
      const top = r0 * cam.scale + cam.y;
      const right = c1 * cam.scale + cam.x;
      const bottom = r1 * cam.scale + cam.y;
      for (let col = c0; col <= c1; col += 1) {
        const x = Math.round(col * cam.scale + cam.x) + 0.5;
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
      }
      for (let row = r0; row <= r1; row += 1) {
        const y = Math.round(row * cam.scale + cam.y) + 0.5;
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
      }
      ctx.stroke();
    }
  };

  requestDrawRef.current = () => {
    if (frameRef.current !== 0) return;
    frameRef.current = window.requestAnimationFrame(() => {
      frameRef.current = 0;
      redrawRef.current();
    });
  };

  /** Stores a binary `render_chunks` payload in the texture cache. */
  const applyChunks = (buffer: ArrayBuffer) => {
    const view = new DataView(buffer);
    if (view.byteLength < 8) throw new Error("invalid chunk response");

    const count = view.getUint32(0, true);
    const side = view.getUint32(4, true);
    if (side > 0) sideRef.current = side;

    const cache = chunkCacheRef.current;
    let offset = 8;

    for (let i = 0; i < count; i += 1) {
      if (offset + 32 > view.byteLength) throw new Error("truncated chunk header");
      const chunkX = view.getUint32(offset, true);
      const chunkY = view.getUint32(offset + 4, true);
      const lod = view.getUint32(offset + 8, true);
      // offset + 12 is the texel factor, which only the backend needs.
      const cellsX = view.getUint32(offset + 16, true);
      const cellsY = view.getUint32(offset + 20, true);
      const texW = view.getUint32(offset + 24, true);
      const texH = view.getUint32(offset + 28, true);
      offset += 32;

      const size = texW * texH;
      if (offset + size > view.byteLength) throw new Error("truncated chunk pixels");
      const density = new Uint8Array(buffer, offset, size);
      offset += size;

      const key = chunkKey(lod, chunkX, chunkY);
      const previous = cache.get(key);
      const texture = makeChunkTexture(
        cellsX,
        cellsY,
        texW,
        texH,
        density,
        previous?.canvas,
      );
      if (texture) cache.set(key, texture);
      else cache.delete(key);
    }

    // Map keeps insertion order, so the oldest chunks are evicted first.
    while (cache.size > MAX_CACHED_CHUNKS) {
      const oldest = cache.keys().next();
      if (oldest.done) break;
      cache.delete(oldest.value);
    }
  };

  /** Fetches only the chunks the current camera needs and that are not cached. */
  pumpRef.current = () => {
    const viewport = viewportRef.current;
    const side = sideRef.current;
    const cam = camRef.current;
    if (!viewport || side === 0 || cam.scale <= 0) return;

    if (inflightRef.current) {
      queuedRef.current = true;
      return;
    }

    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    if (width === 0 || height === 0) return;

    const minX = Math.max(0, Math.floor(-cam.x / cam.scale) - PREFETCH_CELLS);
    const maxX = Math.min(side, Math.ceil((width - cam.x) / cam.scale) + PREFETCH_CELLS);
    const minY = Math.max(0, Math.floor(-cam.y / cam.scale) - PREFETCH_CELLS);
    const maxY = Math.min(side, Math.ceil((height - cam.y) / cam.scale) + PREFETCH_CELLS);
    if (maxX <= minX || maxY <= minY) return;

    const chunkCells = chunkCellsRef.current;
    const chunkX0 = Math.floor(minX / chunkCells);
    const chunkY0 = Math.floor(minY / chunkCells);
    const chunkX1 = Math.ceil(maxX / chunkCells);
    const chunkY1 = Math.ceil(maxY / chunkCells);

    const cache = chunkCacheRef.current;
    const layers = lodPlan(cam.scale * (window.devicePixelRatio || 1));
    const requests: [number, number, number][] = [];

    for (const layer of layers) {
      for (let chunkY = chunkY0; chunkY < chunkY1; chunkY += 1) {
        for (let chunkX = chunkX0; chunkX < chunkX1; chunkX += 1) {
          const key = chunkKey(layer.lod, chunkX, chunkY);
          const cached = cache.get(key);
          if (cached && !cached.stale) continue;
          requests.push([chunkX, chunkY, layer.lod]);
        }
      }
    }
    if (requests.length === 0) return;

    const generation = generationRef.current;
    inflightRef.current = true;

    void invoke<unknown>("render_chunks", { chunks: requests })
      .then((payload) => {
        if (generation !== generationRef.current) return;
        applyChunks(toArrayBuffer(payload));
        setStatus("ready");
        setMessage("");
      })
      .catch((error) => {
        if (generation !== generationRef.current) return;
        const detail = error instanceof Error ? error.message : String(error);
        setStatus("error");
        setMessage(detail ? `Не удалось получить чанки: ${detail}` : "Не удалось получить чанки");
      })
      .finally(() => {
        inflightRef.current = false;
        if (generation !== generationRef.current || queuedRef.current) {
          queuedRef.current = false;
          pumpRef.current();
        }
        requestDrawRef.current();
      });
  };

  const schedulePump = () => {
    if (fetchTimerRef.current !== 0) return;
    fetchTimerRef.current = window.setTimeout(() => {
      fetchTimerRef.current = 0;
      pumpRef.current();
    }, FETCH_DELAY_MS);
  };

  useEffect(() => {
    let cancelled = false;

    void invoke<WorldInfo>("world_info")
      .then((info) => {
        if (cancelled) return;
        applyWorld(info, true);
      })
      .catch((error) => {
        if (cancelled) return;
        const detail = error instanceof Error ? error.message : String(error);
        setStatus("error");
        setMessage(detail ? `Не удалось получить сетку: ${detail}` : "Не удалось получить сетку");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  // The right-click menu is switched off in Rust for WebView2; this keeps the
  // page-level menu from appearing elsewhere too.
  useEffect(() => {
    const onContextMenu = (event: MouseEvent) => event.preventDefault();
    document.addEventListener("contextmenu", onContextMenu);
    return () => document.removeEventListener("contextmenu", onContextMenu);
  }, []);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const redraw = () => requestDrawRef.current();
    const onWheel = (event: WheelEvent) => {
      const side = sideRef.current;
      if (side === 0) return;
      // The panel keeps its own scrolling/typing behaviour.
      if (event.target instanceof Element && event.target.closest(".panel")) return;
      event.preventDefault();

      const rect = viewport.getBoundingClientRect();
      const pointerX = event.clientX - rect.left;
      const pointerY = event.clientY - rect.top;
      const cam = camRef.current;
      let delta = event.deltaY;
      if (event.deltaMode === 1) delta *= 16;
      else if (event.deltaMode === 2) delta *= viewport.clientHeight;

      const fit = Math.min(viewport.clientWidth / side, viewport.clientHeight / side);
      const minScale = Math.min(0.05, fit * 0.25);
      const nextScale = Math.min(192, Math.max(minScale, cam.scale * Math.exp(-delta * 0.0012)));
      if (nextScale === cam.scale) return;

      const worldX = (pointerX - cam.x) / cam.scale;
      const worldY = (pointerY - cam.y) / cam.scale;
      cam.scale = nextScale;
      cam.x = pointerX - worldX * nextScale;
      cam.y = pointerY - worldY * nextScale;
      movedRef.current = true;
      redraw();
      schedulePump();
    };

    const onResize = () => {
      if (!movedRef.current) fitRef.current();
      redraw();
      schedulePump();
    };

    viewport.addEventListener("wheel", onWheel, { passive: false });
    const observer = new ResizeObserver(onResize);
    observer.observe(viewport);
    onResize();

    return () => {
      viewport.removeEventListener("wheel", onWheel);
      observer.disconnect();
      if (frameRef.current !== 0) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = 0;
      }
      if (fetchTimerRef.current !== 0) {
        window.clearTimeout(fetchTimerRef.current);
        fetchTimerRef.current = 0;
      }
    };
  }, []);

  useEffect(() => {
    const hold = { active: false, start: 0, done: 0, frame: 0 };

    const stop = () => {
      hold.active = false;
      if (hold.frame !== 0) {
        window.cancelAnimationFrame(hold.frame);
        hold.frame = 0;
      }
    };

    // While space is held the total step count follows `speed * seconds`, so a
    // tap still yields exactly one step.
    const tick = () => {
      if (!hold.active) return;
      const elapsed = (performance.now() - hold.start) / 1000;
      const target = Math.max(1, Math.floor(elapsed * Math.max(1, speedRef.current)));
      if (target > hold.done) {
        queueSteps(target - hold.done);
        hold.done = target;
      }
      hold.frame = window.requestAnimationFrame(tick);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code !== "Space") return;
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
      event.preventDefault();
      if (event.repeat || hold.active) return;

      hold.active = true;
      hold.start = performance.now();
      hold.done = 1;
      queueSteps(1);
      hold.frame = window.requestAnimationFrame(tick);
    };

    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") stop();
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", stop);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", stop);
      stop();
    };
  }, []);

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || sideRef.current === 0) return;
    const cam = camRef.current;
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      camX: cam.x,
      camY: cam.y,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  };

  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const cam = camRef.current;
    cam.x = drag.camX + (event.clientX - drag.x);
    cam.y = drag.camY + (event.clientY - drag.y);
    movedRef.current = true;
    requestDrawRef.current();
    schedulePump();
  };

  const endDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setDragging(false);
  };

  /** Marks every LOD of the given chunks for a refetch. */
  const invalidateChunks = (indices: Iterable<number>) => {
    const wanted = new Set(indices);
    if (wanted.size === 0) return;

    const perSide = Math.max(1, Math.round(sideRef.current / chunkCellsRef.current));
    const cache = chunkCacheRef.current;
    for (const [key, texture] of cache) {
      const [, chunkX, chunkY] = key.split(":");
      if (wanted.has(Number(chunkY) * perSide + Number(chunkX))) texture.stale = true;
    }
    // Any batch that was already in flight holds pre-change pixels.
    generationRef.current += 1;
  };

  /**
   * Adopts a fresh grid: old textures belong to the previous world, so the
   * whole cache is dropped before anything is drawn again.
   */
  const applyWorld = (info: WorldInfo, refit: boolean) => {
    chunkCacheRef.current.clear();
    generationRef.current += 1;
    if (info.chunk_cells > 0) chunkCellsRef.current = info.chunk_cells;
    sideRef.current = info.side;
    if (!sizeInitialisedRef.current) {
      sizeInitialisedRef.current = true;
      setSizeText(String(info.side));
    }
    setStep(info.step);
    if (refit) fitRef.current();
    requestDrawRef.current();
    pumpRef.current();
    setStatus("ready");
    setMessage("");
  };

  /** Empties the field and restarts the step counter. */
  const resetWorld = () => {
    if (busyRef.current) return;
    // Steps queued or running now describe a world that is about to vanish.
    stepEpochRef.current += 1;
    pendingStepsRef.current = 0;
    const epoch = stepEpochRef.current;
    busyRef.current = true;
    setBusy(true);

    void invoke<WorldInfo>("reset_world")
      .then((info) => {
        if (epoch !== stepEpochRef.current) return;
        applyWorld(info, false);
      })
      .catch((error) => {
        if (epoch !== stepEpochRef.current) return;
        const detail = error instanceof Error ? error.message : String(error);
        setStatus("error");
        setMessage(detail ? `Не удалось сбросить поле: ${detail}` : "Не удалось сбросить поле");
      })
      .finally(() => {
        if (epoch !== stepEpochRef.current) return;
        busyRef.current = false;
        setBusy(false);
      });
  };

  /**
   * Relaunches the app with another size. The grid is sized once per process,
   * which keeps the bit layout static instead of resizing chunks on the fly.
   */
  const restartWithSize = () => {
    if (busyRef.current) return;
    const side = parseFieldSide(sizeText);
    if (side === null) {
      setStatus("error");
      setMessage(
        `Размер поля должен быть квадратом целого числа от ${MIN_SIDE} до ${MAX_SIDE}`,
      );
      return;
    }

    busyRef.current = true;
    setBusy(true);
    setStatus("ready");
    setMessage("Перезапуск…");
    void invoke("restart_with_size", { side }).catch((error) => {
      const detail = error instanceof Error ? error.message : String(error);
      busyRef.current = false;
      setBusy(false);
      setStatus("error");
      setMessage(detail ? `Не удалось перезапустить: ${detail}` : "Не удалось перезапустить");
    });
  };

  /** Coalesces step requests so a long hold becomes one backend call. */
  flushStepsRef.current = () => {
    if (stepBusyRef.current) return;
    const count = pendingStepsRef.current;
    if (count <= 0) return;

    pendingStepsRef.current = 0;
    stepBusyRef.current = true;
    const epoch = stepEpochRef.current;

    void invoke<StepResult>("next_step", { steps: count })
      .then((result) => {
        // A reset may have replaced the field while this batch was running.
        if (epoch !== stepEpochRef.current) return;
        setStep(result.step);
        setStatus("ready");
        setMessage("");
        invalidateChunks(result.changed);
        if (result.changed.length > 0) pumpRef.current();
        else requestDrawRef.current();
      })
      .catch((error) => {
        if (epoch !== stepEpochRef.current) return;
        const detail = error instanceof Error ? error.message : String(error);
        setStatus("error");
        setMessage(detail ? `Не удалось выполнить шаг: ${detail}` : "Не удалось выполнить шаг");
      })
      .finally(() => {
        stepBusyRef.current = false;
        flushStepsRef.current();
      });
  };

  const queueSteps = (count: number) => {
    if (count <= 0 || busyRef.current) return;
    pendingStepsRef.current += count;
    flushStepsRef.current();
  };

  const flipAt = (clientX: number, clientY: number) => {
    const viewport = viewportRef.current;
    const side = sideRef.current;
    if (!viewport || side === 0) return;

    const rect = viewport.getBoundingClientRect();
    const cam = camRef.current;
    const x = Math.floor((clientX - rect.left - cam.x) / cam.scale);
    const y = Math.floor((clientY - rect.top - cam.y) / cam.scale);
    if (x < 0 || x >= side || y < 0 || y >= side) return;

    const index = y * side + x;
    void invoke<number>("flip_cell", { index })
      .then((chunkIndex) => {
        invalidateChunks([chunkIndex]);
        pumpRef.current();
        setStatus("ready");
        setMessage("");
      })
      .catch((error) => {
        const detail = error instanceof Error ? error.message : String(error);
        setStatus("error");
        setMessage(detail ? `Не удалось изменить клетку: ${detail}` : "Не удалось изменить клетку");
      });
  };

  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag) {
      // A click flips a cell; anything that moved further was a pan.
      const distance = Math.hypot(event.clientX - drag.x, event.clientY - drag.y);
      if (distance < 4) flipAt(event.clientX, event.clientY);
    }
    endDrag(event);
  };

  return (
    <main className="app">
      <div
        ref={viewportRef}
        className={dragging ? "viewport dragging" : "viewport"}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={endDrag}
      >
        <canvas ref={canvasRef} />
        <div className="step-counter">
          <span className="step-label">шаг</span>
          <span className="step-value">{step.toLocaleString("ru-RU")}</span>
        </div>
        <aside className="panel" onPointerDown={(event) => event.stopPropagation()}>
          <button
            type="button"
            className="panel-reset"
            disabled={busy}
            onClick={(event) => {
              event.currentTarget.blur();
              resetWorld();
            }}
          >
            Сбросить
          </button>
          <div className="panel-divider" />
          <label className="panel-field">
            <span>Шагов/с</span>
            <input
              type="number"
              min={1}
              max={120}
              value={speed}
              onChange={(event) => setSpeed(clamp(Number(event.target.value) || 1, 1, 120))}
            />
          </label>
          <label className="panel-field">
            <span>Пропуск</span>
            <input
              type="number"
              min={1}
              max={10000}
              value={skipCount}
              onChange={(event) => setSkipCount(clamp(Number(event.target.value) || 1, 1, 10000))}
            />
          </label>
          <button
            type="button"
            disabled={busy}
            onClick={(event) => {
              event.currentTarget.blur();
              queueSteps(skipCount);
            }}
          >
            Пропустить
          </button>
          <div className="panel-divider" />
          <label className="panel-field">
            <span>Поле</span>
            <input
              type="number"
              min={MIN_SIDE}
              max={MAX_SIDE}
              step={1}
              value={sizeText}
              onChange={(event) => setSizeText(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") restartWithSize();
              }}
            />
          </label>
          <button type="button" disabled={busy} onClick={restartWithSize}>
            Изменить размер
          </button>
        </aside>
        {message ? <div className={status === "error" ? "viewport-message error" : "viewport-message"}>{message}</div> : null}
      </div>
    </main>
  );
}
