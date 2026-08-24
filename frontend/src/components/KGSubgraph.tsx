import { useMemo, useRef, useState } from "react";
import { ObjectPreviewCard } from "./ObjectPreviewCard";
import type { KGSubgraph as KGSubgraphData } from "../lib/types";

/* Deliberately NOT force-directed.
 *
 * This subgraph explains exactly one answer -- build_kg_subgraph() puts only
 * the cited objects and their events in it (see answer_generator.py). For
 * that job a stable tiered arrangement beats an organic blob: the same
 * question always draws the same shape, it reads at a glance, and it holds
 * up dropped into a paper figure. A force layout also settles differently on
 * every render, which makes two runs of the same query look like two
 * different findings.
 *
 * Tiers, top to bottom: Event (what happened) -> TrafficObject (who) ->
 * Location (where), matching how the sentence above it reads.
 *
 * Node SHAPE carries the label as well as hue, so the three kinds stay
 * distinguishable in greyscale, under CVD, and in forced-colors mode.
 *
 * GLOW is the one thing added to a node after it settles, and it means
 * exactly one thing: this TrafficObject has a real photograph on file
 * (evidenceFramesByObject has an entry for it). Not "important", not
 * "high confidence" by some invented score -- just "there is a picture,
 * hover to see it". Event/Location nodes never glow: this app has no photo
 * for an occurrence or a place, only for the thing the camera actually saw. */

const TIER: Record<string, number> = { Event: 0, TrafficObject: 1, Location: 2 };

const W = 640;
const ROW_H = 104;
const PAD_X = 44;
const PAD_Y = 34;

interface Placed {
  id: string;
  label: string;
  properties: Record<string, unknown>;
  x: number;
  y: number;
}

interface HoverState {
  node: Placed;
  x: number;
  y: number;
}

export function KGSubgraph({
  data,
  evidenceFramesByObject = {},
}: {
  data: KGSubgraphData;
  evidenceFramesByObject?: Record<string, string[]>;
}) {
  const figureRef = useRef<HTMLElement>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  const { nodes, edges, height, counts, revealMs } = useMemo(() => {
    const tiers = new Map<number, typeof data.nodes>();
    for (const n of data.nodes) {
      const t = TIER[n.label] ?? 1;
      if (!tiers.has(t)) tiers.set(t, []);
      tiers.get(t)!.push(n);
    }
    const usedTiers = [...tiers.keys()].sort((a, b) => a - b);

    const placed: Placed[] = [];
    usedTiers.forEach((tier, row) => {
      const group = tiers.get(tier)!;
      const span = W - PAD_X * 2;
      group.forEach((n, i) => {
        // Single node in a tier centres; otherwise spread evenly.
        const x = group.length === 1 ? W / 2 : PAD_X + (span * i) / (group.length - 1);
        placed.push({ ...n, x, y: PAD_Y + row * ROW_H });
      });
    });

    const byId = new Map(placed.map((p) => [p.id, p]));
    const drawn = data.edges
      .map((e) => ({ ...e, a: byId.get(e.source), b: byId.get(e.target) }))
      .filter((e) => e.a && e.b);

    return {
      nodes: placed,
      edges: drawn,
      height: PAD_Y * 2 + Math.max(0, usedTiers.length - 1) * ROW_H,
      counts: {
        objects: data.nodes.filter((n) => n.label === "TrafficObject").length,
        events: data.nodes.filter((n) => n.label === "Event").length,
        relations: data.edges.length,
      },
      // Heuristic upper bound on when the settle/draw animation finishes, so
      // the summary panel arrives after the graph it summarises rather than
      // racing it. Not a precise replay of NodeMark's per-node delay math --
      // an approximation on the safe side is enough for a panel that only
      // needs to feel like it follows, not frame-match.
      revealMs: 420 + placed.length * 70 + drawn.length * 55,
    };
  }, [data]);

  if (data.nodes.length === 0) {
    return <p className="text-sm text-ink-3">No graph evidence for this answer.</p>;
  }

  const showHoverAt = (node: Placed, clientX: number, clientY: number) => {
    const rect = figureRef.current?.getBoundingClientRect();
    if (!rect) return;
    setHover({ node, x: clientX - rect.left, y: clientY - rect.top });
  };

  return (
    <div className="flex flex-col gap-2">
      {/* Scrolling and popover-anchoring are deliberately split across two
          nested boxes: overflow-x-auto forces overflow-y to compute as auto
          too (not "visible"), which would clip an absolutely-positioned
          popover the instant it extended past this box. The outer figure
          stays overflow-visible so the popover is never cropped; only the
          inner div scrolls the (possibly wide) SVG itself. */}
      <figure ref={figureRef} className="relative m-0 rounded-sm border border-hairline bg-well grid-paper">
        <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${height}`}
          width="100%"
          style={{ minWidth: 480, display: "block" }}
          role="img"
          aria-label={`Knowledge subgraph: ${data.nodes.length} nodes, ${data.edges.length} relations`}
        >
          <defs>
            <marker
              id="kg-arrow"
              viewBox="0 0 8 8"
              refX="7"
              refY="4"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M0 1 L7 4 L0 7 z" fill="var(--color-hairline-lit)" />
            </marker>
          </defs>

          {/* Relations first, so nodes sit above their connectors. */}
          {edges.map((e, i) => {
            const len = Math.hypot(e.b!.x - e.a!.x, e.b!.y - e.a!.y);
            const mx = (e.a!.x + e.b!.x) / 2;
            const my = (e.a!.y + e.b!.y) / 2;
            return (
              <g key={`${e.source}-${e.target}-${i}`}>
                <line
                  className="kg-edge"
                  x1={e.a!.x}
                  y1={e.a!.y}
                  x2={e.b!.x}
                  y2={e.b!.y}
                  stroke="var(--color-hairline-lit)"
                  strokeWidth="1.5"
                  markerEnd="url(#kg-arrow)"
                  style={
                    {
                      "--edge-len": len,
                      animationDelay: `${360 + i * 45}ms`,
                    } as React.CSSProperties
                  }
                />
                <text
                  x={mx}
                  y={my - 5}
                  textAnchor="middle"
                  className="kg-node"
                  style={{ animationDelay: `${420 + i * 45}ms` }}
                  fill="var(--color-ink-3)"
                  fontSize="10"
                  fontFamily="var(--font-mono)"
                >
                  {e.type}
                </text>
              </g>
            );
          })}

          {nodes.map((n, i) => (
            <NodeMark
              key={n.id}
              node={n}
              index={i}
              hasPhoto={n.label === "TrafficObject" && !!evidenceFramesByObject[n.id]?.length}
              onHover={(clientX, clientY) => showHoverAt(n, clientX, clientY)}
              onLeave={() => setHover((h) => (h?.node.id === n.id ? null : h))}
            />
          ))}
        </svg>
        </div>

        {hover && (
          <div
            className="pointer-events-none absolute z-30"
            style={{
              // Clamped against the container's actual rendered width, not
              // the SVG's internal viewBox width -- those are different
              // coordinate systems, and using the wrong one let the popover
              // overflow off-screen on a narrow (e.g. mobile) viewport.
              left: Math.min(hover.x + 12, Math.max(0, (figureRef.current?.clientWidth ?? W) - 232)),
              top: Math.max(hover.y - 12, 4),
            }}
          >
            <ObjectPreviewCard
              globalId={hover.node.id}
              label={hover.node.label === "Event" ? "event" : (hover.node.properties.class as string) ?? hover.node.label}
              cropUrl={
                hover.node.label === "TrafficObject" ? (evidenceFramesByObject[hover.node.id]?.[0] ?? null) : undefined
              }
              properties={hover.node.properties}
            />
          </div>
        )}
      </figure>

      {/* The glass readout -- arrives once the graph has finished settling,
          so it reads as a summary of what just assembled, not a caption
          racing its own subject. */}
      <div
        className="glass w-fit rounded-sm px-2.5 py-1.5 font-mono text-[11px] text-ink-2 panel-in"
        style={{ animationDelay: `${revealMs}ms` }}
      >
        {counts.objects} object{counts.objects === 1 ? "" : "s"} · {counts.events} event
        {counts.events === 1 ? "" : "s"} · {counts.relations} relation{counts.relations === 1 ? "" : "s"}
      </div>
    </div>
  );
}

function NodeMark({
  node,
  index,
  hasPhoto,
  onHover,
  onLeave,
}: {
  node: Placed;
  index: number;
  hasPhoto: boolean;
  onHover: (clientX: number, clientY: number) => void;
  onLeave: () => void;
}) {
  const isEvent = node.label === "Event";
  const isLocation = node.label === "Location";
  // Events are the machine's reading of motion; objects are what the camera
  // saw. Same split the palette uses everywhere else.
  const fill = isEvent ? "var(--color-annotate)" : "var(--color-sodium)";
  const delay = isEvent ? index * 55 : 160 + index * 55;

  const title = [
    `${node.id} (${node.label})`,
    ...Object.entries(node.properties)
      .filter(([, v]) => v != null)
      .map(([k, v]) => `${k}: ${v}`),
  ].join("\n");

  return (
    <g
      className="kg-node cursor-pointer"
      style={{ animationDelay: `${delay}ms` }}
      tabIndex={0}
      role="button"
      aria-label={title}
      onMouseEnter={(e) => onHover(e.clientX, e.clientY)}
      onMouseMove={(e) => onHover(e.clientX, e.clientY)}
      onMouseLeave={onLeave}
      onFocus={(e) => {
        const rect = (e.currentTarget as SVGGElement).getBoundingClientRect();
        onHover(rect.left + rect.width / 2, rect.top);
      }}
      onBlur={onLeave}
    >
      <title>{title}</title>
      {hasPhoto && (
        // The glow ring: drawn as its own soft-edged circle beneath the
        // mark, animated, rather than a filter on the mark itself -- keeps
        // the mark's own edge crisp (still legible at small sizes) while the
        // glow reads as a halo around it.
        <circle
          cx={node.x}
          cy={node.y}
          r="13"
          fill="none"
          stroke="var(--color-sodium)"
          strokeWidth="2"
          opacity="0.55"
          className="glow-breathe"
          style={{ "--glow-color": "var(--color-sodium)" } as React.CSSProperties}
        />
      )}
      {isEvent ? (
        // Diamond: an occurrence.
        <rect
          x={node.x - 9}
          y={node.y - 9}
          width="18"
          height="18"
          transform={`rotate(45 ${node.x} ${node.y})`}
          fill={fill}
          stroke="var(--color-well)"
          strokeWidth="2"
        />
      ) : isLocation ? (
        // Circle: a place.
        <circle cx={node.x} cy={node.y} r="9" fill={fill} stroke="var(--color-well)" strokeWidth="2" />
      ) : (
        // Square: a physical thing.
        <rect
          x={node.x - 8}
          y={node.y - 8}
          width="16"
          height="16"
          rx="2"
          fill={fill}
          stroke="var(--color-well)"
          strokeWidth="2"
        />
      )}
      <text
        x={node.x}
        y={node.y + 25}
        textAnchor="middle"
        fill="var(--color-ink-2)"
        fontSize="11"
        fontFamily="var(--font-mono)"
      >
        {node.id}
      </text>
    </g>
  );
}

/** Shape legend -- identity is never hue alone. */
export function KGLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-ink-3">
      <span className="flex items-center gap-1.5">
        <svg width="11" height="11" aria-hidden>
          <rect x="1" y="1" width="9" height="9" rx="1.5" fill="var(--color-sodium)" />
        </svg>
        object
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="13" height="13" aria-hidden>
          <rect x="3" y="3" width="7" height="7" transform="rotate(45 6.5 6.5)" fill="var(--color-annotate)" />
        </svg>
        event
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="11" height="11" aria-hidden>
          <circle cx="5.5" cy="5.5" r="4.5" fill="var(--color-sodium)" />
        </svg>
        location
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="15" height="15" aria-hidden>
          <circle cx="7.5" cy="7.5" r="6" fill="none" stroke="var(--color-sodium)" strokeWidth="1.5" opacity="0.6" />
        </svg>
        has photo
      </span>
    </div>
  );
}
