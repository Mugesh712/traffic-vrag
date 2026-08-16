import { useMemo, useRef } from "react";
import ForceGraph2D, { type NodeObject, type LinkObject } from "react-force-graph-2d";
import type { KGSubgraph as KGSubgraphData } from "../lib/types";

interface Props {
  data: KGSubgraphData;
}

const LABEL_COLOR: Record<string, string> = {
  TrafficObject: "#6cc0d4", // accent
  Event: "#eab868", // pending/amber
  Location: "#4fbe86", // done/green
};

// Explains THIS answer -- only the objects/events M13 actually cited, per
// build_kg_subgraph()'s own design choice (see answer_generator.py). Rendered
// with react-force-graph per the roadmap; kept deliberately plain (flat
// circles, monospace labels, no gradients) since this is meant to sit
// directly in a paper figure.
export function KGSubgraph({ data }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  const graphData = useMemo(
    () => ({
      nodes: data.nodes.map((n) => ({ id: n.id, label: n.label, properties: n.properties })),
      links: data.edges.map((e) => ({ source: e.source, target: e.target, type: e.type })),
    }),
    [data],
  );

  if (data.nodes.length === 0) {
    return <p className="text-sm text-ink-faint">No graph evidence for this answer.</p>;
  }

  return (
    <div ref={containerRef} className="overflow-hidden rounded-md border border-border bg-surface-sunken">
      <ForceGraph2D
        graphData={graphData}
        width={containerRef.current?.clientWidth ?? 560}
        height={320}
        backgroundColor="#0f161c"
        nodeRelSize={5}
        nodeColor={(node) => LABEL_COLOR[(node as NodeObject & { label: string }).label] ?? "#93a4af"}
        nodeLabel={(node) => {
          const n = node as NodeObject & { label: string; properties: Record<string, unknown> };
          const props = Object.entries(n.properties)
            .filter(([, v]) => v != null)
            .map(([k, v]) => `${k}: ${v}`)
            .join("\n");
          return `${n.id} (${n.label})\n${props}`;
        }}
        linkLabel={(link) => (link as LinkObject & { type: string }).type}
        linkColor={() => "#263139"}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkCurvature={0.15}
        nodeCanvasObjectMode={() => "after"}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const n = node as NodeObject & { id: string };
          const fontSize = 11 / globalScale;
          ctx.font = `${fontSize}px ui-monospace, monospace`;
          ctx.fillStyle = "#93a4af";
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.fillText(String(n.id), n.x ?? 0, (n.y ?? 0) + 7);
        }}
      />
    </div>
  );
}
