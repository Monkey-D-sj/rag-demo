export type Node = {
  id: string;
  next?: string;
};
export function runBasicGraph(nodes: Record<string, Node>, startId: string) {
  const path: string[] = [];
  let current = startId;
  while (current) {
    path.push(current);
    const next = nodes[current]?.next;
    if (!next) break;
    current = next;
  }
  return path;
}
