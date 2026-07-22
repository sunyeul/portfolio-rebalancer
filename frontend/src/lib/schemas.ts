export const layerValues = ['core', 'satellite', 'experiment'] as const;
export const thesisStatusValues = ['valid', 'watch', 'broken', 'unknown'] as const;

export type LayerType = (typeof layerValues)[number];
export type ThesisStatusInput = (typeof thesisStatusValues)[number];

export type PortfolioRowInput = {
  ticker: string;
  allocation?: number | string | null;
  return_total?: number | string | null;
  layer?: LayerType | '' | null;
  thesis_status?: ThesisStatusInput | '' | null;
};
