export type SortDirection = "asc" | "desc";

export type SortState = {
  key: string;
  direction: SortDirection;
} | null;

export type TableFilter<T> = {
  value: string;
  getValue: (row: T) => unknown;
};

function isMissing(value: unknown) {
  return value === null || value === undefined || (typeof value === "number" && !Number.isFinite(value)) || (typeof value === "string" && value.trim() === "");
}

function compareValues(left: unknown, right: unknown) {
  const leftMissing = isMissing(left);
  const rightMissing = isMissing(right);
  if (leftMissing || rightMissing) {
    if (leftMissing && rightMissing) return 0;
    return leftMissing ? 1 : -1;
  }
  if (typeof left === "number" && typeof right === "number") return left - right;
  if (typeof left === "boolean" && typeof right === "boolean") return Number(left) - Number(right);
  return String(left).localeCompare(String(right), "ko", { numeric: true, sensitivity: "base" });
}

export function toggleSort(current: SortState, key: string): SortState {
  if (!current || current.key !== key) return { key, direction: "asc" };
  if (current.direction === "asc") return { key, direction: "desc" };
  return null;
}

export function filterAndSortRows<T>({
  rows,
  query = "",
  searchText,
  filters = [],
  sort,
  columns,
}: {
  rows: T[];
  query?: string;
  searchText: (row: T) => string;
  filters?: TableFilter<T>[];
  sort?: SortState;
  columns: Record<string, (row: T) => unknown>;
}) {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const filtered = rows.filter(row => {
    if (normalizedQuery && !searchText(row).toLocaleLowerCase().includes(normalizedQuery)) return false;
    return filters.every(filter => !filter.value || String(filter.getValue(row) ?? "") === filter.value);
  });
  if (!sort) return filtered;
  const accessor = columns[sort.key];
  if (!accessor) return filtered;
  return filtered
    .map((row, index) => ({ row, index }))
    .sort((left, right) => {
      const leftValue = accessor(left.row);
      const rightValue = accessor(right.row);
      const leftMissing = isMissing(leftValue);
      const rightMissing = isMissing(rightValue);
      if (leftMissing || rightMissing) {
        if (leftMissing !== rightMissing) return leftMissing ? 1 : -1;
        return left.index - right.index;
      }
      const comparison = compareValues(leftValue, rightValue);
      if (comparison !== 0) return sort.direction === "asc" ? comparison : -comparison;
      return left.index - right.index;
    })
    .map(entry => entry.row);
}

export function uniqueFilterValues<T>(rows: T[], getValue: (row: T) => unknown) {
  return Array.from(new Set(rows.map(row => getValue(row)).filter(value => typeof value === "string" && value.length > 0) as string[]))
    .sort((left, right) => left.localeCompare(right, "ko", { numeric: true, sensitivity: "base" }));
}
