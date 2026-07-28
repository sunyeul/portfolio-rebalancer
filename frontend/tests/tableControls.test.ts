import { expect, test } from "bun:test";

import { filterAndSortRows, toggleSort, uniqueFilterValues, type SortState } from "../src/lib/tableControls";

type Row = { name: string; layer: string; status: string; value: number | null };
const rows: Row[] = [
  { name: "QQQ", layer: "satellite", status: "Watch", value: 11.4 },
  { name: "KODEX 200", layer: "core", status: "OK", value: null },
  { name: "실험 자산", layer: "experiment", status: "Review", value: 3.2 },
];

const controls = (sort: SortState = null) => filterAndSortRows({
  rows,
  searchText: row => `${row.name} ${row.layer} ${row.status}`,
  columns: { name: row => row.name, value: row => row.value },
  sort,
});

test("search and exact filters narrow rows without changing source order", () => {
  expect(filterAndSortRows({
    rows,
    query: "kodex",
    searchText: row => `${row.name} ${row.layer} ${row.status}`,
    filters: [{ value: "core", getValue: row => row.layer }],
    columns: {},
  }).map(row => row.name)).toEqual(["KODEX 200"]);
  expect(controls().map(row => row.name)).toEqual(["QQQ", "KODEX 200", "실험 자산"]);
});

test("numeric sort keeps missing values after present values in both directions", () => {
  expect(controls({ key: "value", direction: "asc" }).map(row => row.name)).toEqual(["실험 자산", "QQQ", "KODEX 200"]);
  expect(controls({ key: "value", direction: "desc" }).map(row => row.name)).toEqual(["QQQ", "실험 자산", "KODEX 200"]);
});

test("strings use natural locale ordering and unknown sort keys preserve order", () => {
  expect(controls({ key: "name", direction: "asc" }).map(row => row.name)).toEqual(["실험 자산", "KODEX 200", "QQQ"]);
  expect(controls({ key: "unknown", direction: "asc" }).map(row => row.name)).toEqual(["QQQ", "KODEX 200", "실험 자산"]);
});

test("sort cycles through ascending, descending, and reset", () => {
  expect(toggleSort(null, "value")).toEqual({ key: "value", direction: "asc" });
  expect(toggleSort({ key: "value", direction: "asc" }, "value")).toEqual({ key: "value", direction: "desc" });
  expect(toggleSort({ key: "value", direction: "desc" }, "value")).toBeNull();
  expect(toggleSort({ key: "name", direction: "asc" }, "value")).toEqual({ key: "value", direction: "asc" });
});

test("filter options are unique and locale sorted", () => {
  expect(uniqueFilterValues([...rows, rows[0]], row => row.layer)).toEqual(["core", "experiment", "satellite"]);
});
