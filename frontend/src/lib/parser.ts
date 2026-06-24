import Papa from 'papaparse';

import {
  categoryValues,
  layerValues,
  thesisStatusValues,
  type CategoryType,
  type LayerType,
  type PortfolioRowInput,
  type ThesisStatusInput
} from './schemas';

const fields = [
  'ticker',
  'allocation',
  'return_total',
  'layer',
  'category',
  'dca_enabled',
  'thesis_status'
] as const;

const headerMap: Record<string, keyof PortfolioRowInput> = {
  ticker: 'ticker',
  티커: 'ticker',
  종목: 'ticker',
  allocation: 'allocation',
  weight: 'allocation',
  비중: 'allocation',
  가중치: 'allocation',
  현재비중: 'allocation',
  returntotal: 'return_total',
  return_total: 'return_total',
  return: 'return_total',
  누적수익률: 'return_total',
  수익률: 'return_total',
  현재수익률: 'return_total',
  layer: 'layer',
  계층: 'layer',
  레이어: 'layer',
  category: 'category',
  카테고리: 'category',
  분류: 'category',
  dcaenabled: 'dca_enabled',
  dca_enabled: 'dca_enabled',
  dca: 'dca_enabled',
  정기매수: 'dca_enabled',
  정기매수대상: 'dca_enabled',
  thesisstatus: 'thesis_status',
  thesis_status: 'thesis_status',
  thesis: 'thesis_status',
  투자논리: 'thesis_status',
  논리상태: 'thesis_status'
};

export function blankRow(): PortfolioRowInput {
  return {
    ticker: '',
    allocation: '',
    return_total: '',
    layer: '',
    category: '',
    dca_enabled: true,
    thesis_status: 'intact'
  };
}

function normalizeHeader(value: unknown) {
  return String(value ?? '').trim().toLowerCase().replace(/[\s()%_-]+/g, '');
}

function normalizeNumber(value: unknown) {
  return String(value ?? '').trim().replace(/%/g, '').replace(/,/g, '');
}

function normalizeLayer(value: unknown): LayerType | '' {
  const normalized = String(value ?? '').trim().toLowerCase();
  return layerValues.includes(normalized as LayerType) ? normalized as LayerType : '';
}

function normalizeCategory(value: unknown): CategoryType | '' {
  const normalized = String(value ?? '').trim().toLowerCase();
  return categoryValues.includes(normalized as CategoryType) ? normalized as CategoryType : '';
}

function normalizeThesisStatus(value: unknown): ThesisStatusInput | '' {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized === '유지' || normalized === '유효') return 'valid';
  if (normalized === '관찰') return 'watch';
  if (normalized === '훼손') return 'broken';
  return thesisStatusValues.includes(normalized as ThesisStatusInput) ? normalized as ThesisStatusInput : '';
}

function booleanValue(value: unknown) {
  if (typeof value === 'boolean') return value;
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized === '') return true;
  return ['true', '1', 'yes', 'y', 'on', '정기', '가능'].includes(normalized);
}

function looksLikeTicker(value: unknown) {
  const text = String(value ?? '').trim();
  return (
    /^[A-Za-z0-9.-]{1,15}$/.test(text) &&
    (/[A-Za-z]/.test(text) || /^\d{7,8}(\.JP)?$/i.test(text))
  );
}

function parseFreeLine(line: string): PortfolioRowInput | null {
  const tokens = line
    .replace(/[()]/g, ' ')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);

  const tickerIndex = tokens.findIndex(looksLikeTicker);
  if (tickerIndex === -1) return null;

  const row = blankRow();
  row.ticker = tokens[tickerIndex].toUpperCase();

  const textTokens: string[] = [];
  const numericTokens: string[] = [];
  tokens.slice(tickerIndex + 1).forEach((token) => {
    const normalized = normalizeNumber(token);
    if (/^-?\d+(\.\d+)?$/.test(normalized)) numericTokens.push(normalized);
    else textTokens.push(token);
  });

  if (!numericTokens.length) return null;
  row.allocation = numericTokens[0];
  row.return_total = numericTokens[1] ?? '';

  const thesisIndex = textTokens.findIndex((token) =>
    ['intact', 'watch', 'broken', 'unknown', '유지', '관찰', '훼손'].includes(
      token.toLowerCase()
    )
  );
  if (thesisIndex !== -1) {
    row.thesis_status = normalizeThesisStatus(textTokens.splice(thesisIndex, 1)[0]);
  }

  const layerToken = textTokens.find((token) => normalizeLayer(token) !== '');
  const categoryToken = textTokens.find((token) => normalizeCategory(token) !== '');
  if (layerToken) row.layer = normalizeLayer(layerToken);
  if (categoryToken) row.category = normalizeCategory(categoryToken);
  return row;
}

function mapDelimitedRows(rows: string[][]) {
  if (!rows.length) return [];
  const firstRowKeys = rows[0].map((value) => headerMap[normalizeHeader(value)]);
  const hasHeader = firstRowKeys.filter(Boolean).length >= 2;
  const mappedFields = hasHeader ? firstRowKeys : fields;
  const dataRows = hasHeader ? rows.slice(1) : rows;

  return dataRows
    .map((values) => {
      const row = blankRow();
      values.forEach((value, index) => {
        const field = mappedFields[index];
        if (!field) return;
        if (field === 'dca_enabled') row[field] = booleanValue(value);
        else if (field === 'allocation' || field === 'return_total') row[field] = normalizeNumber(value);
        else if (field === 'ticker') row[field] = String(value ?? '').trim().toUpperCase();
        else if (field === 'layer') row.layer = normalizeLayer(value);
        else if (field === 'category') row.category = normalizeCategory(value);
        else if (field === 'thesis_status') row.thesis_status = normalizeThesisStatus(value);
      });
      return row;
    })
    .filter((row) => row.ticker || row.allocation);
}

export function parsePortfolioText(text: string): PortfolioRowInput[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  const firstLine = trimmed.split(/\r?\n/).find((line) => line.trim()) ?? '';
  if (trimmed.includes('\t') || firstLine.includes(',')) {
    const parsed = Papa.parse<string[]>(trimmed, {
      delimiter: trimmed.includes('\t') ? '\t' : ',',
      skipEmptyLines: true
    });
    const rows = mapDelimitedRows(parsed.data);
    if (rows.length) return rows;
  }

  return trimmed
    .replace(/\s+\/\s+/g, '\n')
    .replace(/\s*;\s*/g, '\n')
    .split(/\r?\n/)
    .map(parseFreeLine)
    .filter((row): row is PortfolioRowInput => Boolean(row));
}
