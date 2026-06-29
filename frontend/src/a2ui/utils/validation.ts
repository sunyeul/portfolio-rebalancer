import { z } from 'zod';

import {
  IPS_PILOT_REVIEW_CATALOG_ID,
  allowedReviewSurfaces,
  appSurfaceModes,
  appSurfaceTargetValues,
  forbiddenReviewActionValues,
  reviewDispositionValues
} from '../catalogs/ipsPilotReviewCatalog';
import { journalDraftComposerSurfaceSchema } from '../schemas/journalDraftComposer.schema';
import { reviewQueueTriageSurfaceSchema } from '../schemas/reviewQueueTriage.schema';
import type {
  A2UiAppSurfaceEnvelope,
  AppSurfaceTarget,
  GenerativeSurfacePatch,
  ReviewA2UIEnvelope,
  ReviewA2UISurfacePayload
} from '../types';

export type A2UiAppSurfaceValidationResult =
  | {
      ok: true;
      envelope: A2UiAppSurfaceEnvelope;
    }
  | {
      ok: false;
      fallback_text: string;
      validation_errors: string[];
    };

function collectStrings(value: unknown, output: string[] = []) {
  if (typeof value === 'string') {
    output.push(value);
    return output;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectStrings(item, output));
    return output;
  }
  if (value && typeof value === 'object') {
    Object.values(value).forEach((item) => collectStrings(item, output));
  }
  return output;
}

function forbiddenVocabularyErrors(value: unknown) {
  const haystack = collectStrings(value).join(' ').toLowerCase();
  return forbiddenReviewActionValues
    .filter((action) => {
      if (action === 'buy' || action === 'sell') {
        return new RegExp(`\\b${action}\\b`).test(haystack);
      }
      return haystack.includes(action);
    })
    .map((action) => `Forbidden A2UI action vocabulary: ${action}`);
}

function zodErrors(error: z.ZodError) {
  return error.issues.map((issue) => `${issue.path.join('.') || 'payload'}: ${issue.message}`);
}

function surfaceSchemaForComponent(component: string) {
  if (component === 'ReviewQueueTriageSurface') return reviewQueueTriageSurfaceSchema;
  if (component === 'JournalDraftComposerSurface') return journalDraftComposerSurfaceSchema;
  return null;
}

function targetForComponent(component: string): AppSurfaceTarget | null {
  if (component === 'ReviewQueueTriageSurface') return 'review_queue';
  if (component === 'JournalDraftComposerSurface') return 'journal_draft';
  return null;
}

function componentFrom(value: unknown) {
  return value && typeof value === 'object' && 'component' in value ? String(value.component) : '';
}

function fallback(validation_errors: string[]): A2UiAppSurfaceValidationResult {
  return {
    ok: false,
    fallback_text: 'Generated UI를 앱 본문에 반영하지 못했습니다. 대신 텍스트 요약으로 표시합니다.',
    validation_errors
  };
}

export function validateReviewA2UISurface(value: unknown): ReviewA2UIEnvelope {
  const component = componentFrom(value);
  if (!allowedReviewSurfaces.includes(component as (typeof allowedReviewSurfaces)[number])) {
    return {
      ok: false,
      fallback_text: 'A2UI validation failed. Review Queue를 일반 텍스트 요약으로 확인하세요.',
      validation_errors: [`Unknown A2UI component: ${component || 'missing'}`]
    };
  }

  const schema = surfaceSchemaForComponent(component);
  if (!schema) {
    return {
      ok: false,
      fallback_text: 'A2UI validation failed. 이 surface는 Phase 2에서 렌더링하지 않습니다.',
      validation_errors: [`Unsupported renderable A2UI component: ${component}`]
    };
  }

  const parsed = schema.safeParse(value);
  const errors = parsed.success ? [] : zodErrors(parsed.error);
  const vocabularyErrors = forbiddenVocabularyErrors(value);
  const validation_errors = [...errors, ...vocabularyErrors];

  if (validation_errors.length > 0) {
    return {
      ok: false,
      fallback_text: 'A2UI validation failed. 검토용 plain text fallback을 사용하세요.',
      validation_errors
    };
  }

  return {
    ok: true,
    surface: parsed.data as ReviewA2UISurfacePayload
  };
}

export function validateA2UiAppSurfaceEnvelope(payload: unknown): A2UiAppSurfaceValidationResult {
  if (!payload || typeof payload !== 'object') {
    return fallback(['A2UI app surface envelope must be an object.']);
  }

  const envelope = payload as Record<string, unknown>;
  if (envelope.catalog_id !== IPS_PILOT_REVIEW_CATALOG_ID) {
    return fallback([`Unsupported A2UI catalog_id: ${String(envelope.catalog_id || 'missing')}`]);
  }

  if (!appSurfaceTargetValues.includes(envelope.target as (typeof appSurfaceTargetValues)[number])) {
    return fallback([`Unsupported A2UI app surface target: ${String(envelope.target || 'missing')}`]);
  }

  if (!appSurfaceModes.includes(envelope.mode as (typeof appSurfaceModes)[number])) {
    return fallback([`Unsupported A2UI app surface mode: ${String(envelope.mode || 'missing')}`]);
  }

  const surface = envelope.surface;
  const component = componentFrom(surface);
  if (!allowedReviewSurfaces.includes(component as (typeof allowedReviewSurfaces)[number])) {
    return fallback([`Unknown A2UI component: ${component || 'missing'}`]);
  }

  const expectedTarget = targetForComponent(component);
  if (expectedTarget !== envelope.target) {
    return fallback([`${component} cannot patch ${String(envelope.target)}.`]);
  }

  const surfaceEnvelope = validateReviewA2UISurface(surface);
  if (!surfaceEnvelope.ok) {
    return fallback(surfaceEnvelope.validation_errors);
  }

  const source = envelope.source;
  if (!source || typeof source !== 'object') {
    return fallback(['A2UI app surface source is required.']);
  }
  if ((source as Record<string, unknown>).agent !== 'reviewCopilot') {
    return fallback([`Unsupported A2UI app surface agent: ${String((source as Record<string, unknown>).agent || 'missing')}`]);
  }
  if (typeof (source as Record<string, unknown>).created_at !== 'string') {
    return fallback(['A2UI app surface source.created_at is required.']);
  }

  return {
    ok: true,
    envelope: {
      catalog_id: IPS_PILOT_REVIEW_CATALOG_ID,
      target: envelope.target as A2UiAppSurfaceEnvelope['target'],
      mode: envelope.mode as A2UiAppSurfaceEnvelope['mode'],
      surface: surfaceEnvelope.surface,
      source: {
        agent: 'reviewCopilot',
        created_at: (source as Record<string, unknown>).created_at as string
      }
    }
  };
}

export function convertA2UiToSurfacePatch(envelope: A2UiAppSurfaceEnvelope): GenerativeSurfacePatch {
  if (envelope.target === 'review_queue' && envelope.surface.component === 'ReviewQueueTriageSurface') {
    return {
      target: 'review_queue',
      surface: 'ReviewQueueTriageSurface',
      mode: envelope.mode,
      payload: envelope.surface
    };
  }

  if (envelope.target === 'journal_draft' && envelope.surface.component === 'JournalDraftComposerSurface') {
    return {
      target: 'journal_draft',
      surface: 'JournalDraftComposerSurface',
      mode: envelope.mode,
      payload: envelope.surface
    };
  }

  throw new Error(`${envelope.surface.component} cannot patch ${envelope.target}.`);
}

export function applyA2UiAppSurfacePayload(
  payload: unknown,
  applySurfacePatch: (patch: GenerativeSurfacePatch) => void
): A2UiAppSurfaceValidationResult {
  const validation = validateA2UiAppSurfaceEnvelope(payload);
  if (!validation.ok) return validation;

  applySurfacePatch(convertA2UiToSurfacePatch(validation.envelope));
  return validation;
}

export function validateReviewDispositions(value: unknown) {
  const dispositions = Array.isArray(value) ? value : [];
  return dispositions.every(
    (decision) =>
      decision &&
      typeof decision === 'object' &&
      'disposition' in decision &&
      reviewDispositionValues.includes(String(decision.disposition) as (typeof reviewDispositionValues)[number])
  );
}
