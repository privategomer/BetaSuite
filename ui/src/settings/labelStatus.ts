// Whether a per-label value is the one a run uses, from the engine's own
// resolution (config.get labels: true). The page never works out tiers
// itself: it reads which layer won, which do not apply, and why.
import { t } from '../i18n/format';
import type { ChainLayer, LabelView } from './api';
import { segments } from './model';

export type LabelStatus =
  | { readonly kind: 'in_effect' }
  | { readonly kind: 'overridden'; readonly by: string }
  | { readonly kind: 'not_set'; readonly uses: string }
  | { readonly kind: 'inactive'; readonly reason: NonNullable<ChainLayer[ 'inactive' ]> };

export interface LabelFacts {
  readonly label: string;
  readonly status: LabelStatus;
  readonly offBy: string | null;
  readonly unused: boolean;
}

// pattern is the registry path (with <label>), path one instance of it.
export function labelFacts( pattern: string, path: string, labels: ReadonlyMap<string, LabelView> ): LabelFacts | null {
  const at = segments( pattern ).indexOf( '<label>' );
  const parts = segments( path );
  const label = at >= 0 ? parts[ at ] : undefined;
  const key = parts.at( -1 );
  if ( label === undefined || key === undefined ) {
    return null;
  }
  const view = labels.get( label );
  const field = view?.fields[ key ];
  if ( view === undefined || field === undefined ) {
    return null;
  }
  const layer = field.chain.find( ( item ) => item.source === path );
  let status: LabelStatus;
  if ( layer?.wins === true ) {
    status = { kind: 'in_effect' };
  } else if ( layer?.inactive !== undefined ) {
    status = { kind: 'inactive', reason: layer.inactive };
  } else if ( layer !== undefined ) {
    status = { kind: 'overridden', by: field.source };
  } else {
    status = { kind: 'not_set', uses: field.source };
  }
  return { label, status, offBy: field.off ?? null, unused: !view.rendered && !view.window };
}

function sourceText( source: string ): string {
  return source === 'builtin' ? t( 'settings.label.builtin' ) : source;
}

export function labelStatusText( facts: LabelFacts ): string {
  const { status } = facts;
  switch ( status.kind ) {
    case 'in_effect':
      return t( 'settings.label.in_effect', { label: facts.label } );
    case 'overridden':
      return t( 'settings.label.overridden', { source: sourceText( status.by ) } );
    case 'not_set':
      return t( 'settings.label.not_set', { source: sourceText( status.uses ) } );
    case 'inactive':
      switch ( status.reason ) {
        case 'backend_not_selected': return t( 'settings.label.inactive.backend_not_selected' );
        case 'variant_not_selected': return t( 'settings.label.inactive.variant_not_selected' );
        case 'not_backend_tunable': return t( 'settings.label.inactive.not_backend_tunable' );
        case 'look_key_in_window': return t( 'settings.label.inactive.look_key_in_window' );
        case 'reverse_off': return t( 'settings.label.inactive.reverse_off' );
        case 'not_a_window': return t( 'settings.label.inactive.not_a_window' );
        case 'is_a_window': return t( 'settings.label.inactive.is_a_window' );
        case 'per_profile': return t( 'settings.label.inactive.per_profile' );
      }
  }
}
