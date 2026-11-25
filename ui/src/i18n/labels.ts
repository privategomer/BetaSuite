// A detection label's name, for every screen that shows one. A label with
// no name in the catalog (one a newer model added) shows its id.
import { hasMessage, msgKey, t } from './format';

export function labelName( id: string ): string {
  const key = msgKey( 'label.', `${id}.name` );
  return hasMessage( key ) ? t( key ) : id;
}

// A per-label setting path's instance by its label's name, when one of its
// parts is a label: item_overrides.exposed_breast.censor_style reads as
// that label's name. Anything else is shown as the path.
export function instanceName( path: string, labels: readonly string[] ): string {
  const label = path.split( '.' ).find( ( part ) => labels.includes( part ) );
  return label === undefined ? path : labelName( label );
}

// Why a label cannot be hidden, from the code config.get gives it
// (label.not_in_model: the selected backend does not report it).
export function unavailableText( reason: { readonly code: string; readonly params: Readonly<Record<string, unknown>> } ): string {
  const key = msgKey( 'hide.unavailable.', reason.code );
  const backend = reason.params[ 'backend' ];
  return hasMessage( key ) ? t( key, { backend: typeof backend === 'string' ? backend : '' } ) : t( 'hide.unavailable.other' );
}
