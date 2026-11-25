// Settings by screen: one tab per screen with that screen's fold. A
// stand-in host until the library, run and what-to-hide screens carry
// their own folds; Other is a tab of its own.
import { useId, useRef, useState, type KeyboardEvent } from 'react';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { SCREENS, type Screen } from './model';
import { SettingsFold } from './SettingsFold';
import { screenName } from './text';
import { useSettings } from './useSettings';

interface SettingsViewProps {
  readonly onHelp: ( anchor: string ) => void;
}

export function SettingsView( { onHelp }: SettingsViewProps ) {
  const { settings, reload } = useSettings();
  const [ current, setCurrent ] = useState<Screen>( SCREENS[ 0 ] ?? 'run' );
  const tabs = useRef( new Map<Screen, HTMLButtonElement>() );
  const baseId = useId();
  const data = settings.state === 'ready' ? settings.data : settings.state === 'loading' ? settings.last : null;

  // Arrow keys, Home and End move between tabs, as a tab list does.
  const onKey = ( event: KeyboardEvent ) => {
    const at = SCREENS.indexOf( current );
    const moves: Record<string, number> = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: SCREENS.length - 1 };
    const target = moves[ event.key ];
    if ( target === undefined ) {
      return;
    }
    event.preventDefault();
    const next = SCREENS[ ( target + SCREENS.length ) % SCREENS.length ];
    if ( next !== undefined ) {
      setCurrent( next );
      tabs.current.get( next )?.focus();
    }
  };

  if ( data === null ) {
    return (
      <div className="settings-view">
        <p className="settings-wait" role="status">
          {settings.state === 'failed' ? ( settings.error === null ? t( 'settings.request_failed' ) : errorText( settings.error ) )
            : t( 'settings.loading' )}
        </p>
      </div>
    );
  }

  return (
    <div className="settings-view">
      <div role="tablist" aria-label={t( 'settings.tabs_label' )} className="settings-tabs">
        {SCREENS.map( ( screen ) => (
          <button
            key={screen}
            ref={( element ) => { if ( element === null ) { tabs.current.delete( screen ); } else { tabs.current.set( screen, element ); } }}
            type="button"
            role="tab"
            id={`${baseId}-${screen}-tab`}
            aria-selected={screen === current}
            aria-controls={`${baseId}-${screen}-panel`}
            tabIndex={screen === current ? 0 : -1}
            onKeyDown={onKey}
            onClick={() => { setCurrent( screen ); }}
          >
            {screenName( screen )}
          </button>
        ) )}
      </div>
      <div role="tabpanel" id={`${baseId}-${current}-panel`} aria-labelledby={`${baseId}-${current}-tab`} className="settings-panel">
        <SettingsFold key={current} screen={current} data={data} onHelp={onHelp} onChanged={reload} />
      </div>
    </div>
  );
}
