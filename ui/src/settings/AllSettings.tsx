// All Settings: every setting in the registry, searchable, filtered by
// tab, by what a change costs, or to the ones changed from their default.
import { useDeferredValue, useId, useMemo, useState } from 'react';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { byScreen, COST_ORDER, filterRows, NO_FILTERS, SCREENS, searchText, type Cost, type Filters, type Row, type Screen } from './model';
import { SettingRow } from './SettingRow';
import { costTag, screenName, settingLabel, settingTip } from './text';
import { useSettings } from './useSettings';

interface AllSettingsProps {
  readonly onHelp: ( anchor: string ) => void;
}

export function AllSettings( { onHelp }: AllSettingsProps ) {
  const { settings, reload } = useSettings();
  const [ filters, setFilters ] = useState<Filters>( NO_FILTERS );
  const query = useDeferredValue( filters.query );
  const searchId = useId();
  const data = settings.state === 'ready' ? settings.data : settings.state === 'loading' ? settings.last : null;

  const texts = useMemo( () => {
    const map = new Map<Row, string>();
    if ( data !== null ) {
      for ( const row of data.rows ) {
        map.set( row, searchText( row, settingLabel( row.entry ), settingTip( row.entry ), data.schema ) );
      }
    }
    return map;
  }, [ data ] );

  const groups = useMemo( () => {
    if ( data === null ) {
      return new Map<Screen, Row[]>();
    }
    const found = filterRows( data.rows, { ...filters, query }, ( row ) => texts.get( row ) ?? '', data.values );
    return byScreen( found );
  }, [ data, filters, query, texts ] );

  if ( data === null ) {
    return (
      <div className="all-settings">
        <p className="settings-wait" role="status">
          {settings.state === 'failed' ? ( settings.error === null ? t( 'settings.request_failed' ) : errorText( settings.error ) )
            : t( 'settings.loading' )}
        </p>
      </div>
    );
  }

  const count = [ ...groups.values() ].reduce( ( sum, rows ) => sum + rows.length, 0 );
  const set = ( change: Partial<Filters> ) => { setFilters( ( previous ) => ( { ...previous, ...change } ) ); };

  return (
    <div className="all-settings">
      <div className="settings-filters" role="search">
        <label htmlFor={searchId} className="visually-hidden">{t( 'settings.search_label' )}</label>
        <input
          id={searchId}
          type="search"
          placeholder={t( 'settings.search_placeholder' )}
          value={filters.query}
          onChange={( event ) => { set( { query: event.target.value } ); }}
        />
        <label className="filter">
          <input type="checkbox" checked={filters.changedOnly} onChange={( event ) => { set( { changedOnly: event.target.checked } ); }} />
          {t( 'settings.filter.changed' )}
        </label>
        <label className="filter">
          {t( 'settings.filter.screen' )}
          <select
            value={filters.screen ?? ''}
            onChange={( event ) => { set( { screen: event.target.value === '' ? null : event.target.value as Screen } ); }}
          >
            <option value="">{t( 'settings.filter.any' )}</option>
            {SCREENS.map( ( screen ) => <option key={screen} value={screen}>{screenName( screen )}</option> )}
          </select>
        </label>
        <label className="filter">
          {t( 'settings.filter.cost' )}
          <select
            value={filters.cost ?? ''}
            onChange={( event ) => { set( { cost: event.target.value === '' ? null : event.target.value as Cost } ); }}
          >
            <option value="">{t( 'settings.filter.any' )}</option>
            {COST_ORDER.map( ( cost ) => <option key={cost} value={cost}>{costTag( cost )}</option> )}
          </select>
        </label>
        <span className="settings-count" role="status">{t( 'settings.count', { count } )}</span>
      </div>
      {count === 0 && <p className="settings-empty">{t( 'settings.none_found' )}</p>}
      {[ ...groups.entries() ].filter( ( [ , rows ] ) => rows.length > 0 ).map( ( [ screen, rows ] ) => (
        <section key={screen} className="settings-group" aria-label={screenName( screen )}>
          <h2>{screenName( screen )}</h2>
          {rows.map( ( row ) => <SettingRow key={row.entry.path} row={row} data={data} onHelp={onHelp} onChanged={reload} /> )}
        </section>
      ) )}
    </div>
  );
}
