// What to hide: the figures (a live preview and a click target), a tile per
// label (the list form and the keyboard path), the art style, and a look
// per label. Both the figures and the tiles drive the same switches.
//
// A change is held as pending until applied: the pictures show it at once
// (preview.look with the pending values as per-run overrides) and the bar
// says what applying it costs (a config.set dry run), as every setting
// does before it is written. The art style is applied straight away,
// since it costs nothing.
import { useCallback, useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { t, hasMessage, msgKey } from '../i18n/format';
import { errorText } from '../i18n/errors';
import { labelName, unavailableText } from '../i18n/labels';
import { describeError, log } from '../log';
import { setValues } from '../settings/api';
import { CostNotice, refusal, type Check } from '../settings/SettingRow';
import { SettingsFold } from '../settings/SettingsFold';
import { useSettings, type Loaded } from '../settings/useSettings';
import { setGuard } from '../unsaved';
import type { PreviewRequest } from './api';
import {
  ART_STYLE_PATH, boxPercent, byPart, DEFAULT_ART_STYLE, dressFor, figureItem, ITEMS_PATH, matchingPreset, neighbour,
  stylePath, tileItem, tileSex, withLabel, type Dress, type LabelInfo, type Preset, type Sex, type View,
} from './model';
import { usePreviews, type Drawn } from './usePreviews';

const DRY_RUN_DELAY_MS = 300;
const VIEWS: readonly View[] = [ 'front', 'back' ];
const CUSTOM = '';

function presetName( preset: Preset ): string {
  if ( preset.name !== undefined ) {
    return preset.name;
  }
  const key = msgKey( 'look.preset.', preset.id );
  return hasMessage( key ) ? t( key ) : preset.id;
}

function styleName( id: string ): string {
  const key = msgKey( 'art_style.', id );
  return hasMessage( key ) ? t( key ) : id;
}

function partName( part: string ): string {
  const key = msgKey( 'hide.part.', part );
  return hasMessage( key ) ? t( key ) : part;
}

// The labels the engine describes; from an older engine, just their ids.
function labelsOf( data: Loaded ): LabelInfo[] {
  if ( data.schema.labels !== undefined ) {
    return [ ...data.schema.labels ];
  }
  return ( data.schema.placeholders[ '<label>' ] ?? [] ).map( ( id ) => ( { id, part: 'other', state: 'none', sex: 'any', view: 'front' } ) );
}

function stringList( value: unknown ): string[] {
  return Array.isArray( value ) ? value.filter( ( item ): item is string => typeof item === 'string' ) : [];
}

interface Toggle<T extends string> {
  readonly legend: string;
  readonly value: T;
  readonly options: readonly { readonly id: T; readonly label: string }[];
  readonly onChange: ( value: T ) => void;
}

function Segmented<T extends string>( { legend, value, options, onChange }: Toggle<T> ) {
  const name = useId();
  return (
    <fieldset className="segmented">
      <legend>{legend}</legend>
      {options.map( ( option ) => (
        <label key={option.id} className={option.id === value ? 'selected' : undefined}>
          <input type="radio" name={name} value={option.id} checked={option.id === value} onChange={() => { onChange( option.id ); }} />
          {option.label}
        </label>
      ) )}
    </fieldset>
  );
}

interface FigureProps {
  readonly view: View;
  readonly drawn: Drawn | undefined;
  readonly alt: string;
  readonly hidden: ReadonlySet<string>;
  readonly available: ( label: string ) => boolean;
  readonly focused: string | null;
  readonly onToggle: ( label: string ) => void;
}

function Figure( { view, drawn, alt, hidden, available, focused, onToggle }: FigureProps ) {
  return (
    <figure className="hide-figure" data-view={view}>
      <div className="hide-figure-frame">
        {drawn === undefined ? <div className="hide-figure-placeholder" /> : <img src={drawn.url} alt={alt} />}
        {drawn?.result.boxes.map( ( box ) => (
          // The tiles are the keyboard path; these spots are for the pointer.
          <button
            key={`${box.label}-${String( box.x )}-${String( box.y )}`}
            type="button"
            tabIndex={-1}
            className={`hotspot${hidden.has( box.label ) ? ' on' : ''}${focused === box.label ? ' focused' : ''}`}
            style={boxPercent( box, drawn.result.width, drawn.result.height )}
            disabled={!available( box.label )}
            aria-pressed={hidden.has( box.label )}
            aria-label={t( 'hide.figure.spot', { label: labelName( box.label ) } )}
            onClick={() => { onToggle( box.label ); }}
          />
        ) )}
      </div>
      <figcaption>{view === 'front' ? t( 'hide.figure.front' ) : t( 'hide.figure.back' )}</figcaption>
    </figure>
  );
}

interface TileProps {
  readonly label: LabelInfo;
  readonly index: number;
  readonly active: boolean;
  readonly drawn: Drawn | undefined;
  readonly missing: boolean;
  readonly on: boolean;
  readonly style: unknown;
  readonly presets: readonly Preset[];
  readonly unavailable: { code: string; params: Readonly<Record<string, unknown>> } | undefined;
  readonly onToggle: () => void;
  readonly onPreset: ( preset: Preset ) => void;
  readonly onFocusTile: ( index: number ) => void;
  readonly onKey: ( event: KeyboardEvent<HTMLInputElement> ) => void;
}

function Tile( { label, index, active, drawn, missing, on, style, presets, unavailable, onToggle, onPreset, onFocusTile, onKey }: TileProps ) {
  const id = useId();
  const name = labelName( label.id );
  const preset = matchingPreset( style, presets );
  return (
    <div className={`hide-tile${on ? ' on' : ''}${unavailable !== undefined ? ' unavailable' : ''}`} data-label={label.id}>
      <div className="hide-tile-art">
        {drawn !== undefined && !missing
          ? <img src={drawn.url} alt={t( 'hide.tile.alt', { label: name } )} />
          : <div className="hide-tile-placeholder">{missing ? t( 'hide.tile.no_art' ) : null}</div>}
      </div>
      <label className="hide-tile-switch">
        <input
          type="checkbox"
          data-tile-index={index}
          tabIndex={active ? 0 : -1}
          checked={on}
          // Not disabled: a disabled control cannot take focus, and the
          // keyboard path must reach the tile to hear why it is off.
          aria-disabled={unavailable !== undefined}
          aria-describedby={unavailable !== undefined ? `${id}-why` : undefined}
          aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Home End"
          onChange={onToggle}
          onFocus={() => { onFocusTile( index ); }}
          onKeyDown={onKey}
        />
        <span>{name}</span>
      </label>
      <label className="hide-tile-look">
        <span>{t( 'hide.tile.look' )}</span>
        <select
          tabIndex={active ? 0 : -1}
          value={preset?.id ?? CUSTOM}
          aria-disabled={unavailable !== undefined}
          aria-describedby={unavailable !== undefined ? `${id}-why` : undefined}
          onFocus={() => { onFocusTile( index ); }}
          onChange={( event ) => {
            if ( unavailable !== undefined ) {
              return;
            }
            const chosen = presets.find( ( item ) => item.id === event.target.value );
            if ( chosen !== undefined ) {
              onPreset( chosen );
            }
          }}
        >
          {presets.map( ( item ) => <option key={item.id} value={item.id}>{presetName( item )}</option> )}
          {preset === null && <option value={CUSTOM} disabled>{t( 'hide.look.custom' )}</option>}
        </select>
      </label>
      {unavailable !== undefined && <p id={`${id}-why`} className="hide-tile-note">{unavailableText( unavailable )}</p>}
    </div>
  );
}

interface ScreenProps {
  readonly data: Loaded;
  readonly onHelp: ( anchor: string ) => void;
  readonly reload: () => void;
}

function Screen( { data, onHelp, reload }: ScreenProps ) {
  const labels = useMemo( () => labelsOf( data ), [ data ] );
  const presets = useMemo( () => [ ...( data.schema.presets ?? [] ) ], [ data ] );
  const groups = useMemo( () => byPart( labels ), [ labels ] );
  // Tiles in the order they are shown, which the arrow keys follow.
  const ordered = useMemo( () => groups.flatMap( ( group ) => group.labels ), [ groups ] );
  const position = useMemo( () => new Map( ordered.map( ( label, at ) => [ label.id, at ] ) ), [ ordered ] );
  const styleChoices = useMemo( () => {
    const entry = data.schema.settings.find( ( setting ) => setting.path === ART_STYLE_PATH );
    return stringList( entry?.choices );
  }, [ data ] );
  const saved = data.values.get( ART_STYLE_PATH )?.value;
  const [ artStyle, setArtStyle ] = useState( typeof saved === 'string' ? saved : DEFAULT_ART_STYLE );
  const [ sex, setSex ] = useState<Sex>( 'female' );
  const [ dress, setDress ] = useState<Dress>( 'unclothed' );
  const [ pending, setPending ] = useState<Readonly<Record<string, unknown>>>( {} );
  const [ active, setActive ] = useState( 0 );
  const [ focused, setFocused ] = useState<string | null>( null );
  const [ answer, setAnswer ] = useState<{ readonly to: Readonly<Record<string, unknown>>; readonly check: Check } | null>( null );
  const [ applying, setApplying ] = useState( false );
  const grid = useRef<HTMLDivElement>( null );

  const items = stringList( ITEMS_PATH in pending ? pending[ ITEMS_PATH ] : data.values.get( ITEMS_PATH )?.value );
  const hidden = useMemo( () => new Set( items ), [ items ] );
  const view = ( label: string ) => data.labels.get( label );
  const available = useCallback( ( label: string ) => data.labels.get( label )?.available !== false, [ data ] );
  const styleOf = ( label: string ) => ( stylePath( label ) in pending ? pending[ stylePath( label ) ] : view( label )?.fields[ 'censor_style' ]?.value );

  const requests = useMemo( () => {
    const out = new Map<string, PreviewRequest>();
    for ( const side of VIEWS ) {
      out.set( `figure.${side}`, { art: { style: artStyle, item: figureItem( sex, side, dress ) } } );
    }
    for ( const label of labels ) {
      out.set( `tile.${label.id}`, { art: { style: artStyle, item: tileItem( label, sex ) }, labels: [ label.id ] } );
    }
    for ( const choice of styleChoices ) {
      out.set( `style.${choice}`, { art: { style: choice, item: figureItem( sex, 'front', dress ) } } );
    }
    return out;
  }, [ artStyle, sex, dress, labels, styleChoices ] );
  const previews = usePreviews( requests, pending, true );

  // What applying the pending changes would cost, asked as a dry run.
  useEffect( () => {
    if ( Object.keys( pending ).length === 0 ) {
      return undefined;
    }
    let live = true;
    const timer = window.setTimeout( () => {
      setValues( pending, true ).then(
        ( change ) => { if ( live ) { setAnswer( { to: pending, check: { state: 'ok', change } } ); } },
        ( error: unknown ) => { if ( live ) { setAnswer( { to: pending, check: refusal( error ) } ); } },
      );
    }, DRY_RUN_DELAY_MS );
    return () => { live = false; window.clearTimeout( timer ); };
  }, [ pending ] );
  // Counted in parts: each label switched, and each look changed.
  const savedItems = new Set( stringList( data.values.get( ITEMS_PATH )?.value ) );
  const pendingCount = Object.keys( pending ).filter( ( path ) => path !== ITEMS_PATH ).length
    + ( ITEMS_PATH in pending ? [ ...new Set( [ ...savedItems, ...items ] ) ].filter( ( label ) => savedItems.has( label ) !== hidden.has( label ) ).length : 0 );
  const check: Check = useMemo( () => ( pendingCount === 0 ? { state: 'idle' }
    : answer?.to === pending ? answer.check : { state: 'checking' } ), [ pendingCount, answer, pending ] );

  const change = ( path: string, value: unknown ) => {
    setPending( ( previous ) => {
      const next = { ...previous, [ path ]: value };
      const held = path === ITEMS_PATH ? data.values.get( ITEMS_PATH )?.value : view( path.split( '.' )[ 1 ] ?? '' )?.fields[ 'censor_style' ]?.value;
      if ( JSON.stringify( held ) === JSON.stringify( value ) ) {
        // Back to what is saved: nothing pending for this path.
        return Object.fromEntries( Object.entries( next ).filter( ( [ key ] ) => key !== path ) );
      }
      return next;
    } );
  };
  const toggle = ( label: string ) => {
    if ( !available( label ) ) {
      return;
    }
    log.info( `what to hide: ${label} ${hidden.has( label ) ? 'off' : 'on'} (pending)` );
    change( ITEMS_PATH, withLabel( items, label, !hidden.has( label ) ) );
  };
  const pickPreset = ( label: string, preset: Preset ) => {
    log.info( `what to hide: ${label} look ${preset.id} (pending)` );
    change( stylePath( label ), preset.style );
  };
  // Resolves true once written; false when the engine refused it.
  const apply = useCallback( (): Promise<boolean> => {
    setApplying( true );
    log.info( `what to hide: applying ${Object.keys( pending ).join( ', ' )}` );
    return setValues( pending, false ).then(
      () => { setPending( {} ); reload(); return true; },
      ( error: unknown ) => { setAnswer( { to: pending, check: refusal( error ) } ); return false; },
    ).finally( () => { setApplying( false ); } );
  }, [ pending, reload ] );
  const discard = useCallback( () => { setPending( {} ); }, [] );

  // While changes are pending, leaving the screen or closing the window
  // asks first (App, unsaved.ts).
  useEffect( () => {
    setGuard( pendingCount > 0 ? { count: pendingCount, check, apply, discard } : null );
  }, [ pendingCount, check, apply, discard ] );
  useEffect( () => () => { setGuard( null ); }, [] );
  const chooseStyle = ( choice: string ) => {
    setArtStyle( choice );
    log.info( `art style: ${choice}` );
    // Costs nothing (no cache key), so it is written straight away.
    setValues( { [ ART_STYLE_PATH ]: choice }, false ).catch( ( error: unknown ) => {
      log.warn( `art style not saved: ${describeError( error )}` );
    } );
  };

  // Focusing a tile shows the figure that has its part and marks the part.
  const focusTile = ( index: number ) => {
    const label = ordered[ index ];
    if ( label === undefined ) {
      return;
    }
    setActive( index );
    setFocused( label.id );
    setDress( ( current ) => dressFor( label, current ) );
    setSex( ( current ) => tileSex( label, current ) );
  };
  const moveTo = ( index: number ) => {
    const target = grid.current?.querySelector<HTMLInputElement>( `input[data-tile-index="${String( index )}"]` );
    target?.focus();
  };
  const onTileKey = ( event: KeyboardEvent<HTMLInputElement> ) => {
    const from = Number( event.currentTarget.dataset[ 'tileIndex' ] );
    const last = ordered.length - 1;
    let to: number;
    switch ( event.key ) {
      case 'ArrowRight': to = Math.min( from + 1, last ); break;
      case 'ArrowLeft': to = Math.max( from - 1, 0 ); break;
      case 'Home': to = 0; break;
      case 'End': to = last; break;
      case 'ArrowDown':
      case 'ArrowUp': {
        const rects = [ ...( grid.current?.querySelectorAll<HTMLElement>( '.hide-tile' ) ?? [] ) ].map( ( tile ) => tile.getBoundingClientRect() );
        to = neighbour( rects, from, event.key === 'ArrowDown' ? 'down' : 'up' );
        break;
      }
      default: return;
    }
    event.preventDefault();
    moveTo( to );
  };

  const busy = previews.error !== null && previews.error !== 'unknown' && previews.error.code === 'config.busy';
  const figureAlt = ( side: View ) => t( 'hide.figure.alt', { sex, view: side, dress } );

  return (
    <div className="hide-screen">
      <p className="view-intro">{t( 'hide.intro' )}</p>

      <fieldset className="art-styles">
        <legend>{t( 'hide.style.legend' )}</legend>
        {styleChoices.map( ( choice ) => {
          const thumb = previews.drawn.get( `style.${choice}` );
          return (
            <label key={choice} className={`art-style${choice === artStyle ? ' selected' : ''}`}>
              <input type="radio" name="art-style" value={choice} checked={choice === artStyle} onChange={() => { chooseStyle( choice ); }} />
              {thumb !== undefined ? <img src={thumb.url} alt="" /> : <span className="art-style-placeholder" />}
              <span>{styleName( choice )}</span>
            </label>
          );
        } )}
      </fieldset>

      {( busy || previews.error !== null ) && (
        <p className="hide-notice" role="status">
          {busy ? t( 'hide.preview.busy' ) : previews.error === 'unknown' ? t( 'hide.preview.failed' ) : errorText( previews.error )}
        </p>
      )}

      <div className="hide-layout">
        <section className="hide-figures" aria-labelledby="hide-figures-title">
          <h3 id="hide-figures-title" className="visually-hidden">{t( 'hide.figure.title' )}</h3>
          <div className="hide-figure-controls">
            <Segmented legend={t( 'hide.figure.sex' )} value={sex} onChange={setSex}
              options={[ { id: 'female', label: t( 'hide.figure.female' ) }, { id: 'male', label: t( 'hide.figure.male' ) } ]} />
            <Segmented legend={t( 'hide.figure.dress' )} value={dress} onChange={setDress}
              options={[ { id: 'unclothed', label: t( 'hide.figure.unclothed' ) }, { id: 'clothed', label: t( 'hide.figure.clothed' ) } ]} />
          </div>
          <div className="hide-figure-pair" aria-busy={previews.busy}>
            {VIEWS.map( ( side ) => (
              <Figure key={side} view={side} drawn={previews.drawn.get( `figure.${side}` )} alt={figureAlt( side )}
                hidden={hidden} available={available} focused={focused} onToggle={toggle} />
            ) )}
          </div>
          <p className="hide-hint">{t( 'hide.keyboard_hint' )}</p>
        </section>

        <section className="hide-tiles" aria-labelledby="hide-tiles-title">
          <h3 id="hide-tiles-title">{t( 'hide.tiles.title' )}</h3>
          <div ref={grid}>
            {groups.map( ( group ) => (
              <div key={group.part} className="hide-group" role="group" aria-label={partName( group.part )}>
                <h4>{partName( group.part )}</h4>
                <div className="hide-grid">
                  {group.labels.map( ( label ) => {
                    const index = position.get( label.id ) ?? 0;
                    const labelView = view( label.id );
                    return (
                      <Tile
                        key={label.id}
                        label={label}
                        index={index}
                        active={index === active}
                        drawn={previews.drawn.get( `tile.${label.id}` )}
                        missing={previews.missing.has( `tile.${label.id}` )}
                        on={hidden.has( label.id )}
                        style={styleOf( label.id )}
                        presets={presets}
                        unavailable={labelView?.available === false ? labelView.unavailable ?? { code: 'other', params: {} } : undefined}
                        onToggle={() => { toggle( label.id ); }}
                        onPreset={( preset ) => { pickPreset( label.id, preset ); }}
                        onFocusTile={focusTile}
                        onKey={onTileKey}
                      />
                    );
                  } )}
                </div>
              </div>
            ) )}
          </div>
        </section>
      </div>

      {pendingCount > 0 && (
        <div className="hide-pending" role="region" aria-label={t( 'hide.pending.region' )}>
          <span>{t( 'hide.pending.count', { count: pendingCount } )}</span>
          <CostNotice check={check} />
          <button type="button" className="primary" disabled={applying || check.state !== 'ok'} onClick={() => { void apply(); }}>{t( 'hide.pending.apply' )}</button>
          <button type="button" disabled={applying} onClick={discard}>{t( 'hide.pending.discard' )}</button>
        </div>
      )}

      <section className="hide-more" aria-labelledby="hide-more-title">
        <h3 id="hide-more-title">{t( 'hide.more.title' )}</h3>
        <SettingsFold screen="what_to_hide" data={data} onHelp={onHelp} onChanged={reload} />
      </section>
    </div>
  );
}

export function WhatToHide( { onHelp }: { onHelp: ( anchor: string ) => void } ) {
  const { settings, reload } = useSettings();
  const data = settings.state === 'ready' ? settings.data : settings.state === 'loading' ? settings.last : null;
  return (
    <section className="view hide-view" aria-labelledby="hide-title">
      <h2 id="hide-title" className="hide-title">{t( 'hide.title' )}</h2>
      {data === null
        ? (
          <p className="settings-wait" role="status">
            {settings.state === 'failed' ? ( settings.error === null ? t( 'settings.request_failed' ) : errorText( settings.error ) )
              : t( 'settings.loading' )}
          </p>
        )
        : <Screen data={data} onHelp={onHelp} reload={reload} />}
    </section>
  );
}
