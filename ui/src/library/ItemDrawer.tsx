// One video in detail: where it is, its type and why (with the pin), its
// results with Play and Save as, and its runs. A result has a friendly
// name here and when saved; its keyed file name stays underneath as the
// cache identity (spec section 4).
import { useEffect, useState } from 'react';
import { isEngineError, type EngineError } from '../engine/client';
import { onEvent } from '../engine/events';
import { errorText } from '../i18n/errors';
import { msgKey, t } from '../i18n/format';
import { labelName } from '../i18n/labels';
import { describeError, log } from '../log';
import { getItem, pinType, saveAs, videoTypes, type ItemDetail } from './api';
import { lookName, statusName, typeHint, typeName, typeWhy } from './names';

interface ItemDrawerProps {
  readonly item: string;
  readonly onPlay: ( path: string ) => void;
  readonly onClose: () => void;
}

function stemAndExtension( path: string ): { stem: string; ext: string } {
  const name = path.split( /[\\/]/ ).pop() ?? path;
  const dot = name.lastIndexOf( '.' );
  return dot > 0 ? { stem: name.slice( 0, dot ), ext: name.slice( dot + 1 ) } : { stem: name, ext: 'mp4' };
}

function dateText( seconds: number ): string {
  return new Intl.DateTimeFormat( document.documentElement.lang || undefined, { dateStyle: 'medium', timeStyle: 'short' } )
    .format( new Date( seconds * 1000 ) );
}

export function ItemDrawer( { item, onPlay, onClose }: ItemDrawerProps ) {
  const [ detail, setDetail ] = useState<ItemDetail | null>( null );
  const [ problem, setProblem ] = useState<EngineError | null>( null );
  const [ saved, setSaved ] = useState<string | null>( null );
  const [ generation, setGeneration ] = useState( 0 );
  const [ types, setTypes ] = useState<readonly string[]>( [] );

  useEffect( () => {
    videoTypes().then( setTypes, ( error: unknown ) => { log.warn( `video types: ${describeError( error )}` ); } );
  }, [] );

  useEffect( () => onEvent( 'catalog.changed', ( data ) => {
    const items = data[ 'items' ];
    if ( data[ 'all' ] === true || ( Array.isArray( items ) && items.includes( item ) ) ) {
      setGeneration( ( value ) => value + 1 );
    }
  } ), [ item ] );

  useEffect( () => {
    let live = true;
    getItem( item ).then( ( found ) => {
      if ( live ) {
        setDetail( found );
      }
    }, ( error: unknown ) => {
      log.warn( `catalog.get failed: ${describeError( error )}` );
      if ( live && isEngineError( error ) ) {
        setProblem( error );
      }
    } );
    return () => { live = false; };
  }, [ item, generation ] );

  const fail = ( what: string ) => ( error: unknown ) => {
    log.warn( `${what} failed: ${describeError( error )}` );
    setProblem( isEngineError( error ) ? error : { code: 'unknown', params: {} } );
  };

  const pin = ( value: string ) => {
    setProblem( null );
    pinType( item, value === '' ? null : value ).then( () => { setGeneration( ( current ) => current + 1 ); }, fail( 'catalog.pin_type' ) );
  };

  const source = detail?.locations.find( ( location ) => location.availability === 'present' )?.path ?? null;

  const save = ( output: string ) => {
    const { stem, ext } = stemAndExtension( source ?? detail?.name ?? output );
    const outputExt = stemAndExtension( output ).ext;
    setSaved( null );
    saveAs( output, t( 'library.save_title' ), t( 'library.save_name', { stem, ext: outputExt === '' ? ext : outputExt } ) )
      .then( ( where ) => { setSaved( where ); }, fail( 'save_as' ) );
  };

  return (
    <aside className="item-drawer" aria-labelledby="item-drawer-title">
      <div className="item-drawer-head">
        <h3 id="item-drawer-title">{detail?.name ?? t( 'library.loading' )}</h3>
        <button type="button" onClick={onClose}>{t( 'library.close_details' )}</button>
      </div>
      {problem !== null && <p className="run-problem" role="alert">{errorText( problem )}</p>}
      {detail !== null && (
        <>
          <dl className="item-facts">
            <dt>{t( 'library.col.status' )}</dt>
            <dd>{statusName( detail.status )}</dd>
            <dt>{t( 'library.col.type' )}</dt>
            <dd>
              <span>{typeName( detail.type.profile )}</span>
              <span className="item-why">{typeWhy( detail.type )}</span>
            </dd>
          </dl>

          <label className="item-pin">
            <span>{t( 'library.pin_label' )}</span>
            <select value={detail.pin ?? ''} onChange={( event ) => { pin( event.target.value ); }}>
              <option value="">{t( 'type.auto' )}</option>
              {types.map( ( profile ) => (
                <option key={profile} value={profile} title={typeHint( profile ) ?? undefined}>{typeName( profile )}</option>
              ) )}
            </select>
          </label>

          <h4>{t( 'library.results' )}</h4>
          {detail.output_files.length === 0 && <p>{t( 'library.no_results' )}</p>}
          <ul className="item-results">
            {detail.output_files.map( ( output ) => (
              <li key={output.path}>
                <span className="item-result-name">
                  {t( 'library.result_name', { name: stemAndExtension( detail.name ).stem } )}
                </span>
                {output.present
                  ? (
                    <>
                      <button type="button" className="primary" onClick={() => { onPlay( output.path ); }}>{t( 'library.play' )}</button>
                      <button type="button" onClick={() => { save( output.path ); }}>{t( 'library.save_as' )}</button>
                    </>
                  )
                  : <span>{t( 'library.result_missing' )}</span>}
              </li>
            ) )}
          </ul>
          {saved !== null && <p role="status">{t( 'library.saved', { path: saved } )}</p>}

          {detail.censors.length > 0 && (
            <>
              <h4>{t( 'library.col.censors' )}</h4>
              <ul className="item-censors">
                {detail.censors.map( ( censor ) => (
                  <li key={censor.label}>
                    <span>{labelName( censor.label )}</span>
                    <span className="item-why">{censor.looks.map( lookName ).join( ', ' )}</span>
                  </li>
                ) )}
              </ul>
            </>
          )}

          <h4>{t( 'library.where' )}</h4>
          <ul className="item-locations">
            {detail.locations.map( ( location ) => (
              <li key={location.path}>
                <span className="item-path">{location.path}</span>
                <span className="item-why">{t( msgKey( 'availability.', location.availability ) )}</span>
                {location.availability === 'present' && (
                  <button type="button" onClick={() => { onPlay( location.path ); }}>{t( 'library.play_source' )}</button>
                )}
              </li>
            ) )}
          </ul>

          <h4>{t( 'library.history' )}</h4>
          {detail.history.length === 0 && <p>{t( 'library.no_history' )}</p>}
          <ul className="item-history">
            {detail.history.slice( 0, 10 ).map( ( run ) => (
              <li key={`${String( run.started_at )}-${run.outcome}`}>
                <span>{dateText( run.ended_at )}</span>
                <span>{t( msgKey( 'history.outcome.', run.outcome ) )}</span>
                {run.preview && <span>{t( 'queue.preview' )}</span>}
              </li>
            ) )}
          </ul>
        </>
      )}
    </aside>
  );
}
