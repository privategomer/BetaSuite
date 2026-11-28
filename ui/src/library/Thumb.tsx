// A library item's thumbnail under the privacy rule (spec section 3): the
// source's blurred thumbnail, and its clear frame only while the pointer or
// focus is on it. The clear frame is asked of the engine each time, kept in
// a data: URL in this component's state and dropped on leave: it never goes
// through the HTTP or asset cache, so nothing of it reaches the disk.
import { useEffect, useRef, useState } from 'react';
import { t } from '../i18n/format';
import { describeError, log } from '../log';
import { dataUrl, thumbnail } from './api';

interface ThumbProps {
  readonly item: string;
  readonly name: string;
  // Changes when the item changes, so the blurred one is asked again.
  readonly version: number;
}

export function Thumb( { item, name, version }: ThumbProps ) {
  const [ blurred, setBlurred ] = useState<string | null>( null );
  const [ clear, setClear ] = useState<string | null>( null );
  const wanted = useRef( false );

  useEffect( () => {
    let live = true;
    thumbnail( item, false ).then( ( thumb ) => {
      if ( live ) {
        setBlurred( dataUrl( thumb ) );
      }
    }, ( error: unknown ) => { log.debug( `thumbnail of ${item}: ${describeError( error )}` ); } );
    return () => { live = false; };
  }, [ item, version ] );

  const reveal = () => {
    wanted.current = true;
    thumbnail( item, true ).then( ( thumb ) => {
      if ( wanted.current ) {
        setClear( dataUrl( thumb ) );
      }
    }, ( error: unknown ) => { log.debug( `reveal of ${item}: ${describeError( error )}` ); } );
  };
  const hide = () => {
    wanted.current = false;
    setClear( null );
  };

  const shown = clear ?? blurred;
  return (
    <button
      type="button"
      className={`thumb${clear === null ? ' thumb-blurred' : ''}`}
      aria-label={t( 'library.thumb_label', { name } )}
      aria-pressed={clear !== null}
      onPointerEnter={reveal}
      onPointerLeave={hide}
      onFocus={reveal}
      onBlur={hide}
      onClick={() => { if ( clear === null ) { reveal(); } else { hide(); } }}
    >
      {shown !== null && <img src={shown} alt="" draggable={false} />}
    </button>
  );
}
