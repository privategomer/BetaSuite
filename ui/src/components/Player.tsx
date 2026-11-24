// Plays one local video. The file comes from a native dialog (opened by the
// shell, so the page holds no file-system permission) or from --open, which
// the shell prepares before the page starts so the first render plays it. The
// shell allows that one file and says how to address it: a file:// URL on
// Linux, where WebKitGTK cannot play the asset protocol, else a path for
// the asset protocol.
import { convertFileSrc } from '@tauri-apps/api/core';
import { useEffect, useRef, useState } from 'react';
import type { EngineError } from '../engine/client';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { describeError, log } from '../log';
import { pickVideo, uiMark, type PlayableVideo } from '../shell';

const VIDEO_EXTENSIONS = [ 'mp4', 'm4v', 'mov', 'mkv', 'webm', 'avi', 'wmv', 'ts', 'mts' ];

function source( video: PlayableVideo ): string {
  return video.file_url ?? convertFileSrc( video.path );
}

function errorMessage( code: number | undefined ): string {
  switch ( code ) {
    case MediaError.MEDIA_ERR_ABORTED:
      return t( 'player.error.aborted' );
    case MediaError.MEDIA_ERR_NETWORK:
      return t( 'player.error.read' );
    case MediaError.MEDIA_ERR_DECODE:
      return t( 'player.error.decode' );
    case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
      return t( 'player.error.unsupported' );
    default:
      return t( 'player.error.unknown' );
  }
}

interface PlayerProps {
  // The --open video, played at once, muted; or the error opening it.
  openAtStart: PlayableVideo | null;
  openError: EngineError | null;
}

export function Player( { openAtStart, openError }: PlayerProps ) {
  const [ video, setVideo ] = useState<PlayableVideo | null>( openAtStart );
  const [ autoPlay, setAutoPlay ] = useState( openAtStart !== null );
  const [ error, setError ] = useState<string | null>( openError === null ? null : errorText( openError ) );
  const element = useRef<HTMLVideoElement>( null );

  useEffect( () => {
    const node = element.current;
    if ( video === null || node === null ) {
      return;
    }
    log.info( `playing ${video.name}` );
    node.requestVideoFrameCallback( () => { void uiMark( 'first_frame' ); } );
  }, [ video ] );

  const open = () => {
    pickVideo( t( 'player.dialog_title' ), t( 'player.dialog_filter' ), VIDEO_EXTENSIONS ).then(
      ( picked ) => {
        if ( picked !== null ) {
          setError( null );
          setAutoPlay( false );
          setVideo( picked );
        }
      },
      ( reason: unknown ) => {
        log.error( `pick_video failed: ${describeError( reason )}` );
        setError( t( 'player.error.open_failed' ) );
      },
    );
  };

  return (
    <section className="player">
      <div className="player-bar">
        <button type="button" className="primary" onClick={open}>{t( 'player.open' )}</button>
        {video !== null && <span className="player-file">{video.name}</span>}
      </div>
      <div className="player-stage">
        {error !== null && <p className="player-error" role="alert">{error}</p>}
        {error === null && video === null && <p className="player-empty">{t( 'player.empty' )}</p>}
        {error === null && video !== null && (
          // eslint-disable-next-line jsx-a11y-x/media-has-caption -- the user's own files come with no caption track
          <video
            key={video.path}
            ref={element}
            src={source( video )}
            controls
            autoPlay={autoPlay}
            muted={autoPlay}
            preload="auto"
            onError={( event ) => {
              const code = event.currentTarget.error?.code;
              log.warn( `cannot play ${video.name}: media error ${String( code )}` );
              setError( errorMessage( code ) );
            }}
          />
        )}
      </div>
    </section>
  );
}
