// Plays one local video. The file comes from a native dialog (opened by the
// shell, so the page holds no file-system permission), from the library, or
// from --open, which the shell prepares before the page starts so the first
// render plays it. The shell allows that one file and says how to address
// it: a file:// URL on Linux, where WebKitGTK cannot play the asset
// protocol, else a path for the asset protocol.
//
// SCRUBBING (spec section 10). Seeking a long-GOP video while dragging is
// decode-bound in any webview, seconds behind the pointer at 4K. So while
// the scrub track is dragged, the frame from the engine's scrub strip
// nearest at or before the pointer is drawn over the video at once, and the
// video is left alone; release makes one exact seek, and the strip frame
// stays over the video until it lands. (Chasing the pointer with seeks
// under the strip was measured: each 4K long-GOP seek costs about what it
// would save, and release waited for the one in flight.) Until a strip
// exists, the video chases the pointer, one seek in flight at a time, the
// newest position next, as the spike's "latest" did. The keyboard (arrows,
// Home, End, Page Up and Down) seeks at once.
//
// FEEDBACK. Under the controls, on a result only: "missed something" and
// "censored too much" at the time they are pressed (../feedback).
import { convertFileSrc } from '@tauri-apps/api/core';
import { useCallback, useEffect, useRef, useState } from 'react';
import { isEngineError, type EngineError } from '../engine/client';
import { FeedbackPanel } from '../feedback/FeedbackPanel';
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { describeError, log } from '../log';
import { openVideo } from '../library/api';
import { VIDEO_EXTENSIONS } from '../library/names';
import { pickVideo, uiMark, type PlayableVideo } from '../shell';
import { clock } from './clock';
import { frameAt, tileRect, useStrip, type LoadedStrip } from './strip';

function source( video: PlayableVideo ): string {
  return video.file_url ?? convertFileSrc( video.path );
}

function drawFrame( canvas: HTMLCanvasElement | null, strip: LoadedStrip, seconds: number ): boolean {
  const context = canvas?.getContext( '2d' );
  const index = frameAt( strip.frames, seconds );
  if ( canvas === null || context === null || context === undefined || index < 0 ) {
    return false;
  }
  const from = tileRect( strip, index );
  context.drawImage( strip.image, from.x, from.y, from.width, from.height, 0, 0, canvas.width, canvas.height );
  return true;
}

interface DragStats {
  started: number;
  drawn: number;
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
  // A library video to play (a result or a source); nonce changes to play it again.
  request: { path: string; nonce: number } | null;
  // --measure-scrub: sweep the scrub track once per strip pass and log what it showed.
  measureScrub?: boolean;
}

export function Player( { openAtStart, openError, request, measureScrub = false }: PlayerProps ) {
  const [ video, setVideo ] = useState<PlayableVideo | null>( openAtStart );
  const [ autoPlay, setAutoPlay ] = useState( openAtStart !== null );
  // Only --open starts muted: it plays with no one having asked.
  const [ muted, setMuted ] = useState( openAtStart !== null );
  const [ error, setError ] = useState<string | null>( openError === null ? null : errorText( openError ) );
  const element = useRef<HTMLVideoElement>( null );

  useEffect( () => {
    if ( request === null ) {
      return;
    }
    let live = true;
    openVideo( request.path ).then( ( playable ) => {
      if ( live ) {
        setError( null );
        setAutoPlay( true );
        setMuted( false );
        setVideo( playable );
      }
    }, ( reason: unknown ) => {
      log.warn( `open_video failed: ${describeError( reason )}` );
      if ( live ) {
        setError( isEngineError( reason ) ? errorText( reason ) : t( 'player.error.open_failed' ) );
      }
    } );
    return () => { live = false; };
  }, [ request ] );

  useEffect( () => {
    const node = element.current;
    if ( video === null || node === null ) {
      return;
    }
    log.info( `playing ${video.name}` );
    // Where the time to the first frame goes, at debug: milliseconds since
    // the page started loading.
    const steps = [ 'loadstart', 'loadedmetadata', 'loadeddata', 'canplay', 'playing' ];
    const onStep = ( event: Event ) => {
      log.debug( `video ${event.type} at ${Math.round( performance.now() ).toString()} ms` );
    };
    for ( const step of steps ) {
      node.addEventListener( step, onStep, { once: true } );
    }
    node.requestVideoFrameCallback( () => {
      log.debug( `video first frame at ${Math.round( performance.now() ).toString()} ms` );
      void uiMark( 'first_frame' );
    } );
    return () => {
      for ( const step of steps ) {
        node.removeEventListener( step, onStep );
      }
    };
  }, [ video ] );

  const open = () => {
    pickVideo( t( 'player.dialog_title' ), t( 'player.dialog_filter' ), VIDEO_EXTENSIONS ).then(
      ( picked ) => {
        if ( picked !== null ) {
          setError( null );
          setAutoPlay( false );
          setMuted( false );
          setVideo( picked );
        }
      },
      ( reason: unknown ) => {
        log.error( `pick_video failed: ${describeError( reason )}` );
        setError( t( 'player.error.open_failed' ) );
      },
    );
  };

  const [ playing, setPlaying ] = useState( false );
  const [ current, setCurrent ] = useState( 0 );
  const [ duration, setDuration ] = useState( 0 );
  const [ dragTime, setDragTime ] = useState<number | null>( null );
  const canvas = useRef<HTMLCanvasElement>( null );
  const drag = useRef<DragStats | null>( null );
  const released = useRef<number | null>( null );
  // The chase: whether a seek is in flight, and the newest position asked for meanwhile.
  const inFlight = useRef( false );
  const pending = useRef<number | null>( null );
  // Called with the release-to-exact-frame time once the release seek lands.
  const onReleased = useRef<( ( ms: number ) => void ) | null>( null );
  const strip = useStrip( video?.path ?? null );

  const seek = useCallback( ( seconds: number ) => {
    const node = element.current;
    if ( node !== null && Number.isFinite( seconds ) ) {
      node.currentTime = seconds;
    }
  }, [] );

  const chase = useCallback( ( seconds: number ) => {
    if ( inFlight.current ) {
      pending.current = seconds;
      return;
    }
    inFlight.current = true;
    pending.current = null;
    seek( seconds );
  }, [ seek ] );

  // The exact seek on release, at once: it replaces any seek in flight.
  const release = useCallback( ( seconds: number ) => {
    released.current = performance.now();
    pending.current = null;
    inFlight.current = true;
    seek( seconds );
  }, [ seek ] );

  const startDrag = () => {
    drag.current = { started: performance.now(), drawn: 0 };
    setDragTime( current );
  };

  const moveTo = ( seconds: number ) => {
    const stats = drag.current;
    if ( stats === null ) {
      // Keyboard: seek at once.
      setCurrent( seconds );
      seek( seconds );
      return;
    }
    setDragTime( seconds );
    if ( strip !== null && drawFrame( canvas.current, strip, seconds ) ) {
      stats.drawn += 1;
    } else {
      chase( seconds );
    }
  };

  const endDrag = () => {
    const stats = drag.current;
    if ( stats === null ) {
      return;
    }
    drag.current = null;
    const target = dragTime ?? current;
    const elapsed = performance.now() - stats.started;
    log.debug( `scrub: ${String( stats.drawn )} strip frame(s) in ${String( Math.round( elapsed ) )} ms`
      + ` (${( stats.drawn / Math.max( elapsed / 1000, 0.001 ) ).toFixed( 1 )} per s), ${strip === null ? 'no strip' : strip.density}` );
    setCurrent( target );
    release( target );
    // The overlay goes once the exact frame is in (seeked); this is in case no seek event comes.
    window.setTimeout( () => {
      if ( drag.current === null ) {
        setDragTime( null );
      }
    }, 2000 );
  };

  useEffect( () => {
    const node = element.current;
    if ( node === null ) {
      return;
    }
    const onTime = () => {
      if ( drag.current === null ) {
        setCurrent( node.currentTime );
      }
    };
    const onDuration = () => { setDuration( Number.isFinite( node.duration ) ? node.duration : 0 ); };
    const onPlay = () => { setPlaying( true ); };
    const onPause = () => { setPlaying( false ); };
    const onSeeked = () => {
      inFlight.current = false;
      const next = pending.current;
      if ( next !== null ) {
        // The chase moves on to the newest position (or the release's exact one).
        pending.current = null;
        inFlight.current = true;
        node.currentTime = next;
        return;
      }
      if ( released.current !== null ) {
        const ms = performance.now() - released.current;
        log.debug( `scrub: release to exact frame in ${String( Math.round( ms ) )} ms` );
        released.current = null;
        setDragTime( null );
        onReleased.current?.( ms );
        onReleased.current = null;
      }
    };
    node.addEventListener( 'timeupdate', onTime );
    node.addEventListener( 'durationchange', onDuration );
    node.addEventListener( 'loadedmetadata', onDuration );
    node.addEventListener( 'play', onPlay );
    node.addEventListener( 'pause', onPause );
    node.addEventListener( 'seeked', onSeeked );
    return () => {
      node.removeEventListener( 'timeupdate', onTime );
      node.removeEventListener( 'durationchange', onDuration );
      node.removeEventListener( 'loadedmetadata', onDuration );
      node.removeEventListener( 'play', onPlay );
      node.removeEventListener( 'pause', onPause );
      node.removeEventListener( 'seeked', onSeeked );
    };
  }, [ video ] );

  const togglePlay = () => {
    const node = element.current;
    if ( node === null ) {
      return;
    }
    if ( node.paused ) {
      node.play().catch( ( reason: unknown ) => { log.warn( `play failed: ${describeError( reason )}` ); } );
    } else {
      node.pause();
    }
  };

  const fullScreen = () => {
    element.current?.parentElement?.requestFullscreen().catch( ( reason: unknown ) => {
      log.debug( `full screen refused: ${describeError( reason )}` );
    } );
  };

  // The evidence run (--measure-scrub): the spike's drag, 20 s of video swept in
  // 3 s, one move per 16 ms, through the same drawing as a real drag; then
  // the release's exact seek. Logged as one JSON line per strip pass.
  const measured = useRef( new Set<string>() );
  const sweeps = useRef<Promise<void>>( Promise.resolve() );
  useEffect( () => {
    const node = element.current;
    if ( !measureScrub || strip === null || node === null || duration <= 0 || measured.current.has( strip.density ) ) {
      return;
    }
    measured.current.add( strip.density );
    const span = Math.min( 20, duration );
    const start = Math.min( duration * 0.1, duration - span );
    // Each sweep after the first goes the other way, so its release seeks somewhere new.
    const forward = measured.current.size % 2 === 1;
    const steps = Math.round( 3000 / 16 );
    const sleep = ( ms: number ) => new Promise( ( resolve ) => { window.setTimeout( resolve, ms ); } );
    const sweep = async () => {
      // As the spike did: the video paused, so the release seek is measured alone.
      node.pause();
      let shown = -1;
      let changes = 0;
      let lagTotal = 0;
      const began = performance.now();
      for ( let step = 0; step <= steps; step += 1 ) {
        const seconds = forward ? start + ( span * step ) / steps : start + span - ( span * step ) / steps;
        drawFrame( canvas.current, strip, seconds );
        const index = frameAt( strip.frames, seconds );
        if ( index !== shown ) {
          shown = index;
          changes += 1;
        }
        lagTotal += seconds - ( strip.frames[ index ] ?? seconds );
        await sleep( 16 );
      }
      const elapsed = ( performance.now() - began ) / 1000;
      const target = forward ? start + span : start;
      const seeked = new Promise<number>( ( resolve ) => { onReleased.current = resolve; } );
      release( target );
      const releaseMs = await Promise.race( [ seeked, sleep( 10000 ).then( () => -1 ) ] );
      log.info( `scrub-measure ${JSON.stringify( {
        density: strip.density, strip_frames: strip.frames.length, swept_seconds: span, wall_seconds: Number( elapsed.toFixed( 2 ) ),
        frames_shown_per_s: Number( ( changes / elapsed ).toFixed( 1 ) ),
        mean_content_lag_s: Number( ( lagTotal / ( steps + 1 ) ).toFixed( 2 ) ),
        pointer_to_picture_ms: 'one draw per move, same task',
        release_to_exact_frame_ms: Math.round( releaseMs ),
      } )}` );
    };
    sweeps.current = sweeps.current.then( sweep ).catch( ( reason: unknown ) => {
      log.warn( `scrub-measure failed: ${describeError( reason )}` );
    } );
  }, [ measureScrub, strip, duration, release ] );

  const shownTime = dragTime ?? current;
  const showStrip = dragTime !== null && strip !== null;

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
          <div className="player-frame">
            {/* eslint-disable-next-line jsx-a11y-x/media-has-caption -- the user's own files come with no caption track */}
            <video
              key={video.path}
              ref={element}
              src={source( video )}
              autoPlay={autoPlay}
              muted={muted}
              preload="auto"
              onClick={togglePlay}
              onError={( event ) => {
                const code = event.currentTarget.error?.code;
                log.warn( `cannot play ${video.name}: media error ${String( code )}` );
                setError( errorMessage( code ) );
              }}
            />
            <canvas
              ref={canvas}
              className="player-strip"
              hidden={!showStrip}
              width={strip?.tileWidth ?? 160}
              height={strip?.tileHeight ?? 90}
            />
          </div>
        )}
      </div>
      {error === null && video !== null && (
        <div className="player-controls" role="group" aria-label={t( 'player.controls' )}>
          <button type="button" className="primary" onClick={togglePlay}>{playing ? t( 'player.pause' ) : t( 'player.play' )}</button>
          <input
            type="range"
            className="player-scrub"
            min={0}
            max={duration > 0 ? duration : 0}
            step="any" // catalog-ignore: the HTML step token for any position, not text a user sees
            value={Math.min( shownTime, duration )}
            aria-label={t( 'player.position' )}
            aria-valuetext={t( 'player.position_text', { current: clock( shownTime ), total: clock( duration ) } )}
            onPointerDown={startDrag}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onChange={( event ) => { moveTo( Number( event.target.value ) ); }}
          />
          <span className="player-time">{t( 'player.time', { current: clock( shownTime ), total: clock( duration ) } )}</span>
          <button type="button" aria-pressed={muted} onClick={() => { setMuted( !muted ); }}>{muted ? t( 'player.unmute' ) : t( 'player.mute' )}</button>
          <button type="button" onClick={fullScreen}>{t( 'player.full_screen' )}</button>
        </div>
      )}
      {error === null && video !== null && (
        <FeedbackPanel
          key={video.path}
          path={video.path}
          now={() => element.current?.currentTime ?? current}
          pause={() => { element.current?.pause(); }}
          seek={( seconds ) => { setCurrent( seconds ); seek( seconds ); }}
        />
      )}
    </section>
  );
}
