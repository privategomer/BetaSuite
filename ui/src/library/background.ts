// Running in the background: the tray icon's words, its tooltip as the
// queue moves, its Pause, and a desktop notice when the queue finishes
// while the window is hidden (src-tauri/src/tray.rs).
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { t } from '../i18n/format';
import { describeError, log } from '../log';
import { setJobsHold } from '../unsaved';
import { jobCommand } from './api';
import { onQueueDrained, queueNow, useQueue, type Queue } from './jobs';
import { useEffect } from 'react';

let trayAvailable = false;
let hidden = false;

export function hasTray(): boolean {
  return trayAvailable;
}

function texts() {
  return {
    tooltip: t( 'tray.tooltip_idle' ),
    show: t( 'tray.show' ),
    pause: t( 'tray.pause' ),
    quit: t( 'tray.quit' ),
  };
}

function tooltip( queue: Queue ): string {
  const running = queue.rows.find( ( row ) => row.state === 'running' );
  if ( running !== undefined ) {
    const done = running.counts.processed + running.counts.skipped + running.counts.failed;
    return t( 'tray.tooltip_running', { done, total: running.counts.total } );
  }
  if ( queue.rows.some( ( row ) => row.state === 'paused' && row.restored !== true ) ) {
    return t( 'tray.tooltip_paused' );
  }
  return t( 'tray.tooltip_idle' );
}

// Keep running in the background: the window hides; the tray, a second
// start of the app, or Quit brings it back.
export async function hideToTray(): Promise<void> {
  await invoke( 'hide_to_tray' );
  hidden = true;
}

async function pauseRunning(): Promise<void> {
  const running = queueNow().rows.find( ( row ) => row.state === 'running' );
  if ( running !== undefined ) {
    await jobCommand( 'job.pause', running.job );
  }
}

let started = false;

function start(): void {
  if ( started ) {
    return;
  }
  started = true;
  invoke<boolean>( 'tray_setup', { texts: texts() } ).then( ( available ) => {
    trayAvailable = available;
    log.info( `tray ${available ? 'ready' : 'not available on this desktop'}` );
  }, ( error: unknown ) => { log.warn( `tray_setup failed: ${describeError( error )}` ); } );
  listen( 'tray-pause', () => {
    pauseRunning().catch( ( error: unknown ) => { log.warn( `pause from the tray failed: ${describeError( error )}` ); } );
  } ).catch( ( error: unknown ) => { log.warn( `cannot follow the tray: ${describeError( error )}` ); } );
  document.addEventListener( 'visibilitychange', () => {
    if ( document.visibilityState === 'visible' ) {
      hidden = false;
    }
  } );
  onQueueDrained( ( rows ) => {
    if ( !hidden && document.visibilityState === 'visible' ) {
      return;
    }
    const failed = rows.some( ( row ) => row.state === 'stopped' || row.counts.failed > 0 );
    invoke( 'notify', {
      title: t( 'tray.notice_title' ),
      body: failed ? t( 'tray.notice_problems' ) : t( 'tray.notice_done' ),
    } ).catch( ( error: unknown ) => { log.warn( `notify failed: ${describeError( error )}` ); } );
  } );
}

// Mounted once by the app: keeps the close hold and the tray in step with the queue.
export function useBackground(): void {
  const queue = useQueue();
  useEffect( () => { start(); }, [] );
  useEffect( () => {
    setJobsHold( queue.busy );
    if ( trayAvailable ) {
      invoke( 'tray_status', { tooltip: tooltip( queue ) } ).catch( ( error: unknown ) => {
        log.debug( `tray_status failed: ${describeError( error )}` );
      } );
    }
  }, [ queue ] );
}
