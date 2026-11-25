// The help panel: the reference document beside the page, opened at the
// section a setting's "Learn more" points to. Everything is bundled, so it
// works offline. Links inside the document move within the panel; links
// out of it are not followed (the app opens no web pages).
import { useEffect, useMemo, useRef, type MouseEvent } from 'react';
import { activeLocale, t } from '../i18n/format';
import { log } from '../log';
import { helpHtml } from './render';

interface HelpPanelProps {
  readonly anchor: string;
  readonly onClose: () => void;
}

export function HelpPanel( { anchor, onClose }: HelpPanelProps ) {
  const body = useRef<HTMLDivElement>( null );
  const close = useRef<HTMLButtonElement>( null );
  const html = useMemo( () => helpHtml( activeLocale() ), [] );

  // Focus comes to the panel when it opens and goes back where it was.
  useEffect( () => {
    const before = document.activeElement;
    close.current?.focus();
    return () => {
      if ( before instanceof HTMLElement ) {
        before.focus();
      }
    };
  }, [] );

  useEffect( () => {
    const target = body.current?.querySelector( `#${CSS.escape( anchor )}` );
    if ( target === null || target === undefined ) {
      log.warn( `help has no section ${anchor}` );
      body.current?.scrollTo( { top: 0 } );
      return;
    }
    target.scrollIntoView( { block: 'start' } );
  }, [ anchor ] );

  useEffect( () => {
    const onKey = ( event: KeyboardEvent ) => {
      if ( event.key === 'Escape' ) {
        onClose();
      }
    };
    window.addEventListener( 'keydown', onKey );
    return () => { window.removeEventListener( 'keydown', onKey ); };
  }, [ onClose ] );

  const onClick = ( event: MouseEvent ) => {
    const link = event.target instanceof Element ? event.target.closest( 'a' ) : null;
    if ( link === null ) {
      return;
    }
    event.preventDefault();
    const href = link.getAttribute( 'href' ) ?? '';
    if ( href.startsWith( '#' ) ) {
      body.current?.querySelector( `#${CSS.escape( decodeURIComponent( href.slice( 1 ) ) )}` )?.scrollIntoView( { block: 'start' } );
    } else {
      log.info( `help link not followed: ${href}` );
    }
  };

  return (
    <aside className="help-panel" aria-labelledby="help-title">
      <div className="help-head">
        <h2 id="help-title">{t( 'help.title' )}</h2>
        <button ref={close} type="button" onClick={onClose}>{t( 'help.close' )}</button>
      </div>
      {/* The document's own markdown, rendered with raw HTML off (render.ts). */}
      {/* eslint-disable-next-line jsx-a11y-x/no-static-element-interactions, jsx-a11y-x/click-events-have-key-events -- links inside handle the keyboard themselves; this only catches their clicks */}
      <div ref={body} className="help-body" onClick={onClick} dangerouslySetInnerHTML={{ __html: html }} />
    </aside>
  );
}
