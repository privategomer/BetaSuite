// The help text: the reference document, bundled into the page so help
// works offline. A locale's own copy (i18n/<locale>/help/) wins over the
// English one at the repository root. Rendered with raw HTML turned off,
// so the document can only ever produce markdown's own elements.
import markdownIt, { type MarkdownIt } from 'markdown-it';
import reference from '../../../CONFIG_REFERENCE.md?raw';
import { anchorHeadings } from './anchors';

const localized = import.meta.glob<string>( '../../../i18n/*/help/CONFIG_REFERENCE.md', {
  query: '?raw',
  import: 'default',
  eager: true,
} );

export function helpSource( locale: string ): string {
  return localized[ `../../../i18n/${locale}/help/CONFIG_REFERENCE.md` ] ?? reference;
}

// The page's own markdown parser: every heading gets the anchor the
// registry's help keys point at.
export function helpParser(): MarkdownIt {
  const parser = markdownIt( { html: false, linkify: false, typographer: false } );
  parser.core.ruler.push( 'heading_anchors', ( state ) => { anchorHeadings( state.tokens ); } );
  return parser;
}

const rendered = new Map<string, string>();

export function helpHtml( locale: string ): string {
  let html = rendered.get( locale );
  if ( html === undefined ) {
    html = helpParser().render( helpSource( locale ) );
    rendered.set( locale, html );
  }
  return html;
}
