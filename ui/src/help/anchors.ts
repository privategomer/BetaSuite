// Heading anchors for the help panel, the way GitHub makes them and the
// engine checks them (betautils_settings.heading_anchors): the heading's
// text lowercased, everything but letters, digits, underscores, hyphens and
// spaces dropped, spaces made hyphens, and repeats numbered -1, -2.
// No imports, so scripts/check-settings-reach.mjs runs this same file.

export function slugify( text: string ): string {
  return text.trim().toLowerCase().replace( /[^\p{L}\p{N}_\- ]/gu, '' ).replaceAll( ' ', '-' );
}

// Hands out anchors in document order, numbering repeats.
export class Slugger {
  private readonly seen = new Map<string, number>();

  slug( text: string ): string {
    const base = slugify( text );
    const count = this.seen.get( base ) ?? 0;
    this.seen.set( base, count + 1 );
    return count === 0 ? base : `${base}-${String( count )}`;
  }
}

// The few token fields anchorHeadings reads, so this file needs no imports.
interface HeadingToken {
  readonly type: string;
  readonly content: string;
  attrSet: ( name: string, value: string ) => void;
}

// Gives every heading in a parsed document its anchor as an id, in order.
// The page's parser and the reach check both run this.
export function anchorHeadings( tokens: readonly HeadingToken[] ): string[] {
  const slugger = new Slugger();
  const ids: string[] = [];
  tokens.forEach( ( token, index ) => {
    if ( token.type === 'heading_open' ) {
      const id = slugger.slug( tokens[ index + 1 ]?.content ?? '' );
      token.attrSet( 'id', id );
      ids.push( id );
    }
  } );
  return ids;
}
