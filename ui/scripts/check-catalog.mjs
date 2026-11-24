// The catalog check: every string the UI shows comes from the message
// catalog, and the catalog is sound.
//
//   node scripts/check-catalog.mjs [--log-level L] [--console-level L] [--log-file PATH]
//
// No input needed. Exits 0 when clean, 1 with one line per problem. Rules:
//
//   catalog.shape         a catalog file is not a flat map of strings
//   catalog.key           a key is not dotted lower case (a.b, a.b_c.d)
//   catalog.duplicate     a key is in two files of one locale
//   catalog.icu           a message is not valid ICU MessageFormat
//   catalog.plural_other  a plural or select has no "other" case
//   catalog.em_dash       a message contains an em dash
//   catalog.locale_extra  a translation has a key English does not
//   catalog.unused        a key in an app-owned file (ui.json) is never used
//   errors.missing        an error code has no error.<code> message
//   errors.args           an error message uses a value its code does not carry
//   errors.orphan         an error.<code> message has no code
//   source.literal        text in the source that a user could see
//   source.key_unknown    t(), msg() or <Msg id> names a key that does not exist
//   source.args           the values given do not match the message's arguments
//   source.key_dynamic    msgKey() has no literal prefix, or one no key starts with
//   css.physical          a left or right property (use logical properties)
//   css.content           text in a CSS content value
//   html.text             text in index.html
//   ignore.reason         a catalog-ignore comment without a reason
//
// A line can be exempted with `// catalog-ignore: <reason>` on it or on the
// line before; each exemption is counted in the output.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse as parseIcu, TYPE } from '@formatjs/icu-messageformat-parser';
import ts from 'typescript';
import { createLogger, parseLogArgs } from './lib/log.mjs';

const BASE_LOCALE = 'en-US';
const KEY_PATTERN = /^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/;
const EM_DASH = '\u2014';
const LETTER = /\p{L}/u;

// JSX attributes that never carry text for the user.
const CODE_ATTRIBUTES = new Set( [
  'accept', 'autoComplete', 'autoPlay', 'className', 'controls', 'crossOrigin', 'cx', 'cy', 'd', 'dir',
  'draggable', 'fill', 'focusable', 'form', 'height', 'hidden', 'href', 'htmlFor', 'id', 'inputMode', 'key',
  'lang', 'loop', 'method', 'muted', 'name', 'playsInline', 'points', 'poster', 'preload', 'r', 'ref', 'rel',
  'role', 'rx', 'ry', 'spellCheck', 'src', 'stroke', 'strokeLinecap', 'strokeLinejoin', 'strokeWidth',
  'tabIndex', 'target', 'transform', 'type', 'viewBox', 'width', 'x', 'xmlns', 'y',
] );
const TEXT_ARIA = new Set( [ 'aria-label', 'aria-description', 'aria-valuetext', 'aria-roledescription', 'aria-placeholder' ] );

// Single capitalized words that are DOM tokens, not text.
const DOM_TOKENS = new Set( [ 'Backspace', 'Delete', 'End', 'Enter', 'Escape', 'Home', 'Insert', 'Tab' ] );

const PHYSICAL_CSS = [
  /(?<![\w-])(?:margin|padding)-(?:left|right)\s*:/,
  /(?<![\w-])border-(?:left|right)(?:-[a-z]+)?\s*:/,
  /(?<![\w-])border-(?:top|bottom)-(?:left|right)-radius\s*:/,
  /(?<![\w-])(?:left|right)\s*:/,
  /(?<![\w-])(?:text-align|float|clear)\s*:\s*(?:left|right)\b/,
];

function lineOf( text, index ) {
  return index < 0 ? 1 : text.slice( 0, index ).split( '\n' ).length;
}

function walk( dir, accept ) {
  if ( !fs.existsSync( dir ) ) {
    return [];
  }
  const found = [];
  for ( const entry of fs.readdirSync( dir, { withFileTypes: true } ) ) {
    const full = path.join( dir, entry.name );
    if ( entry.isDirectory() ) {
      found.push( ...walk( full, accept ) );
    } else if ( accept( entry.name ) ) {
      found.push( full );
    }
  }
  return found.sort();
}

function messageArgs( elements, out = new Set() ) {
  for ( const element of elements ) {
    switch ( element.type ) {
      case TYPE.argument:
      case TYPE.number:
      case TYPE.date:
      case TYPE.time:
        out.add( element.value );
        break;
      case TYPE.select:
      case TYPE.plural:
        out.add( element.value );
        for ( const option of Object.values( element.options ) ) {
          messageArgs( option.value, out );
        }
        break;
      case TYPE.tag:
        out.add( element.value );
        messageArgs( element.children, out );
        break;
      default:
        break;
    }
  }
  return out;
}

// --- catalogs ---------------------------------------------------------------

function loadCatalogs( i18nDir, report ) {
  const locales = new Map();
  if ( !fs.existsSync( i18nDir ) ) {
    report( 'catalog.shape', i18nDir, 1, 'no catalog folder' );
    return locales;
  }
  for ( const locale of fs.readdirSync( i18nDir ).sort() ) {
    const dir = path.join( i18nDir, locale );
    if ( !fs.statSync( dir ).isDirectory() ) {
      continue;
    }
    const messages = new Map();
    locales.set( locale, messages );
    for ( const file of walk( dir, ( name ) => name.endsWith( '.json' ) ) ) {
      const text = fs.readFileSync( file, 'utf8' );
      let data;
      try {
        data = JSON.parse( text );
      } catch ( error ) {
        report( 'catalog.shape', file, 1, `not JSON: ${error.message}` );
        continue;
      }
      if ( data === null || typeof data !== 'object' || Array.isArray( data ) ) {
        report( 'catalog.shape', file, 1, 'not an object of key to message' );
        continue;
      }
      for ( const [ key, message ] of Object.entries( data ) ) {
        const line = lineOf( text, text.indexOf( JSON.stringify( key ) ) );
        if ( typeof message !== 'string' ) {
          report( 'catalog.shape', file, line, `${key}: the message is not a string` );
          continue;
        }
        if ( !KEY_PATTERN.test( key ) ) {
          report( 'catalog.key', file, line, `${key}: keys are dotted lower case, like area.name` );
        }
        if ( messages.has( key ) ) {
          report( 'catalog.duplicate', file, line, `${key} is also in ${messages.get( key ).file}` );
          continue;
        }
        if ( message.includes( EM_DASH ) ) {
          report( 'catalog.em_dash', file, line, `${key} contains an em dash` );
        }
        let args = new Set();
        try {
          args = messageArgs( parseIcu( message, { requiresOtherClause: true } ) );
        } catch ( error ) {
          const rule = error.message === 'MISSING_OTHER_CLAUSE' ? 'catalog.plural_other' : 'catalog.icu';
          report( rule, file, line, `${key}: ${error.message}` );
        }
        messages.set( key, { message, file, line, args } );
      }
    }
  }
  const base = locales.get( BASE_LOCALE ) ?? new Map();
  for ( const [ locale, messages ] of locales ) {
    if ( locale === BASE_LOCALE ) {
      continue;
    }
    for ( const [ key, entry ] of messages ) {
      if ( !base.has( key ) ) {
        report( 'catalog.locale_extra', entry.file, entry.line, `${key} is not in ${BASE_LOCALE}` );
      }
    }
  }
  return locales;
}

// --- error codes ------------------------------------------------------------

function engineCodes( schemaFile ) {
  const schema = JSON.parse( fs.readFileSync( schemaFile, 'utf8' ) );
  const codes = new Map();
  for ( const definition of Object.values( schema.$defs ?? {} ) ) {
    const code = definition?.properties?.code?.const;
    if ( typeof code === 'string' ) {
      codes.set( code, new Set( Object.keys( definition.properties.params?.properties ?? {} ) ) );
    }
  }
  return codes;
}

function shellCodes( rustDir ) {
  const codes = new Map();
  for ( const file of walk( rustDir, ( name ) => name.endsWith( '.rs' ) ) ) {
    const text = fs.readFileSync( file, 'utf8' );
    for ( const match of text.matchAll( /"(ui\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)"/g ) ) {
      if ( !codes.has( match[ 1 ] ) ) {
        codes.set( match[ 1 ], { file, line: lineOf( text, match.index ) } );
      }
    }
  }
  return codes;
}

function checkErrors( base, errorsSchema, rustDir, report ) {
  const engine = engineCodes( errorsSchema );
  const shell = shellCodes( rustDir );
  const required = new Set( [ ...engine.keys(), ...shell.keys() ] );
  for ( const [ code, params ] of engine ) {
    const entry = base.get( `error.${code}` );
    if ( entry === undefined ) {
      report( 'errors.missing', errorsSchema, 1, `engine code ${code} has no error.${code} message` );
      continue;
    }
    for ( const arg of entry.args ) {
      if ( !params.has( arg ) ) {
        report( 'errors.args', entry.file, entry.line, `error.${code} uses {${arg}}, which ${code} does not carry` );
      }
    }
  }
  for ( const [ code, where ] of shell ) {
    if ( !base.has( `error.${code}` ) ) {
      report( 'errors.missing', where.file, where.line, `shell code ${code} has no error.${code} message` );
    }
  }
  if ( !base.has( 'error.unknown' ) ) {
    report( 'errors.missing', errorsSchema, 1, 'no error.unknown message, the fallback for codes without one' );
  }
  for ( const [ key, entry ] of base ) {
    if ( key.startsWith( 'error.' ) && key !== 'error.unknown' && !required.has( key.slice( 'error.'.length ) ) ) {
      report( 'errors.orphan', entry.file, entry.line, `${key} matches no engine or shell code` );
    }
  }
}

// --- source -----------------------------------------------------------------

function looksLikeProse( text ) {
  if ( !LETTER.test( text ) ) {
    return false;
  }
  return /\p{L}[\s\u00a0]+\p{L}/u.test( text )
    || ( /^\p{Lu}\p{Ll}+$/u.test( text ) && !DOM_TOKENS.has( text ) )
    || /\p{L}[!?\u2026]\s*$/u.test( text );
}

function ignoredLines( text, file, report, exemptions ) {
  const ignored = new Set();
  text.split( '\n' ).forEach( ( line, index ) => {
    const match = /catalog-ignore\b:?\s*(.*?)\s*(?:\*\/\s*\}?)?\s*$/.exec( line );
    if ( !match ) {
      return;
    }
    if ( match[ 1 ] === '' ) {
      report( 'ignore.reason', file, index + 1, 'catalog-ignore needs a reason after the colon' );
      return;
    }
    exemptions.push( `${file}:${index + 1}: ${match[ 1 ]}` );
    ignored.add( index + 1 );
    ignored.add( index + 2 );
  } );
  return ignored;
}

function calleeName( expression ) {
  if ( ts.isIdentifier( expression ) ) {
    return expression.text;
  }
  if ( ts.isPropertyAccessExpression( expression ) ) {
    const left = calleeName( expression.expression );
    return left === null ? null : `${left}.${expression.name.text}`;
  }
  if ( ts.isMetaProperty( expression ) ) {
    return `${ts.tokenToString( expression.keywordToken )}.${expression.name.text}`;
  }
  return null;
}

function isStringish( node ) {
  return ts.isStringLiteral( node ) || ts.isNoSubstitutionTemplateLiteral( node );
}

function checkSource( file, base, usage, report, exemptions ) {
  const text = fs.readFileSync( file, 'utf8' );
  const kind = file.endsWith( '.tsx' ) ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const source = ts.createSourceFile( file, text, ts.ScriptTarget.Latest, true, kind );
  const ignored = ignoredLines( text, file, report, exemptions );
  const lineAt = ( node ) => source.getLineAndCharacterOfPosition( node.getStart( source ) ).line + 1;
  const flag = ( rule, node, message ) => {
    const line = lineAt( node );
    if ( !ignored.has( line ) ) {
      report( rule, file, line, message );
    }
  };

  const checkKey = ( node, valuesNode ) => {
    if ( !isStringish( node ) ) {
      return false;
    }
    const key = node.text;
    usage.keys.add( key );
    const entry = base.get( key );
    if ( entry === undefined ) {
      flag( 'source.key_unknown', node, `no message ${key}` );
      return true;
    }
    if ( valuesNode === undefined ) {
      if ( entry.args.size > 0 ) {
        flag( 'source.args', node, `${key} needs ${[ ...entry.args ].join( ', ' )}` );
      }
      return true;
    }
    if ( !ts.isObjectLiteralExpression( valuesNode ) ) {
      return true;
    }
    const given = new Set();
    for ( const property of valuesNode.properties ) {
      if ( ts.isSpreadAssignment( property ) || property.name === undefined || ts.isComputedPropertyName( property.name ) ) {
        return true;
      }
      given.add( property.name.text );
    }
    const missing = [ ...entry.args ].filter( ( arg ) => !given.has( arg ) );
    const extra = [ ...given ].filter( ( arg ) => !entry.args.has( arg ) );
    if ( missing.length > 0 || extra.length > 0 ) {
      const parts = [];
      if ( missing.length > 0 ) {
        parts.push( `missing ${missing.join( ', ' )}` );
      }
      if ( extra.length > 0 ) {
        parts.push( `not in the message: ${extra.join( ', ' )}` );
      }
      flag( 'source.args', valuesNode, `${key}: ${parts.join( '; ' )}` );
    }
    return true;
  };

  const checkPrefix = ( node ) => {
    if ( node === undefined || !isStringish( node ) ) {
      flag( 'source.key_dynamic', node ?? source, 'msgKey() needs a literal prefix' );
      return;
    }
    usage.prefixes.add( node.text );
    if ( ![ ...base.keys() ].some( ( key ) => key.startsWith( node.text ) ) ) {
      flag( 'source.key_dynamic', node, `no message key starts with ${node.text}` );
    }
  };

  // strict: the node is shown as is (JSX text position), so any letter counts.
  const checkString = ( node, value, strict ) => {
    if ( strict ? LETTER.test( value ) : looksLikeProse( value ) ) {
      flag( 'source.literal', node, `text a user could see: ${JSON.stringify( value )}` );
    }
  };

  const visit = ( node, strict = false ) => {
    if ( ts.isImportDeclaration( node ) || ts.isExportDeclaration( node ) || ts.isLiteralTypeNode( node ) ) {
      return;
    }
    if ( ts.isExpressionStatement( node ) && ts.isStringLiteral( node.expression ) ) {
      return; // a directive such as 'use strict'
    }
    if ( ts.isCallExpression( node ) || ts.isNewExpression( node ) ) {
      const name = calleeName( node.expression ) ?? '';
      const args = node.arguments ?? [];
      if ( /^(log|console)\.\w+$/.test( name ) || name === 'import.meta.glob' || /Error$/.test( name ) ) {
        return; // logs and errors stay English; they are never shown
      }
      if ( ts.isCallExpression( node ) && node.expression.kind === ts.SyntaxKind.ImportKeyword ) {
        return;
      }
      if ( name === 't' || name === 'msg' ) {
        const [ key, values, ...rest ] = args;
        if ( key !== undefined && !checkKey( key, values ) ) {
          visit( key );
        }
        if ( values !== undefined ) {
          visit( values );
        }
        rest.forEach( ( arg ) => visit( arg ) );
        return;
      }
      if ( name === 'msgKey' ) {
        checkPrefix( args[ 0 ] );
        args.slice( 1 ).forEach( ( arg ) => visit( arg ) );
        return;
      }
    }
    if ( ts.isPropertyAssignment( node ) ) {
      if ( ts.isComputedPropertyName( node.name ) ) {
        visit( node.name );
      }
      visit( node.initializer );
      return;
    }
    if ( ts.isElementAccessExpression( node ) && isStringish( node.argumentExpression ) ) {
      visit( node.expression );
      return;
    }
    if ( ts.isJsxText( node ) ) {
      if ( node.text.trim() !== '' ) {
        flag( 'source.literal', node, `JSX text: ${JSON.stringify( node.text.trim() )}` );
      }
      return;
    }
    if ( ts.isJsxSelfClosingElement( node ) || ts.isJsxOpeningElement( node ) ) {
      const tag = node.tagName.getText( source );
      const attributes = node.attributes.properties;
      const valuesAttribute = attributes.find( ( a ) => ts.isJsxAttribute( a ) && a.name.getText( source ) === 'values' );
      const valuesNode = valuesAttribute?.initializer && ts.isJsxExpression( valuesAttribute.initializer )
        ? valuesAttribute.initializer.expression
        : undefined;
      for ( const attribute of attributes ) {
        if ( ts.isJsxSpreadAttribute( attribute ) ) {
          visit( attribute.expression );
          continue;
        }
        const name = attribute.name.getText( source );
        const init = attribute.initializer;
        if ( init === undefined ) {
          continue;
        }
        if ( tag === 'Msg' && name === 'id' ) {
          const keyNode = ts.isJsxExpression( init ) ? init.expression : init;
          if ( keyNode !== undefined && !checkKey( keyNode, valuesNode ) ) {
            visit( keyNode );
          }
          continue;
        }
        const code = CODE_ATTRIBUTES.has( name ) || name.startsWith( 'data-' )
          || ( name.startsWith( 'aria-' ) && !TEXT_ARIA.has( name ) );
        if ( ts.isStringLiteral( init ) ) {
          if ( !code ) {
            checkString( init, init.text, /^[a-z]/.test( tag ) );
          }
          continue;
        }
        if ( ts.isJsxExpression( init ) && init.expression !== undefined ) {
          visit( init.expression, !code );
        }
      }
      return;
    }
    if ( ts.isJsxExpression( node ) ) {
      if ( node.expression !== undefined ) {
        visit( node.expression, true );
      }
      return;
    }
    if ( isStringish( node ) ) {
      checkString( node, node.text, strict );
      return;
    }
    if ( ts.isTemplateExpression( node ) ) {
      const shown = node.head.text + node.templateSpans.map( ( span ) => `X${span.literal.text}` ).join( '' );
      checkString( node, shown, strict );
      node.templateSpans.forEach( ( span ) => visit( span.expression ) );
      return;
    }
    // Strictness carries through the expressions that pass a value straight
    // to the screen: parentheses, a ? b : c, a ?? b, a || b, a && b.
    if ( ts.isParenthesizedExpression( node ) ) {
      visit( node.expression, strict );
      return;
    }
    if ( ts.isConditionalExpression( node ) ) {
      visit( node.condition );
      visit( node.whenTrue, strict );
      visit( node.whenFalse, strict );
      return;
    }
    if ( ts.isBinaryExpression( node ) ) {
      const passes = [ ts.SyntaxKind.QuestionQuestionToken, ts.SyntaxKind.BarBarToken, ts.SyntaxKind.AmpersandAmpersandToken ]
        .includes( node.operatorToken.kind );
      visit( node.left, passes && strict );
      visit( node.right, passes && strict );
      return;
    }
    ts.forEachChild( node, ( child ) => visit( child ) );
  };
  visit( source );
}

// --- CSS and HTML -----------------------------------------------------------

function checkCss( file, report ) {
  const text = fs.readFileSync( file, 'utf8' ).replace( /\/\*[\s\S]*?\*\//g, ( comment ) => comment.replace( /[^\n]/g, ' ' ) );
  text.split( '\n' ).forEach( ( line, index ) => {
    if ( PHYSICAL_CSS.some( ( pattern ) => pattern.test( line ) ) ) {
      report( 'css.physical', file, index + 1, `use a logical property: ${line.trim()}` );
    }
    const content = /(?<![\w-])content\s*:\s*(['"])(.*?)\1/.exec( line );
    if ( content && LETTER.test( content[ 2 ] ) ) {
      report( 'css.content', file, index + 1, `text in a content value: ${JSON.stringify( content[ 2 ] )}` );
    }
  } );
}

function checkHtml( file, report ) {
  const raw = fs.readFileSync( file, 'utf8' );
  const blank = ( match ) => match.replace( /[^\n]/g, ' ' );
  const text = raw.replace( /<!--[\s\S]*?-->/g, blank ).replace( /<(script|style)\b[\s\S]*?<\/\1>/gi, blank );
  for ( const match of text.matchAll( />([^<]+)</g ) ) {
    if ( LETTER.test( match[ 1 ] ) ) {
      report( 'html.text', file, lineOf( text, match.index ), `text: ${JSON.stringify( match[ 1 ].trim() )}` );
    }
  }
  for ( const match of text.matchAll( /\b(title|alt|placeholder|aria-label)\s*=\s*"([^"]*)"/g ) ) {
    if ( LETTER.test( match[ 2 ] ) ) {
      report( 'html.text', file, lineOf( text, match.index ), `${match[ 1 ]}: ${JSON.stringify( match[ 2 ] )}` );
    }
  }
}

// --- the check --------------------------------------------------------------

// Runs every rule. Paths: i18nDir (the catalogs), srcDir (TypeScript and
// CSS), htmlFiles, errorsSchema (protocol error codes, optional), rustDir
// (shell codes, optional), unusedFiles (catalog file names whose keys must
// all be used). Returns { findings, exemptions }.
export function check( { i18nDir, srcDir, htmlFiles = [], errorsSchema = null, rustDir = null, unusedFiles = [ 'ui.json' ] } ) {
  const findings = [];
  const exemptions = [];
  const report = ( rule, file, line, message ) => findings.push( { rule, file, line, message } );

  const locales = loadCatalogs( i18nDir, report );
  const base = locales.get( BASE_LOCALE ) ?? new Map();
  if ( errorsSchema !== null ) {
    checkErrors( base, errorsSchema, rustDir, report );
  }
  const usage = { keys: new Set(), prefixes: new Set() };
  for ( const file of walk( srcDir, ( name ) => /\.tsx?$/.test( name ) && !name.endsWith( '.d.ts' ) ) ) {
    checkSource( file, base, usage, report, exemptions );
  }
  for ( const [ key, entry ] of base ) {
    if ( unusedFiles.includes( path.basename( entry.file ) ) && !usage.keys.has( key )
      && ![ ...usage.prefixes ].some( ( prefix ) => key.startsWith( prefix ) ) ) {
      report( 'catalog.unused', entry.file, entry.line, `${key} is never used` );
    }
  }
  for ( const file of walk( srcDir, ( name ) => name.endsWith( '.css' ) ) ) {
    checkCss( file, report );
  }
  for ( const file of htmlFiles ) {
    checkHtml( file, report );
  }
  return { findings, exemptions };
}

function main() {
  const ui = path.dirname( path.dirname( fileURLToPath( import.meta.url ) ) );
  const repo = path.dirname( ui );
  const options = parseLogArgs( process.argv.slice( 2 ) );
  const log = createLogger( 'check-catalog', {
    logFile: options.logFile ?? path.join( ui, 'scripts', 'logs', 'check-catalog.log' ),
    logLevel: options.logLevel,
    consoleLevel: options.consoleLevel,
  } );
  const paths = {
    i18nDir: path.join( repo, 'i18n' ),
    srcDir: path.join( ui, 'src' ),
    htmlFiles: [ path.join( ui, 'index.html' ) ],
    errorsSchema: path.join( repo, 'protocol', 'v1', 'errors.schema.json' ),
    rustDir: path.join( ui, 'src-tauri', 'src' ),
  };
  log.debug( `paths: ${JSON.stringify( paths )}` );
  const { findings, exemptions } = check( paths );
  for ( const exemption of exemptions ) {
    log.info( `exempted: ${path.relative( repo, exemption )}` );
  }
  for ( const { rule, file, line, message } of findings ) {
    log.error( `${path.relative( repo, file )}:${line}: ${rule}: ${message}` );
  }
  log.info( `${findings.length} problem(s), ${exemptions.length} exemption(s); log: ${log.file}` );
  log.close();
  process.exitCode = findings.length === 0 ? 0 : 1;
}

if ( process.argv[ 1 ] && path.resolve( process.argv[ 1 ] ) === fileURLToPath( import.meta.url ) ) {
  main();
}
