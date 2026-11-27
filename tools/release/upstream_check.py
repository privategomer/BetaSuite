#!/usr/bin/env python3
"""
upstream_check.py - find code in this repository that still comes from the
upstream project this one was forked from.

Run it through tools/release/upstream_check.sh, which fetches the upstream
copy into a scratch folder outside the repository and routes this script's
output to the log. Every output line starts with its level (TRACE, DEBUG,
INFO, WARN, ERROR).

Commands:
  fetch      put the upstream tree at a pinned commit into --dest: a shallow
             clone of --url, or, without network, the same commit read from
             --local-repo; the commit id is checked either way
  inventory  three measures against the upstream tree:
               blame       lines whose last change is an upstream commit
               verbatim    lines (normalized, --min-chars or longer) that
                           also appear in an upstream file
               structural  per Python unit (function, method, class body,
                           block of module-level code), the share of its
                           token shingles found in upstream code, raw and
                           with identifiers, strings and numbers masked
  verify     fails on any verbatim line of --min-chars (default 40) or more
             and any run of --run-length (default 3) consecutive upstream
             lines, unless the allowlist names it with a reason
  gate       inventory without blame; fails on any verbatim line or any unit
             whose raw shingle share reaches --threshold. Used on a tree
             that has no history

Reports name repository paths, line numbers, unit names, scores and line
hashes. They never contain upstream text, so a reader who must not see the
upstream code can read them.
"""

import argparse
import hashlib
import io
import json
import keyword
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tokenize

DEFAULT_URL = 'https://github.com/solarorb93/BetaSuite.git'
DEFAULT_COMMIT = '5bd23e8216e85e84ac375547dd013621ab5fa170'

SKIP_DIRS = { '.git', 'node_modules', '.venv', 'venv', '__pycache__', 'target', 'dist', '.claude' }
MAX_FILE_BYTES = 4 * 1024 * 1024
SHINGLE_RAW = 5
SHINGLE_NORM = 12
RUN_MIN_TOTAL_CHARS = 30
MARKER = '.upstream-commit'


def log( level, message ):
    print( f'{level} {message}', flush=True )


def normalize( line ):
    return ' '.join( line.split() )


IMPORT_LINE = re.compile( r'^(import [\w.]+( as \w+)?(, [\w.]+( as \w+)?)*|from [\w.]+ import [\w, ()*]+)$' )
TRIVIAL_CHARS = 20


def is_trivial( normalized ):
    """Blank lines, plain import statements and lines under TRIVIAL_CHARS
    characters: no expression of their own, so they count apart."""
    return not normalized or len( normalized ) < TRIVIAL_CHARS or bool( IMPORT_LINE.match( normalized ) )


def line_hash( normalized ):
    return hashlib.sha256( normalized.encode( 'utf-8' ) ).hexdigest()[ :16 ]


# ---------------------------------------------------------------- file sets

def _is_text( path ):
    try:
        if os.path.getsize( path ) > MAX_FILE_BYTES:
            return False
        with open( path, 'rb' ) as handle:
            return b'\0' not in handle.read( 8192 )
    except OSError:
        return False


def list_files( root ):
    """Relative paths of the text files under root: git's tracked files when
    root is a git work tree, else every file outside the skipped folders."""
    paths = None
    if os.path.exists( os.path.join( root, '.git' ) ):
        result = subprocess.run( [ 'git', '-C', root, 'ls-files', '-z' ],
                                 capture_output=True, check=False )
        if result.returncode == 0:
            paths = [ p for p in result.stdout.decode( 'utf-8', 'replace' ).split( '\0' ) if p ]
    if paths is None:
        paths = []
        for directory, subdirs, files in os.walk( root ):
            subdirs[ : ] = sorted( d for d in subdirs if d not in SKIP_DIRS )
            for name in sorted( files ):
                paths.append( os.path.relpath( os.path.join( directory, name ), root ) )
    return sorted( p for p in paths if _is_text( os.path.join( root, p ) ) )


def read_lines( path ):
    with open( path, encoding='utf-8', errors='replace' ) as handle:
        return handle.read().splitlines()


# ---------------------------------------------------------------- upstream index

class Upstream:
    def __init__( self, root ):
        self.root = root
        self.files = [ f for f in list_files( root ) if f != MARKER ]
        self.lines = {}           # normalized line -> set of upstream files
        self.trigrams = set()     # consecutive non-blank normalized lines
        self.raw_shingles = set()
        self.norm_shingles = set()
        self.unit_shingles = {}   # 'file:unit' -> raw shingle set
        for rel in self.files:
            lines = read_lines( os.path.join( root, rel ) )
            seq = [ normalize( l ) for l in lines if normalize( l ) ]
            for text in seq:
                self.lines.setdefault( text, set() ).add( rel )
            for i in range( len( seq ) - 2 ):
                self.trigrams.add( tuple( seq[ i:i + 3 ] ) )
            if rel.endswith( '.py' ):
                source = '\n'.join( lines ) + '\n'
                for unit in python_units( source ):
                    raw, norm = unit_shingles( unit[ 'tokens' ] )
                    self.raw_shingles |= raw
                    self.norm_shingles |= norm
                    self.unit_shingles[ f'{rel}:{unit[ "name" ]}' ] = raw


# ---------------------------------------------------------------- python units

def _tokens( source ):
    """Code tokens (type, text, line) of source; comments and layout dropped."""
    skip = { tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
             tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER }
    out = []
    try:
        for tok in tokenize.generate_tokens( io.StringIO( source ).readline ):
            if tok.type not in skip:
                out.append( ( tok.type, tok.string, tok.start[ 0 ] ) )
    except ( tokenize.TokenError, IndentationError, SyntaxError ):
        pass
    return out


def python_units( source ):
    """Split a Python source into units: each top-level function, each method,
    each class's own body, and each block of module-level statements. Units
    carry name, first and last line, and their code tokens."""
    import ast
    try:
        tree = ast.parse( source )
    except SyntaxError:
        tokens = _tokens( source )
        last = tokens[ -1 ][ 2 ] if tokens else 1
        return [ { 'name': '<module>', 'start': 1, 'end': last, 'holes': (), 'tokens': tokens } ]
    tokens = _tokens( source )
    spans = []
    block = []

    def flush_block():
        if block:
            start, end = block[ 0 ].lineno, block[ -1 ].end_lineno
            spans.append( ( f'<top>@L{start}', start, end, () ) )
            block.clear()

    def first_line( node ):
        return min( [ node.lineno ] + [ d.lineno for d in getattr( node, 'decorator_list', [] ) ] )

    for node in tree.body:
        if isinstance( node, ( ast.FunctionDef, ast.AsyncFunctionDef ) ):
            flush_block()
            spans.append( ( node.name, first_line( node ), node.end_lineno, () ) )
        elif isinstance( node, ast.ClassDef ):
            flush_block()
            methods = [ n for n in node.body if isinstance( n, ( ast.FunctionDef, ast.AsyncFunctionDef ) ) ]
            holes = tuple( ( first_line( m ), m.end_lineno ) for m in methods )
            spans.append( ( node.name, first_line( node ), node.end_lineno, holes ) )
            for m in methods:
                spans.append( ( f'{node.name}.{m.name}', first_line( m ), m.end_lineno, () ) )
        else:
            block.append( node )
    flush_block()
    units = []
    for name, start, end, holes in spans:
        def inside( line ):
            return start <= line <= end and not any( a <= line <= b for a, b in holes )
        units.append( { 'name': name, 'start': start, 'end': end, 'holes': holes,
                        'tokens': [ t for t in tokens if inside( t[ 2 ] ) ] } )
    return units


def _masked( tok_type, text ):
    if tok_type == tokenize.NAME and not keyword.iskeyword( text ):
        return 'ID'
    if tok_type == tokenize.STRING:
        return 'STR'
    if tok_type == tokenize.NUMBER:
        return 'NUM'
    return text


def unit_shingles( tokens ):
    raw = [ t[ 1 ] for t in tokens ]
    norm = [ _masked( t[ 0 ], t[ 1 ] ) for t in tokens ]
    raw_set = { tuple( raw[ i:i + SHINGLE_RAW ] ) for i in range( len( raw ) - SHINGLE_RAW + 1 ) }
    norm_set = { tuple( norm[ i:i + SHINGLE_NORM ] ) for i in range( len( norm ) - SHINGLE_NORM + 1 ) }
    return raw_set, norm_set


# ---------------------------------------------------------------- measures

def verbatim( repo, files, upstream, min_chars ):
    """Per file, the lines whose normalized text is in an upstream file."""
    hits = {}
    for rel in files:
        found = []
        for number, line in enumerate( read_lines( os.path.join( repo, rel ) ), 1 ):
            text = normalize( line )
            if len( text ) >= min_chars and text in upstream.lines:
                found.append( { 'line': number, 'chars': len( text ), 'hash': line_hash( text ),
                                'trivial': is_trivial( text ),
                                'upstream_files': sorted( upstream.lines[ text ] ) } )
        if found:
            hits[ rel ] = found
    return hits


def runs( repo, files, upstream, run_length ):
    """Per file, maximal runs of run_length or more consecutive non-blank lines
    whose every window of three appears consecutively upstream."""
    found = {}
    if run_length < 3:
        raise ValueError( 'run length is at least 3' )
    for rel in files:
        lines = read_lines( os.path.join( repo, rel ) )
        seq = [ ( n, normalize( l ) ) for n, l in enumerate( lines, 1 ) if normalize( l ) ]
        marked = [ False ] * len( seq )
        for i in range( len( seq ) - 2 ):
            window = tuple( t for _, t in seq[ i:i + 3 ] )
            if window in upstream.trigrams and sum( len( t ) for t in window ) >= RUN_MIN_TOTAL_CHARS:
                marked[ i ] = marked[ i + 1 ] = marked[ i + 2 ] = True
        result = []
        i = 0
        while i < len( seq ):
            if not marked[ i ]:
                i += 1
                continue
            j = i
            while j + 1 < len( seq ) and marked[ j + 1 ]:
                j += 1
            if j - i + 1 >= run_length:
                result.append( { 'start': seq[ i ][ 0 ], 'end': seq[ j ][ 0 ], 'lines': j - i + 1,
                                 'hashes': [ line_hash( t ) for _, t in seq[ i:j + 1 ] ] } )
            i = j + 1
        if result:
            found[ rel ] = result
    return found


def structural( repo, files, upstream, line_hits=None ):
    """Per Python unit, the share of its shingles found in upstream code, and
    how many of its lines line_hits ({path: {measure: line numbers}}) marks."""
    line_hits = line_hits or {}
    rows = []
    for rel in files:
        if not rel.endswith( '.py' ):
            continue
        source = '\n'.join( read_lines( os.path.join( repo, rel ) ) ) + '\n'
        for unit in python_units( source ):
            raw, norm = unit_shingles( unit[ 'tokens' ] )
            if not raw:
                continue
            raw_shared = raw & upstream.raw_shingles
            counts = {}
            for measure, numbers in line_hits.get( rel, {} ).items():
                counts[ measure ] = sum( 1 for n in numbers if unit[ 'start' ] <= n <= unit[ 'end' ]
                                         and not any( a <= n <= b for a, b in unit[ 'holes' ] ) )
            lines_in_unit = sum( 1 for n in range( unit[ 'start' ], unit[ 'end' ] + 1 )
                                 if not any( a <= n <= b for a, b in unit[ 'holes' ] ) )
            if not raw_shared and not ( norm & upstream.norm_shingles ) and not any( counts.values() ):
                continue
            best, best_count = '', 0
            for name, shingles in upstream.unit_shingles.items():
                count = len( raw & shingles )
                if count > best_count:
                    best, best_count = name, count
            rows.append( {
                'path': rel, 'unit': unit[ 'name' ], 'start': unit[ 'start' ], 'end': unit[ 'end' ],
                'tokens': len( unit[ 'tokens' ] ),
                'raw_share': round( len( raw_shared ) / len( raw ), 3 ),
                'norm_share': round( len( norm & upstream.norm_shingles ) / len( norm ), 3 ) if norm else 0.0,
                'best_upstream_unit': best,
                'lines': lines_in_unit,
                'blame_lines': counts.get( 'blame', 0 ),
                'verbatim_lines': counts.get( 'verbatim', 0 ),
                'substantive_lines': counts.get( 'substantive', 0 ),
            } )
    rows.sort( key=lambda r: ( -r[ 'raw_share' ], -r[ 'blame_lines' ] - r[ 'verbatim_lines' ], r[ 'path' ], r[ 'start' ] ) )
    return rows


def blame( repo, files, commit ):
    """Per file, the line numbers whose blamed commit is reachable from commit."""
    result = subprocess.run( [ 'git', '-C', repo, 'rev-list', commit ],
                             capture_output=True, text=True, check=False )
    if result.returncode != 0:
        log( 'WARN', f'blame skipped: commit {commit[ :12 ]} is not in this repository' )
        return None
    upstream_commits = set( result.stdout.split() )
    log( 'DEBUG', f'{len( upstream_commits )} upstream commits in history' )
    hits = {}
    header = re.compile( r'^([0-9a-f]{40}) \d+ (\d+)' )
    for rel in files:
        out = subprocess.run( [ 'git', '-C', repo, 'blame', '-w', '-M', '-C', '--line-porcelain', 'HEAD', '--', rel ],
                              capture_output=True, text=True, errors='replace', check=False )
        if out.returncode != 0:
            log( 'DEBUG', f'blame failed for {rel}: {out.stderr.strip()[ :200 ]}' )
            continue
        lines = [ int( m.group( 2 ) ) for m in map( header.match, out.stdout.splitlines() )
                  if m and m.group( 1 ) in upstream_commits ]
        if lines:
            text = read_lines( os.path.join( repo, rel ) )
            hits[ rel ] = [ { 'line': n, 'trivial': is_trivial( normalize( text[ n - 1 ] ) if n <= len( text ) else '' ) }
                            for n in lines ]
        log( 'TRACE', f'blamed {rel}: {len( lines )} upstream lines' )
    return hits


# ---------------------------------------------------------------- allowlist

def load_allowlist( path ):
    if not path or not os.path.exists( path ):
        return {}
    with open( path, encoding='utf-8' ) as handle:
        data = json.load( handle )
    allowed = {}
    for entry in data.get( 'entries', [] ):
        if not entry.get( 'reason', '' ).strip():
            raise ValueError( f'allowlist entry for {entry.get( "path" )} has no reason' )
        allowed.setdefault( entry[ 'path' ], set() ).add( entry[ 'hash' ] )
    return allowed


def apply_allowlist( verbatim_hits, run_hits, allowed ):
    v = {}
    for rel, items in verbatim_hits.items():
        keep = [ h for h in items if h[ 'hash' ] not in allowed.get( rel, set() ) ]
        if keep:
            v[ rel ] = keep
    r = {}
    for rel, items in run_hits.items():
        keep = [ h for h in items if not set( h[ 'hashes' ] ) <= allowed.get( rel, set() ) ]
        if keep:
            r[ rel ] = keep
    return v, r


# ---------------------------------------------------------------- reports

def _ranges( numbers ):
    out, start, prev = [], None, None
    for n in sorted( numbers ):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append( ( start, prev ) )
            start = prev = n
    if start is not None:
        out.append( ( start, prev ) )
    return ', '.join( str( a ) if a == b else f'{a}-{b}' for a, b in out )


def render_markdown( report ):
    out = [ f'# Upstream check: {report[ "command" ]}', '',
            f'- Upstream commit: `{report[ "commit" ]}`',
            f'- Files scanned: {report[ "files_scanned" ]}; upstream files: {report[ "upstream_files" ]}',
            f'- Verbatim threshold: {report[ "min_chars" ]} characters (normalized)', '' ]
    if report.get( 'blame' ) is not None:
        total = sum( len( v ) for v in report[ 'blame' ].values() )
        subst = sum( 1 for v in report[ 'blame' ].values() for h in v if not h[ 'trivial' ] )
        out += [ f'## Blame: {total} lines in {len( report[ "blame" ] )} files, {subst} substantive', '',
                 'Trivial: blank, a plain import, or under 20 characters.', '',
                 '| File | Lines | Substantive | Substantive line numbers |', '|---|---|---|---|' ]
        for rel, items in sorted( report[ 'blame' ].items(), key=lambda kv: -len( kv[ 1 ] ) ):
            sub = [ h[ 'line' ] for h in items if not h[ 'trivial' ] ]
            out.append( f'| `{rel}` | {len( items )} | {len( sub )} | {_ranges( sub )} |' )
        out.append( '' )
    v = report[ 'verbatim' ]
    total = sum( len( x ) for x in v.values() )
    subst = sum( 1 for x in v.values() for h in x if not h.get( 'trivial' ) )
    out += [ f'## Verbatim: {total} lines in {len( v )} files, {subst} substantive', '',
             '| File | Lines | Substantive | Substantive line numbers | Upstream files |', '|---|---|---|---|---|' ]
    for rel, items in sorted( v.items(), key=lambda kv: -len( kv[ 1 ] ) ):
        sources = sorted( { s for h in items for s in h[ 'upstream_files' ] } )
        sub = [ h[ 'line' ] for h in items if not h.get( 'trivial' ) ]
        out.append( f'| `{rel}` | {len( items )} | {len( sub )} | {_ranges( sub )} | {", ".join( sources )} |' )
    out.append( '' )
    if 'runs' in report:
        r = report[ 'runs' ]
        out += [ f'## Runs of {report[ "run_length" ]}+ consecutive upstream lines: '
                 f'{sum( len( x ) for x in r.values() )} in {len( r )} files', '',
                 '| File | Lines | Length |', '|---|---|---|' ]
        for rel, items in sorted( r.items() ):
            for h in items:
                out.append( f'| `{rel}` | {h[ "start" ]}-{h[ "end" ]} | {h[ "lines" ]} |' )
        out.append( '' )
    if 'structural' in report:
        rows = report[ 'structural' ]
        out += [ f'## Structural: {len( rows )} Python units share shingles with upstream', '',
                 'raw share: the unit\'s 5-token shingles found in upstream code. '
                 'masked share: 12-token shingles with names, strings and numbers masked.', '',
                 '| File | Unit | Lines | Tokens | Raw share | Masked share | Blamed | Verbatim | Substantive | Closest upstream unit |',
                 '|---|---|---|---|---|---|---|---|---|---|' ]
        for row in rows:
            out.append( f'| `{row[ "path" ]}` | `{row[ "unit" ]}` | {row[ "start" ]}-{row[ "end" ]} | {row[ "tokens" ]} '
                        f'| {row[ "raw_share" ]:.2f} | {row[ "norm_share" ]:.2f} | {row[ "blame_lines" ]} | {row[ "verbatim_lines" ]} '
                        f'| {row[ "substantive_lines" ]} '
                        f'| `{row[ "best_upstream_unit" ]}` |' )
        out.append( '' )
    if 'failures' in report:
        out += [ f'## Result: {"pass" if not report[ "failures" ] else "FAIL"}', '' ]
        out += [ f'- {f}' for f in report[ 'failures' ] ]
        out.append( '' )
    return '\n'.join( out )


def write_reports( report, out_dir, name ):
    os.makedirs( out_dir, exist_ok=True )
    md = os.path.join( out_dir, f'{name}.md' )
    js = os.path.join( out_dir, f'{name}.json' )
    with open( md, 'w', encoding='utf-8' ) as handle:
        handle.write( render_markdown( report ) )
    with open( js, 'w', encoding='utf-8' ) as handle:
        json.dump( report, handle, indent=1, sort_keys=True )
    log( 'INFO', f'report: {md}' )
    log( 'INFO', f'data: {js}' )


# ---------------------------------------------------------------- commands

def cmd_fetch( args ):
    dest = os.path.abspath( args.dest )
    if os.path.exists( dest ) and os.listdir( dest ):
        marker = os.path.join( dest, MARKER )
        if os.path.exists( marker ) and open( marker ).read().strip() == args.commit:
            log( 'INFO', f'upstream copy already at {args.commit[ :12 ]} in {dest}' )
            return 0
        log( 'ERROR', f'{dest} is not empty and holds no copy of {args.commit[ :12 ]}; pick an empty folder' )
        return 1
    os.makedirs( dest, exist_ok=True )
    with tempfile.TemporaryDirectory( dir=os.path.dirname( dest ) ) as work:
        ok = False
        if not args.offline:
            log( 'INFO', f'fetching {args.commit[ :12 ]} from {args.url}' )
            steps = [ [ 'git', 'init', '-q', work ],
                      [ 'git', '-C', work, 'fetch', '-q', '--depth', '1', args.url, args.commit ] ]
            ok = all( subprocess.run( s, capture_output=True, check=False ).returncode == 0 for s in steps )
            if ok:
                got = subprocess.run( [ 'git', '-C', work, 'rev-parse', 'FETCH_HEAD' ],
                                      capture_output=True, text=True, check=False ).stdout.strip()
                ok = got == args.commit
                if not ok:
                    log( 'ERROR', f'fetched commit {got[ :12 ]} is not the pinned {args.commit[ :12 ]}' )
                    return 1
                archive = subprocess.run( [ 'git', '-C', work, 'archive', 'FETCH_HEAD' ],
                                          capture_output=True, check=False )
                ok = archive.returncode == 0
            if not ok:
                log( 'WARN', 'network fetch failed; trying the local repository' )
        if not ok:
            if not args.local_repo:
                log( 'ERROR', 'no network copy and no local repository to read it from' )
                return 1
            got = subprocess.run( [ 'git', '-C', args.local_repo, 'rev-parse', '--verify', '-q', f'{args.commit}^{{commit}}' ],
                                  capture_output=True, text=True, check=False ).stdout.strip()
            if got != args.commit:
                log( 'ERROR', f'commit {args.commit[ :12 ]} is not in {args.local_repo}' )
                return 1
            archive = subprocess.run( [ 'git', '-C', args.local_repo, 'archive', args.commit ],
                                      capture_output=True, check=False )
            if archive.returncode != 0:
                log( 'ERROR', 'git archive failed on the local repository' )
                return 1
            log( 'INFO', 'read the pinned commit from the local repository' )
        with tarfile.open( fileobj=io.BytesIO( archive.stdout ) ) as tar:
            tar.extractall( dest, filter='data' )
    with open( os.path.join( dest, MARKER ), 'w' ) as handle:
        handle.write( args.commit + '\n' )
    log( 'INFO', f'upstream copy at {args.commit[ :12 ]}: {len( list_files( dest ) )} text files in {dest}' )
    return 0


def _load( args ):
    upstream = Upstream( args.upstream )
    log( 'INFO', f'indexed {len( upstream.files )} upstream files' )
    files = list_files( args.repo )
    log( 'INFO', f'scanning {len( files )} files in {args.repo}' )
    return upstream, files


def _base_report( args, command, upstream, files, min_chars ):
    return { 'command': command, 'commit': args.commit, 'files_scanned': len( files ),
             'upstream_files': len( upstream.files ), 'min_chars': min_chars }


def cmd_inventory( args ):
    upstream, files = _load( args )
    report = _base_report( args, 'inventory', upstream, files, args.min_chars )
    if args.command == 'inventory':
        report[ 'blame' ] = blame( args.repo, files, args.commit )
    report[ 'verbatim' ] = verbatim( args.repo, files, upstream, args.min_chars )
    report[ 'runs' ] = runs( args.repo, files, upstream, 3 )
    report[ 'run_length' ] = 3
    hits = {}
    for rel, items in ( report.get( 'blame' ) or {} ).items():
        hits.setdefault( rel, {} )[ 'blame' ] = { h[ 'line' ] for h in items }
        hits[ rel ].setdefault( 'substantive', set() ).update( h[ 'line' ] for h in items if not h[ 'trivial' ] )
    for rel, items in report[ 'verbatim' ].items():
        hits.setdefault( rel, {} )[ 'verbatim' ] = { h[ 'line' ] for h in items }
        hits[ rel ].setdefault( 'substantive', set() ).update( h[ 'line' ] for h in items if not h[ 'trivial' ] )
    report[ 'structural' ] = structural( args.repo, files, upstream, hits )
    if args.command == 'gate':
        report[ 'command' ] = 'gate'
        failures = [ f'verbatim: {rel} line {h[ "line" ]}' for rel, items in report[ 'verbatim' ].items() for h in items ]
        failures += [ f'structural: {r[ "path" ]} {r[ "unit" ]} raw share {r[ "raw_share" ]:.2f}'
                      for r in report[ 'structural' ] if r[ 'raw_share' ] >= args.threshold ]
        report[ 'failures' ] = failures
    v = sum( len( x ) for x in report[ 'verbatim' ].values() )
    log( 'INFO', f'verbatim lines ({args.min_chars}+ chars): {v} in {len( report[ "verbatim" ] )} files, '
                 f'{sum( 1 for x in report[ "verbatim" ].values() for h in x if not h[ "trivial" ] )} substantive' )
    if report.get( 'blame' ) is not None:
        log( 'INFO', f'blamed upstream lines: {sum( len( x ) for x in report[ "blame" ].values() )}, '
                     f'{sum( 1 for x in report[ "blame" ].values() for h in x if not h[ "trivial" ] )} substantive' )
    log( 'INFO', f'Python units sharing shingles: {len( report[ "structural" ] )}' )
    write_reports( report, args.out, args.command )
    if args.command == 'gate':
        for f in report[ 'failures' ]:
            log( 'ERROR', f )
        log( 'INFO' if not report[ 'failures' ] else 'ERROR',
             f'gate {"passed" if not report[ "failures" ] else "failed"}: {len( report[ "failures" ] )} findings' )
        return 0 if not report[ 'failures' ] else 1
    return 0


def cmd_verify( args ):
    upstream, files = _load( args )
    allowed = load_allowlist( args.allowlist )
    files = [ f for f in files if os.path.normpath( f ) != os.path.normpath( os.path.relpath( args.allowlist, args.repo ) ) ] \
        if args.allowlist else files
    v_all = verbatim( args.repo, files, upstream, args.min_chars )
    r_all = runs( args.repo, files, upstream, args.run_length )
    v, r = apply_allowlist( v_all, r_all, allowed )
    report = _base_report( args, 'verify', upstream, files, args.min_chars )
    report.update( { 'verbatim': v, 'runs': r, 'run_length': args.run_length,
                     'allowlisted': sum( len( x ) for x in v_all.values() ) - sum( len( x ) for x in v.values() ) } )
    failures = [ f'verbatim: {rel} line {h[ "line" ]} ({h[ "chars" ]} chars, hash {h[ "hash" ]})'
                 for rel, items in v.items() for h in items ]
    failures += [ f'run: {rel} lines {h[ "start" ]}-{h[ "end" ]}' for rel, items in r.items() for h in items ]
    report[ 'failures' ] = failures
    write_reports( report, args.out, 'verify' )
    for f in failures:
        log( 'ERROR', f )
    log( 'INFO', f'allowlisted lines: {report[ "allowlisted" ]}' )
    if failures:
        log( 'ERROR', f'verify failed: {len( failures )} findings' )
        return 1
    log( 'INFO', 'verify passed: no upstream lines outside the allowlist' )
    return 0


def main( argv=None ):
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    sub = parser.add_subparsers( dest='command', required=True )
    fetch = sub.add_parser( 'fetch' )
    fetch.add_argument( '--dest', required=True )
    fetch.add_argument( '--url', default=DEFAULT_URL )
    fetch.add_argument( '--commit', default=DEFAULT_COMMIT )
    fetch.add_argument( '--local-repo' )
    fetch.add_argument( '--offline', action='store_true' )
    for name in ( 'inventory', 'verify', 'gate' ):
        p = sub.add_parser( name )
        p.add_argument( '--upstream', required=True )
        p.add_argument( '--repo', required=True )
        p.add_argument( '--out', required=True )
        p.add_argument( '--commit', default=DEFAULT_COMMIT )
        p.add_argument( '--min-chars', type=int, default=40 if name == 'verify' else 12 )
        if name == 'verify':
            p.add_argument( '--run-length', type=int, default=3 )
            p.add_argument( '--allowlist' )
        if name == 'gate':
            p.add_argument( '--threshold', type=float, default=0.3 )
    args = parser.parse_args( argv )
    if args.command == 'fetch':
        return cmd_fetch( args )
    args.upstream = os.path.abspath( args.upstream )
    args.repo = os.path.abspath( args.repo )
    if not os.path.isdir( args.upstream ) or not list_files( args.upstream ):
        log( 'ERROR', f'no upstream copy in {args.upstream}; run fetch first' )
        return 1
    if args.command == 'verify':
        return cmd_verify( args )
    return cmd_inventory( args )


if __name__ == '__main__':
    sys.exit( main() )
