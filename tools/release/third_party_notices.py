#!/usr/bin/env python3
"""Write the third-party notices and audit every dependency's license.

    python3 tools/release/third_party_notices.py [--python PATH] [--pypi] [--check]

Stdlib only. Reads, without the network unless --pypi:
  Rust crates      `cargo metadata --offline` on ui/src-tauri (Cargo.lock)
  JavaScript       ui/package-lock.json, runtime packages only (no dev
                   tools), each license from ui/node_modules
  Python           the requirement files' packages and what they pull in,
                   from the engine interpreter's installed metadata
                   (PYTHON_BIN, or --python). Setup installs within the
                   files' ranges, so versions are those of this interpreter.
                   A package not installed here is looked up on PyPI's JSON
                   API with --pypi, else reported and the run fails
  Downloads        ffmpeg (tools/lib/ffmpeg_pin.json) and NVIDIA's CUDA
                   packages, fetched by the user's machine, never bundled
  Models           assets/models/catalog.json

Writes assets/notices/third_party.json (the data; the test suite checks it
against the lockfiles and the license policy) and THIRD_PARTY_NOTICES.md
(rendered from that data; the app's About screen shows it). --check writes
nothing and fails when either file is out of date or a license is not
allowed. Lines on stdout start with a level for the wrapper's log.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, 'assets', 'notices', 'third_party.json')
MARKDOWN = os.path.join(REPO, 'THIRD_PARTY_NOTICES.md')
REVIEWED = os.path.join(HERE, 'license_review.json')
REQUIREMENT_FILES = ('requirements.txt', 'requirements-cpu.txt', 'requirements-gpu.txt',
                     'gpu-requirements.txt', 'requirements-vision-windows.txt')
FORMAT = 1

# Licenses whose terms let them ship in, or alongside, an AGPL-3.0-or-later
# app. An expression passes when every AND part has an allowed OR choice.
ALLOWED = {
    '0BSD', 'Apache-2.0', 'BSD-1-Clause', 'BSD-2-Clause', 'BSD-3-Clause', 'BSL-1.0', 'CC0-1.0',
    'CDLA-Permissive-2.0', 'ISC', 'MIT', 'MIT-0', 'MPL-2.0', 'PSF-2.0', 'Unicode-3.0',
    'Unicode-DFS-2016', 'Unlicense', 'Zlib', 'LGPL-2.1-or-later', 'LGPL-3.0-or-later',
    'GPL-3.0', 'GPL-3.0-or-later', 'AGPL-3.0', 'AGPL-3.0-or-later', 'LLVM-exception',
    'Apache-2.0 WITH LLVM-exception',
}
# Free-text license fields as packages write them, to SPDX ids.
ALIASES = {
    'mit license': 'MIT', 'mit': 'MIT', 'apache 2.0': 'Apache-2.0', 'apache-2.0': 'Apache-2.0',
    'apache software license': 'Apache-2.0', 'apache license 2.0': 'Apache-2.0',
    'apache license, version 2.0': 'Apache-2.0', 'bsd license': 'BSD-3-Clause',
    'bsd': 'BSD-3-Clause', 'bsd-3-clause': 'BSD-3-Clause', '3-clause bsd license': 'BSD-3-Clause',
    'new bsd license': 'BSD-3-Clause', 'isc license': 'ISC', 'psf': 'PSF-2.0',
    'python software foundation license': 'PSF-2.0', 'mozilla public license 2.0 (mpl 2.0)': 'MPL-2.0',
}
CLASSIFIERS = {
    'License :: OSI Approved :: MIT License': 'MIT',
    'License :: OSI Approved :: Apache Software License': 'Apache-2.0',
    'License :: OSI Approved :: BSD License': 'BSD-3-Clause',
    'License :: OSI Approved :: ISC License (ISCL)': 'ISC',
    'License :: OSI Approved :: Python Software Foundation License': 'PSF-2.0',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)': 'MPL-2.0',
}


def say(level, text):
    print('%s %s' % (level, text), flush=True)


# --- license expressions --------------------------------------------------

def normalize(text):
    """An SPDX-like expression from a license field ('MIT/Apache-2.0' -> 'MIT OR Apache-2.0')."""
    if not text:
        return ''
    text = text.strip()
    alias = ALIASES.get(text.lower())
    if alias:
        return alias
    if '\n' in text or len(text) > 120:
        return ''   # a whole license text pasted into the field: unknown, use classifiers
    return re.sub(r'\s*/\s*', ' OR ', text)


def _split(expr, word):
    depth, start, parts = 0, 0, []
    tokens = re.split(r'(\(|\)|\s+%s\s+)' % word, expr)
    current = ''
    for token in tokens:
        if token == '(':
            depth += 1
        elif token == ')':
            depth -= 1
        if depth == 0 and re.fullmatch(r'\s+%s\s+' % word, token or ''):
            parts.append(current)
            current = ''
        else:
            current += token or ''
    parts.append(current)
    del start
    return [p.strip() for p in parts if p.strip()]


def allowed(expr, extra=()):
    """Whether a license expression may ship with the app."""
    expr = expr.strip()
    while expr.startswith('(') and expr.endswith(')') and _balanced(expr[1:-1]):
        expr = expr[1:-1].strip()
    if not expr:
        return False
    ands = _split(expr, 'AND')
    if len(ands) > 1:
        return all(allowed(part, extra) for part in ands)
    ors = _split(expr, 'OR')
    if len(ors) > 1:
        return any(allowed(part, extra) for part in ors)
    return expr in ALLOWED or expr in extra


def _balanced(text):
    depth = 0
    for ch in text:
        depth += ch == '('
        depth -= ch == ')'
        if depth < 0:
            return False
    return depth == 0


# --- sources ----------------------------------------------------------------

def rust():
    manifest = os.path.join(REPO, 'ui', 'src-tauri', 'Cargo.toml')
    out = subprocess.run(['cargo', 'metadata', '--format-version', '1', '--locked', '--offline',
                          '--manifest-path', manifest], capture_output=True, text=True, check=True)
    meta = json.loads(out.stdout)
    rows = []
    for pkg in meta['packages']:
        if pkg.get('source') is None:
            continue   # the app itself
        rows.append({'name': pkg['name'], 'version': pkg['version'], 'license': pkg.get('license') or '',
                     'home': pkg.get('repository') or pkg.get('homepage') or ''})
    say('INFO', 'rust: %d crates' % len(rows))
    return sorted(rows, key=lambda r: (r['name'], r['version']))


def npm_runtime(lock):
    """{ node_modules path: version } of the packages the page ships (no dev tools)."""
    found = {}
    for path, pkg in lock['packages'].items():
        if not path or pkg.get('dev') or pkg.get('devOptional') or pkg.get('link'):
            continue
        found[path] = pkg['version']
    return found


def npm():
    ui = os.path.join(REPO, 'ui')
    with open(os.path.join(ui, 'package-lock.json'), encoding='utf-8') as f:
        lock = json.load(f)
    rows = []
    for path, version in npm_runtime(lock).items():
        name = path.split('node_modules/')[-1]
        manifest = os.path.join(ui, path, 'package.json')
        if not os.path.isfile(manifest):
            raise SystemExit('ERROR %s is not installed; run npm ci in ui/ first' % name)
        with open(manifest, encoding='utf-8') as f:
            pkg = json.load(f)
        license = pkg.get('license') or ''
        if isinstance(license, dict):
            license = license.get('type', '')
        repo = pkg.get('repository') or ''
        if isinstance(repo, dict):
            repo = repo.get('url', '')
        rows.append({'name': name, 'version': version, 'license': license,
                     'home': pkg.get('homepage') or repo})
    say('INFO', 'npm: %d runtime packages' % len(rows))
    return sorted(rows, key=lambda r: (r['name'], r['version']))


def requirement_names():
    """{ normalized name: [ requirement files that list it ] }."""
    found = {}
    for name in REQUIREMENT_FILES:
        path = os.path.join(REPO, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if not line or line.startswith('-'):
                    continue
                req = re.split(r'[<>=!~;\[ ]', line, maxsplit=1)[0]
                found.setdefault(canonical(req), []).append(name)
    return found


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


_PROBE = r'''
import importlib.metadata as md, json, re, sys
def canon(n): return re.sub(r'[-_.]+', '-', n).lower()
want = json.loads(sys.argv[1]); seen = {}; todo = list(want)
while todo:
    name = todo.pop()
    if name in seen: continue
    try: dist = md.distribution(name)
    except md.PackageNotFoundError:
        seen[name] = None; continue
    m = dist.metadata
    reqs = []
    for r in dist.requires or []:
        if 'extra ==' in r: continue
        reqs.append(canon(re.split(r'[<>=!~;\[ (]', r.strip(), maxsplit=1)[0]))
    seen[name] = {'name': m['Name'], 'version': m['Version'],
                  'expression': m.get('License-Expression') or '', 'license': m.get('License') or '',
                  'classifiers': [c for c in (m.get_all('Classifier') or []) if c.startswith('License ::')],
                  'home': m.get('Home-page') or '', 'requires': reqs}
    todo.extend(reqs)
print(json.dumps(seen))
'''


def _license_from(meta):
    for text in (meta.get('expression'), meta.get('license')):
        expr = normalize(text or '')
        if expr:
            return expr
    for c in meta.get('classifiers', []):
        if c in CLASSIFIERS:
            return CLASSIFIERS[c]
    if meta.get('classifiers'):
        return meta['classifiers'][0].split('::')[-1].strip()
    return ''


def pypi(name):
    with urllib.request.urlopen('https://pypi.org/pypi/%s/json' % name, timeout=20) as answer:
        info = json.load(answer)['info']
    meta = {'expression': info.get('license_expression') or '', 'license': info.get('license') or '',
            'classifiers': [c for c in info.get('classifiers') or [] if c.startswith('License ::')]}
    return {'name': info['name'], 'version': 'not installed here; latest %s' % info['version'],
            'license': _license_from(meta), 'home': info.get('home_page') or info.get('project_url') or '',
            'from': 'pypi'}


def python(python_bin, use_pypi):
    wanted = requirement_names()
    out = subprocess.run([python_bin, '-c', _PROBE, json.dumps(sorted(wanted))],
                         capture_output=True, text=True, check=True)
    seen = json.loads(out.stdout)
    rows, missing = [], []
    for key, meta in sorted(seen.items()):
        if meta is None:
            if key in wanted:
                missing.append(key)
            continue
        rows.append({'name': meta['name'], 'version': meta['version'], 'license': _license_from(meta),
                     'home': meta['home'], 'requirement': ', '.join(wanted.get(key, [])) or 'pulled in'})
    for key in missing:
        if not use_pypi:
            raise SystemExit('ERROR %s is not installed in %s; rerun with --pypi to look it up' % (key, python_bin))
        row = pypi(key)
        row['requirement'] = ', '.join(wanted[key])
        say('INFO', 'python: %s looked up on PyPI (%s)' % (key, row['license']))
        rows.append(row)
    say('INFO', 'python: %d packages (%d listed in requirement files)' % (len(rows), len(wanted)))
    return sorted(rows, key=lambda r: canonical(r['name']))


def downloads():
    with open(os.path.join(REPO, 'tools', 'lib', 'ffmpeg_pin.json'), encoding='utf-8') as f:
        pin = json.load(f)
    return [{'name': 'ffmpeg', 'version': pin['version'], 'license': 'GPL (x264 included)',
             'home': 'https://github.com/FFmpeg-Builds/FFmpeg-Builds', 'note': pin['source']}]


def models():
    with open(os.path.join(REPO, 'assets', 'models', 'catalog.json'), encoding='utf-8') as f:
        cat = json.load(f)
    rows = []
    for model in cat['models']:
        lic = cat['licenses'][model['license']]
        rows.append({'name': model['id'], 'version': model['sha256'][:12], 'license': lic['spdx'],
                     'home': lic['home'], 'project': lic['project'], 'upstream': model['upstream']})
    return rows


# --- audit and render -------------------------------------------------------

def reviewed():
    with open(REVIEWED, encoding='utf-8') as f:
        return json.load(f)


def audit(data):
    """[ (section, name, license) ] that no policy allows."""
    review = reviewed()
    problems = []
    for section in ('rust', 'npm', 'python', 'downloads', 'models'):
        for row in data[section]:
            ok = allowed(normalize(row['license']) or row['license'])
            if not ok:
                for rule in review['allow']:
                    if rule['section'] == section and re.fullmatch(rule['name'], row['name']) \
                            and rule['license'] == row['license']:
                        ok = True
            if not ok:
                problems.append((section, row['name'], row['license']))
    return problems


TITLES = {
    'rust': ('The app shell: Rust crates', 'From ui/src-tauri/Cargo.lock: every crate the shell can build with, on every platform.'),
    'npm': ('The app window: JavaScript packages', 'From ui/package-lock.json: the packages the window ships, not build tools.'),
    'python': ('The engine: Python packages', 'Installed by setup from the requirement files, within their version ranges. '
               'The versions shown are those this list was made from.'),
    'downloads': ('Downloaded by setup', 'Fetched by your computer from their publishers when setup runs; '
                  'never stored in this repository. The NVIDIA CUDA packages, under NVIDIA\'s own terms, '
                  'are listed with the Python packages and are fetched only on computers with an NVIDIA GPU.'),
    'models': ('Detection and segmentation models', 'From assets/models/catalog.json, each with its license '
               'text in the models release.'),
}


def _cell(text):
    return (text or '').replace('|', '\\|').replace('\n', ' ')


def render(data):
    lines = ['# Third-party notices', '',
             'BetaSuite is licensed under AGPL-3.0-or-later (see LICENSE and NOTICE). It is built on the '
             'work below, each under its own license. This file is generated by '
             '`tools/release/third_party_notices.sh`; do not edit it by hand.', '']
    for section in ('rust', 'npm', 'python', 'downloads', 'models'):
        title, intro = TITLES[section]
        rows = data[section]
        lines += ['## %s (%d)' % (title, len(rows)), '', intro, '',
                  '| Name | Version | License | Source |', '|---|---|---|---|']
        for row in rows:
            source = row.get('upstream') or row.get('home') or ''
            lines.append('| %s | %s | %s | %s |' % (_cell(row['name']), _cell(row['version']),
                                                    _cell(row['license']), _cell(source)))
        lines.append('')
    return '\n'.join(lines)


def dump(data):
    return json.dumps(data, indent=1, sort_keys=True) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--python', default=os.environ.get('PYTHON_BIN') or sys.executable)
    ap.add_argument('--pypi', action='store_true', help='look up packages not installed here on PyPI')
    ap.add_argument('--check', action='store_true', help='write nothing; fail when out of date')
    args = ap.parse_args(argv)
    data = {'format': FORMAT, 'rust': rust(), 'npm': npm(), 'python': python(args.python, args.pypi),
            'downloads': downloads(), 'models': models()}
    problems = audit(data)
    for section, name, license in problems:
        say('ERROR', 'license not allowed: %s %s (%s)' % (section, name, license or 'none found'))
    text, md = dump(data), render(data)
    if args.check:
        stale = [p for p, want in ((DATA, text), (MARKDOWN, md))
                 if not os.path.isfile(p) or open(p, encoding='utf-8').read() != want]
        for path in stale:
            say('ERROR', '%s is out of date' % os.path.relpath(path, REPO))
        return 1 if problems or stale else 0
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    with open(MARKDOWN, 'w', encoding='utf-8', newline='\n') as f:
        f.write(md)
    say('INFO', 'wrote %s and %s' % (os.path.relpath(DATA, REPO), os.path.relpath(MARKDOWN, REPO)))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
