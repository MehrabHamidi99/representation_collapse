#!/usr/bin/env python3
"""Plan and migrate Markdown documentation, retaining regular-file redirects.

Only doc/ and docs/ are relocated. Repository/skill entrypoints stay discoverable.
No network calls, job operations, or edits to execution code/configuration.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote, unquote

MARKER = '<!-- documentation-layout: redirect -->'
CLUSTERS = {'local', 'cross-cluster', 'mila', 'tamia', 'narval', 'rorqual', 'fir',
            'nibi', 'trillium', 'trillium-gpu', 'vulcan', 'killarney'}
TYPES = {'theory', 'experiment', 'report', 'auto-research', 'submitted-jobs',
         'guidelines', 'literature', 'planning'}
SKIP = {'.git', '.venv', 'node_modules', '__pycache__', '.sync', 'sources', 'document-originals', 'cluster-sources', 'cluster-snapshots'}
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
SUBPROJECTS = {'jepa':'jepa', 'm-jepa':'jepa', 'nexuspretraining':'nexus',
               'nexus-pretraining':'nexus', 'nexus':'nexus', 'opsd':'opsd',
               'text_rl':'text-rl', 'text-rl':'text-rl', 'vit':'vit',
               'textjobs':'text', 'skill-inference':'skill-inference'}


def write(path, text):
    for parent in (path, *path.parents):
        if parent.is_symlink(): raise ValueError(f'Refusing symlink: {parent}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def doc_path(date, cluster, kind, filename, subproject=None):
    day = dt.date.fromisoformat(date)
    for part in (cluster, kind, filename, subproject):
        if part is not None and (not part or '/' in part or '\\' in part or part in {'.', '..'}):
            raise ValueError('Layout components must be single safe path segments')
    if cluster not in CLUSTERS: raise ValueError(f'Unknown cluster: {cluster}')
    if kind not in TYPES: raise ValueError(f'Unknown documentation type: {kind}')
    parts = ['docs', calendar.month_name[day.month], date, cluster]
    if subproject: parts.append(subproject)
    return Path(*parts, kind, filename)


def conforms(relative):
    parts = Path(relative).parts
    if len(parts) < 6 or parts[0] != 'docs' or parts[3] not in CLUSTERS: return False
    try: day = dt.date.fromisoformat(parts[2])
    except ValueError: return False
    return parts[1] == calendar.month_name[day.month] and (parts[4] in TYPES or
           (len(parts) >= 7 and parts[5] in TYPES))


def documents(root, include_unavailable=False):
    for folder in ('doc', 'docs'):
        start = root/folder
        if not start.is_dir() or start.is_symlink(): continue
        for base, dirs, names in os.walk(start, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(base)/d).is_symlink() and not (Path(base)/d/'.git').exists())
            for name in sorted(names):
                p = Path(base)/name
                if p.suffix.lower() == '.md' and not p.is_symlink():
                    if include_unavailable or not (getattr(p.stat(),'st_flags',0) & 0x40000000): yield p


def classify(relative, date):
    parts = Path(relative).parts[1:]
    lower = [p.lower() for p in parts]
    day, date_basis = date, 'migration date; original date not established'
    for part in parts:
        match = re.search(r'(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)', part)
        if match:
            try: dt.date.fromisoformat(match[1]); day = match[1]; date_basis = 'explicit path date'; break
            except ValueError: pass
    if date_basis.startswith('migration'):
        for part in parts:
            match = re.search(r'(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)', part)
            if match:
                try:
                    day = dt.date(*map(int,match.groups())).isoformat(); date_basis = 'explicit compact path date'; break
                except ValueError: pass
    if date_basis.startswith('migration'):
        for part in lower:
            for month, number in MONTHS.items():
                match = re.search(r'(?:^|[_-])'+month+r'[-_ ]?(\d{1,2})(?:st|nd|rd|th)?(?:$|[_.-])', part)
                if match:
                    try:
                        day = dt.date(int(date[:4]), number, int(match[1])).isoformat()
                        date_basis = 'legacy month/day with migration year; verify attribution if older'
                    except ValueError: pass
    clusters = [p for p in lower if p in CLUSTERS]
    stem = Path(parts[-1]).stem.lower()
    if not clusters:
        clusters = [c for c in CLUSTERS-{'local','cross-cluster'} if re.search(r'(?:^|[-_])'+c+r'(?:$|[-_])',stem)]
    cluster = clusters[0] if len(set(clusters)) == 1 else 'cross-cluster' if clusters else 'local'
    if not clusters and ('guides' in lower or 'cluster' in stem): cluster = 'cross-cluster'
    subproject = next((SUBPROJECTS[p] for p in lower if p in SUBPROJECTS), None)
    if any(x in stem for x in ('slurm', 'submission', 'submitted', 'job-ledger')): kind = 'submitted-jobs'
    elif any(p in {'autoresearch','auto','autonomy','auto-research'} for p in lower): kind = 'auto-research'
    elif any(x in stem for x in ('theory','theorem','proof','derivation')): kind = 'theory'
    elif 'guides' in lower or any(x in stem for x in ('guideline','guide','policy','logging','convention')): kind = 'guidelines'
    elif 'literature' in lower: kind = 'literature'
    elif any(x in stem for x in ('protocol','experiment','ablation')): kind = 'experiment'
    elif any(x in stem for x in ('plan','proposal','todo')): kind = 'planning'
    else: kind = 'report'
    residue = []
    for part in parts[:-1]:
        token = part.lower()
        if token in CLUSTERS or token in SUBPROJECTS or token in MONTHS or token in TYPES: continue
        if token in {'guides','autoresearch','auto','autonomy'}: continue
        if re.fullmatch(r'20\d{2}-\d{2}-\d{2}',token): continue
        if any(re.fullmatch(m+r'[-_ ]?\d{1,2}(?:st|nd|rd|th)?',token) for m in MONTHS): continue
        residue.append(part)
    destination = doc_path(day, cluster, kind, parts[-1], subproject)
    destination = destination.parent.joinpath(*residue, destination.name)
    return str(destination), date_basis


def plan(root, date):
    root = root.resolve()
    moves, destinations = [], set()
    manifest = root/'documentation-index.json'
    already = {m['old'] for record in json.loads(manifest.read_text()).get('migrations',[]) for m in record['moves']} if manifest.exists() else set()
    for source in documents(root):
        relative = str(source.relative_to(root))
        text = source.read_text()
        if relative in already or MARKER in text[:150] or conforms(relative): continue
        destination, basis = classify(relative, date)
        # Never collapse two different historical records with the same filename.
        if destination in destinations or (root/destination).exists():
            p = Path(destination)
            destination = str(p.with_name(p.stem+'-'+hashlib.sha256(relative.encode()).hexdigest()[:10]+p.suffix))
        if destination in destinations or (root/destination).exists():
            raise ValueError(f'Unresolved destination collision: {destination}')
        destinations.add(destination)
        moves.append({'old': relative, 'new': destination, 'date_basis': basis,
                      'sha256_before': hashlib.sha256(text.encode()).hexdigest()})
    return moves


def rewrite_links(text, source, destination, root, mapping):
    def convert(target):
        raw = target.strip('<>')
        if not raw or raw.startswith(('#','http:','https:','mailto:','app:')): return target
        # Leave title strings untouched; handle normal and angle-bracket targets.
        suffix = ''
        if '#' in raw: raw, fragment = raw.split('#',1); suffix = '#'+fragment
        if not raw: return target
        path = Path(unquote(raw))
        resolved = path if path.is_absolute() else (source.parent/path).resolve()
        try: relative = str(resolved.relative_to(root))
        except ValueError:
            # Relative cross-project links must keep their original destination too.
            if path.is_absolute(): return target
            return quote(os.path.relpath(resolved,destination.parent),safe='/._-')+suffix
        final = root/mapping.get(relative,relative)
        return quote(os.path.relpath(final,destination.parent),safe='/._-')+suffix
    text = re.sub(r'(?<=\]\()(<[^>]+>|[^\s)]+)(?=\))',lambda m:convert(m[0]),text)
    text = re.sub(r'(?m)^(\s*\[[^\]]+\]:\s*)(<[^>]+>|\S+)',lambda m:m[1]+convert(m[2]),text)
    # Repository-root literal paths in prose/code spans; do not modify binaries,
    # scientific JSON manifests or executable sources.
    pattern = re.compile(r'(?<![\w/])(?:doc|docs)/[^\s`"\'<>\[\]()]+')
    return pattern.sub(lambda m:mapping.get(m[0],m[0]),text)


def migrate(root, date, apply=False):
    root = root.resolve()
    pending = root/'.documentation-migration.pending.json'
    saved = json.loads(pending.read_text()) if pending.exists() else None
    moves = saved['moves'] if saved else plan(root,date)
    if saved and saved['date'] != date: raise ValueError('Resume with original migration date')
    mapping = {m['old']:m['new'] for m in moves}
    unavailable = [str(p.relative_to(root)) for p in documents(root,True) if getattr(p.stat(),'st_flags',0) & 0x40000000]
    if not apply: return {'project':str(root),'moves':moves,'unavailable':unavailable,'applied':False}
    if not moves: return {'project':str(root),'moves':[],'unavailable':unavailable,'applied':True}
    # Read all originals before writing; reject concurrent edits before mutation.
    originals = saved['originals'] if saved else {m['old']:(root/m['old']).read_text() for m in moves}
    if not saved:
        # Machine-readable provenance contracts may hash these exact legacy bytes.
        code = []
        for folder in ('scripts', 'src', 'configs', 'code', 'autoresearch'):
            base = root/folder
            if not base.exists(): continue
            for current, dirs, names in os.walk(base, followlinks=False):
                dirs[:] = [d for d in dirs if d not in SKIP and not (Path(current)/d).is_symlink()]
                for name in names:
                    f = Path(current)/name
                    if not f.is_symlink() and f.suffix in {'.py','.sh','.json','.yaml','.yml','.toml'} and f.stat().st_size < 2_000_000:
                        code.append(f.read_text(errors='replace'))
        contracts = '\n'.join(code)
        for move in moves:
            move['legacy'] = 'frozen-runtime-copy' if move['old'] in contracts else 'redirect'
        write(pending, json.dumps({'date':date,'moves':moves,'originals':originals}))
    for move in moves:
        if hashlib.sha256(originals[move['old']].encode()).hexdigest()!=move['sha256_before']:
            raise ValueError('Source changed during migration planning; replan')
    for move in moves:
        old, new = root/move['old'], root/move['new']
        current = old.read_text()
        if current != originals[move['old']] and not (MARKER in current[:150] and move['new'] in current):
            raise ValueError(f'Source changed during migration: {old}')
        content = rewrite_links(originals[move['old']],old,new,root,mapping)
        if new.exists() and new.read_text() != content:
            raise ValueError(f'Migration destination changed: {new}')
        write(new,content)
        move['sha256_after'] = hashlib.sha256(content.encode()).hexdigest()
    # Update current human-facing references, including skill instructions. Code
    # continues through legacy redirects until its owner updates a dynamic writer.
    references = [root/'README.md',root/'AGENTS.md',root/'RESEARCH_TEMPLATE.md']
    references += list(documents(root))
    for folder in ('skills','.agents/skills','.cursor/skills'):
        base = root/folder
        if base.exists():
            references += [p for p in base.rglob('*.md') if not p.is_symlink() and 'cluster-sources' not in p.parts]
    moved = {root/m['old'] for m in moves} | {root/m['new'] for m in moves}
    for path in set(references)-moved:
        if not path.is_file() or path.is_symlink(): continue
        before = path.read_text()
        after = rewrite_links(before,path,path,root,mapping)
        if before!=after: write(path,after)
    for move in moves:
        old, new = root/move['old'], root/move['new']
        if move.get('legacy') == 'frozen-runtime-copy': continue
        link = quote(os.path.relpath(new,old.parent),safe='/._-')
        write(old,f'{MARKER}\n# Documentation moved\n\nRead [the canonical document]({link}).\n\nCanonical repository path: `{move["new"]}`.\nThis compatibility pointer is not a second documentation source.\n')
    # Only live project navigation fields change; experiment/config pins are untouched.
    config = root/'project.json'
    if config.is_file() and not config.is_symlink():
        data = json.loads(config.read_text())
        for key in ('protocol','tracker'):
            if data.get(key) in mapping: data[key] = mapping[data[key]]
        data['documentation'] = {'layout':'docs/MONTH/DAY/CLUSTER/[SUBPROJECT]/TYPE/',
                                 'month_format':'English month name','day_format':'YYYY-MM-DD',
                                 'timezone':data.get('timezone','America/Toronto')}
        data['notebook']='docs/MONTH/DAY/CLUSTER/[SUBPROJECT]/report/'
        write(config,json.dumps(data,indent=2)+'\n')
    manifest=root/'documentation-index.json'
    state=json.loads(manifest.read_text()) if manifest.exists() else {'schema_version':1,'migrations':[]}
    if not any(record['moves'] == moves for record in state['migrations']):
        state['migrations'].append({'date':date,'moves':moves})
    write(manifest,json.dumps(state,indent=2)+'\n')
    pending.unlink()
    return {'project':str(root),'moves':moves,'unavailable':unavailable,'applied':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path.cwd())
    p.add_argument('--date',default=dt.date.today().isoformat())
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    result=migrate(args.project,args.date,args.apply)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
