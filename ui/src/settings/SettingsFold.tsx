// A screen's settings: its simple ones in the open, the rest behind
// "More options". The host screen decides where this sits and how it
// looks around it; the fold only lists rows, each nested under the
// setting it requires. A screen with no simple settings (Other) lists
// everything with no fold.
import { useId, useState } from 'react';
import { t } from '../i18n/format';
import { nestByParent, type Row, type Screen } from './model';
import { SettingRow } from './SettingRow';
import type { Loaded } from './useSettings';

interface SettingsFoldProps {
  readonly screen: Screen;
  readonly data: Loaded;
  readonly onHelp: ( anchor: string ) => void;
  readonly onChanged: () => void;
}

export function SettingRows( { rows, data, onHelp, onChanged }: { rows: readonly Row[] } & Omit<SettingsFoldProps, 'screen'> ) {
  return (
    <>
      {nestByParent( rows ).map( ( { row, depth } ) => (
        <div key={row.entry.path} className="setting-nest" data-depth={Math.min( depth, 3 )}>
          <SettingRow row={row} data={data} onHelp={onHelp} onChanged={onChanged} />
        </div>
      ) )}
    </>
  );
}

export function SettingsFold( { screen, data, onHelp, onChanged }: SettingsFoldProps ) {
  const [ open, setOpen ] = useState( false );
  const panelId = useId();
  const rows = data.rows.filter( ( row ) => row.entry.screen === screen );
  const simple = rows.filter( ( row ) => row.entry.level === 'simple' );
  const advanced = rows.filter( ( row ) => row.entry.level === 'advanced' );
  if ( simple.length === 0 ) {
    return <SettingRows rows={rows} data={data} onHelp={onHelp} onChanged={onChanged} />;
  }
  return (
    <div className="settings-fold">
      <SettingRows rows={simple} data={data} onHelp={onHelp} onChanged={onChanged} />
      {advanced.length > 0 && (
        <>
          <button type="button" className="fold-toggle" aria-expanded={open} aria-controls={panelId} onClick={() => { setOpen( !open ); }}>
            {open ? t( 'settings.fold.fewer' ) : t( 'settings.fold.more', { count: advanced.length } )}
          </button>
          <div id={panelId} className="fold-panel" hidden={!open}>
            {open && <SettingRows rows={advanced} data={data} onHelp={onHelp} onChanged={onChanged} />}
          </div>
        </>
      )}
    </div>
  );
}
