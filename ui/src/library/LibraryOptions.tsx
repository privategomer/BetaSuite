// The settings that belong with running and with the library (registry
// screens "run" and "library"), folded under the queue as M9 left room
// for: each shows its cost before it applies, as everywhere else. While a
// job holds the queue the engine refuses changes (config.busy), and each
// row says so.
import { errorText } from '../i18n/errors';
import { t } from '../i18n/format';
import { SettingsFold } from '../settings/SettingsFold';
import { useSettings } from '../settings/useSettings';

export function LibraryOptions( { onHelp }: { onHelp: ( anchor: string ) => void } ) {
  const { settings, reload } = useSettings();
  const data = settings.state === 'ready' ? settings.data : settings.state === 'loading' ? settings.last : null;
  if ( data === null ) {
    return settings.state === 'failed'
      ? <p className="run-problem">{settings.error === null ? t( 'settings.request_failed' ) : errorText( settings.error )}</p>
      : null;
  }
  return (
    <>
      <section className="queue" aria-labelledby="run-options-title">
        <h3 id="run-options-title">{t( 'library.run_options' )}</h3>
        <SettingsFold screen="run" data={data} onHelp={onHelp} onChanged={reload} />
      </section>
      <section className="queue" aria-labelledby="library-options-title">
        <h3 id="library-options-title">{t( 'library.library_options' )}</h3>
        <SettingsFold screen="library" data={data} onHelp={onHelp} onChanged={reload} />
      </section>
    </>
  );
}
