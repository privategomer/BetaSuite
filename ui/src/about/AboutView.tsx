// The About screen: the version, an adults-only notice, a plain statement
// that detection misses things, the license, the project's notices and the
// generated third-party notices (tools/release/third_party_notices.sh).
// The three documents are bundled into the page, like the help text, so
// About works offline. The legal texts are shown as written, untranslated;
// every other word comes from the catalog. Markdown is rendered with raw
// HTML off, so the notices can only produce markdown's own elements.
import markdownIt from 'markdown-it';
import { useMemo } from 'react';
import license from '../../../LICENSE?raw';
import notice from '../../../NOTICE?raw';
import thirdParty from '../../../THIRD_PARTY_NOTICES.md?raw';
import { useEngineStatus } from '../engine/status';
import { t } from '../i18n/format';

export function AboutView() {
  const engine = useEngineStatus();
  const notices = useMemo( () => markdownIt( { html: false, linkify: false, typographer: false } ).render( thirdParty ), [] );
  return (
    <section className="view about-view" aria-labelledby="about-title">
      <h2 id="about-title" className="models-title">{t( 'about.title' )}</h2>
      <p className="view-intro">
        {engine.state === 'ready' ? t( 'about.version', { version: engine.version } ) : t( 'about.version_unknown' )}
      </p>

      <section className="about-notice" aria-labelledby="about-adults">
        <h3 id="about-adults">{t( 'about.adults.title' )}</h3>
        <p>{t( 'about.adults.body' )}</p>
      </section>

      <section className="about-notice" aria-labelledby="about-imperfect">
        <h3 id="about-imperfect">{t( 'about.imperfect.title' )}</h3>
        <p>{t( 'about.imperfect.body' )}</p>
      </section>

      <section aria-labelledby="about-license">
        <h3 id="about-license">{t( 'about.license.title' )}</h3>
        <p>{t( 'about.license.body' )}</p>
        <details className="about-document">
          <summary>{t( 'about.license.full' )}</summary>
          <pre lang="en">{license}</pre>
        </details>
        <details className="about-document">
          <summary>{t( 'about.notice' )}</summary>
          <pre lang="en">{notice}</pre>
        </details>
        <details className="about-document">
          <summary>{t( 'about.third_party' )}</summary>
          <div lang="en" className="about-third-party" dangerouslySetInnerHTML={{ __html: notices }} />
        </details>
      </section>
    </section>
  );
}
