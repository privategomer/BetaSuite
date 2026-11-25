// The visual editor for a censor style, in the settings' style slot
// (settings/structured.tsx). Its fields come from the engine's style schema
// (config.schema styles), so a new style type or field needs no change
// here. Above it, the value is drawn by the engine onto the label's tile
// (preview.look, the value sent as a per-run override), so what is seen is
// what a run renders.
import { useId, useMemo } from 'react';
import { hasMessage, msgKey, t } from '../i18n/format';
import type { EditorProps } from '../settings/editors';
import type { FieldSchema } from '../settings/model';
import type { PreviewRequest } from './api';
import { ART_STYLE_PATH, DEFAULT_ART_STYLE, figureItem } from './model';
import { usePreviews } from './usePreviews';

type Style = Readonly<Record<string, unknown>>;

const NEW_STYLE: Style = { type: 'blur', method: 'triple_box', strength: 65 };
// Keys the editor shows on every style, after its type's own.
const COMMON_ORDER = [ 'shape', 'feather', 'merge', 'width_area_safety', 'height_area_safety' ];

function isStyle( value: unknown ): value is Style {
  return typeof value === 'object' && value !== null && !Array.isArray( value );
}

function stylesOf( value: unknown ): Style[] {
  if ( Array.isArray( value ) ) {
    return value.filter( isStyle );
  }
  return isStyle( value ) ? [ value ] : [ NEW_STYLE ];
}

function fieldLabel( name: string, field: FieldSchema ): string {
  const key = msgKey( 'setting.', field.label.replace( /^setting\./, '' ) );
  return hasMessage( key ) ? t( key ) : name;
}

function hex( value: unknown ): string {
  const rgb = Array.isArray( value ) && value.length === 3 ? value.map( Number ) : [ 0, 0, 0 ];
  return `#${rgb.map( ( part ) => Math.max( 0, Math.min( 255, part ) ).toString( 16 ).padStart( 2, '0' ) ).join( '' )}`;
}

function rgb( text: string ): number[] {
  return [ 1, 3, 5 ].map( ( start ) => parseInt( text.slice( start, start + 2 ), 16 ) );
}

interface FieldProps {
  readonly name: string;
  readonly field: FieldSchema;
  readonly value: unknown;
  readonly disabled: boolean;
  readonly onChange: ( value: unknown ) => void;
}

function Field( { name, field, value, disabled, onChange }: FieldProps ) {
  const id = useId();
  let input;
  switch ( field.kind ) {
    case 'choice':
      input = (
        <select id={id} value={typeof value === 'string' ? value : ''} disabled={disabled}
          onChange={( event ) => { onChange( event.target.value === '' ? undefined : event.target.value ); }}>
          <option value="">{t( 'hide.editor.default' )}</option>
          {( field.choices ?? [] ).map( ( choice ) => <option key={String( choice )} value={String( choice )}>{String( choice )}</option> )}
        </select>
      );
      break;
    case 'color':
      input = <input id={id} type="color" value={hex( value )} disabled={disabled} onChange={( event ) => { onChange( rgb( event.target.value ) ); }} />;
      break;
    case 'int':
    case 'number':
      input = (
        <input id={id} type="number" disabled={disabled}
          step={field.kind === 'int' ? 1 : 'any'} // catalog-ignore: the HTML step token for any decimal, not text a user sees
          min={field.minimum} max={field.maximum}
          value={typeof value === 'number' ? value : ''}
          onChange={( event ) => { onChange( event.target.value === '' ? undefined : Number( event.target.value ) ); }} />
      );
      break;
    case 'paths':
      input = (
        <input id={id} type="text" disabled={disabled} value={Array.isArray( value ) ? value.join( ', ' ) : ''}
          onChange={( event ) => {
            const parts = event.target.value.split( ',' ).map( ( part ) => part.trim() ).filter( ( part ) => part !== '' );
            onChange( parts.length === 0 ? undefined : parts );
          }} />
      );
      break;
    default:
      input = (
        <input id={id} type="text" disabled={disabled} value={typeof value === 'string' ? value : ''}
          onChange={( event ) => { onChange( event.target.value === '' ? undefined : event.target.value ); }} />
      );
  }
  return (
    <div className="style-field">
      <label htmlFor={id}>{fieldLabel( name, field )}</label>
      {input}
    </div>
  );
}

export function StyleEditor( props: EditorProps ) {
  const { schema, value, onChange, disabled, context, path } = props;
  const styles = stylesOf( value );
  const many = Array.isArray( value ) || styles.length > 1;
  const types = Object.keys( schema.styles.types );
  const common = schema.styles.common;

  const emit = ( next: Style[] ) => {
    onChange( next.length === 1 && !many ? next[ 0 ] : next );
  };
  const setField = ( index: number, name: string, fieldValue: unknown ) => {
    emit( styles.map( ( style, at ) => {
      if ( at !== index ) {
        return style;
      }
      const rest = Object.fromEntries( Object.entries( style ).filter( ( [ key ] ) => key !== name ) );
      return fieldValue === undefined ? rest : { ...rest, [ name ]: fieldValue };
    } ) );
  };
  const setType = ( index: number, type: string ) => {
    // A new type keeps only the keys every type shares.
    emit( styles.map( ( style, at ) => ( at === index
      ? { ...Object.fromEntries( Object.entries( style ).filter( ( [ key ] ) => key in common ) ), type }
      : style ) ) );
  };

  // The value as a run would draw it, on the label's tile when it is a
  // label's look, else on the front figure. A list draws one of its looks.
  const label = path?.startsWith( 'item_overrides.' ) === true ? path.split( '.' )[ 1 ] : undefined;
  const styleValue = context.valueOf( ART_STYLE_PATH );
  const artStyle = typeof styleValue === 'string' ? styleValue : DEFAULT_ART_STYLE;
  const requests = useMemo( () => new Map<string, PreviewRequest>( [ [ 'preview', label === undefined
    ? { art: { style: artStyle, item: figureItem( 'female', 'front', 'unclothed' ) } }
    : { art: { style: artStyle, item: `tile.${label}.female` }, labels: [ label ] } ] ] ), [ label, artStyle ] );
  const previews = usePreviews( requests, path === undefined ? {} : { [ path ]: value }, path !== undefined );
  const drawn = previews.drawn.get( 'preview' );

  return (
    <div className="style-editor" id={props.inputId}>
      {drawn !== undefined && (
        <figure className="style-preview">
          <img src={drawn.url} alt={t( 'hide.editor.preview' )} />
          <figcaption>{many ? t( 'hide.editor.preview_list' ) : t( 'hide.editor.preview_one' )}</figcaption>
        </figure>
      )}
      {styles.map( ( style, index ) => {
        const type = typeof style[ 'type' ] === 'string' ? style[ 'type' ] : 'blur';
        const own = schema.styles.types[ type ] ?? {};
        return (
          <fieldset key={index} className="style-card">
            <legend>{t( 'hide.editor.style', { number: index + 1 } )}</legend>
            <div className="style-field">
              <label htmlFor={`${props.inputId}-type-${String( index )}`}>{t( 'hide.editor.type' )}</label>
              <select id={`${props.inputId}-type-${String( index )}`} value={type} disabled={disabled} onChange={( event ) => { setType( index, event.target.value ); }}>
                {types.map( ( choice ) => <option key={choice} value={choice}>{choice}</option> )}
              </select>
            </div>
            {Object.entries( own ).map( ( [ name, field ] ) => (
              <Field key={name} name={name} field={field} value={style[ name ]} disabled={disabled} onChange={( next ) => { setField( index, name, next ); }} />
            ) )}
            {[ ...COMMON_ORDER, ...( many ? [ 'weight' ] : [] ) ].map( ( name ) => {
              const field = common[ name ];
              return field === undefined ? null
                : <Field key={name} name={name} field={field} value={style[ name ]} disabled={disabled} onChange={( next ) => { setField( index, name, next ); }} />;
            } )}
            {styles.length > 1 && (
              <button type="button" disabled={disabled} onClick={() => { emit( styles.filter( ( _style, at ) => at !== index ) ); }}>
                {t( 'hide.editor.remove' )}
              </button>
            )}
          </fieldset>
        );
      } )}
      <button type="button" disabled={disabled} onClick={() => { onChange( [ ...styles.map( ( style ) => ( { weight: 1, ...style } ) ), { ...NEW_STYLE, weight: 1 } ] ); }}>
        {t( 'hide.editor.add' )}
      </button>
    </div>
  );
}
