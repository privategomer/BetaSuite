// Editors for the values set as a whole: suppression and promotion rules,
// refused pairs, the path-pattern profile map, a geometry action, and the
// slot where a censor style's visual editor goes. Rule fields come from
// config.schema (rules), each with its own label and tip.
import { createElement, useId, type ComponentType } from 'react';
import { t } from '../i18n/format';
import type { EditorProps } from './editors';
import type { FieldSchema } from './model';
import { settingLabel, settingTip } from './text';

type Rule = Readonly<Record<string, unknown>>;

function isRecord( value: unknown ): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray( value );
}

function rulesOf( value: unknown ): Rule[] {
  return Array.isArray( value ) ? value.filter( isRecord ) : [];
}

function display( value: unknown ): string {
  return typeof value === 'string' ? value : JSON.stringify( value );
}

// --- one field of a rule ---------------------------------------------------

interface FieldProps {
  readonly name: string;
  readonly field: FieldSchema;
  readonly value: unknown;
  readonly onChange: ( value: unknown ) => void;
  readonly disabled: boolean;
}

// An optional field left empty is left out of the rule: the engine's
// checks say when a rule needs it.
function FieldInput( { name, field, value, onChange, disabled }: FieldProps ) {
  const id = useId();
  const label = settingLabel( field );
  const control = ( () => {
    if ( field.choices !== undefined ) {
      const index = field.choices.findIndex( ( choice ) => display( choice ) === display( value ) );
      return (
        <select id={id} value={index} disabled={disabled}
          onChange={( event ) => { const at = Number( event.target.value ); onChange( at < 0 ? undefined : field.choices?.[ at ] ); }}>
          <option value={-1}>{t( 'settings.rules.unset' )}</option>
          {field.choices.map( ( choice, at ) => <option key={display( choice )} value={at}>{display( choice )}</option> )}
        </select>
      );
    }
    if ( field.kind === 'number' || field.kind === 'int' ) {
      return (
        <input id={id} type="number" disabled={disabled} min={field.minimum} max={field.maximum}
          // catalog-ignore: the HTML step token for any decimal, not text a user sees
          step={field.kind === 'int' ? 1 : 'any'}
          value={typeof value === 'number' ? value : ''}
          onChange={( event ) => { onChange( event.target.value.trim() === '' ? undefined : Number( event.target.value ) ); }} />
      );
    }
    return (
      <input id={id} type="text" disabled={disabled} value={value === undefined ? '' : display( value )}
        onChange={( event ) => { onChange( event.target.value === '' ? undefined : event.target.value ); }} />
    );
  } )();
  return (
    <span className="rule-field" data-field={name}>
      <label htmlFor={id} title={settingTip( field )}>{label}</label>
      {control}
    </span>
  );
}

function without( record: Readonly<Record<string, unknown>>, name: string ): Record<string, unknown> {
  return Object.fromEntries( Object.entries( record ).filter( ( [ key ] ) => key !== name ) );
}

function withField( rule: Rule, name: string, value: unknown ): Rule {
  return value === undefined ? without( rule, name ) : { ...rule, [ name ]: value };
}

// --- a list of rules -------------------------------------------------------

interface RuleListProps {
  readonly rules: readonly Rule[];
  readonly fields: Readonly<Record<string, FieldSchema>>;
  readonly onChange: ( rules: Rule[] ) => void;
  readonly disabled: boolean;
  // Fields shown by a nested editor rather than as an input.
  readonly nested?: ( rule: Rule, change: ( rule: Rule ) => void ) => React.ReactNode;
  readonly skip?: readonly string[];
  readonly addLabel: string;
}

function RuleList( { rules, fields, onChange, disabled, nested, skip = [], addLabel }: RuleListProps ) {
  const names = Object.keys( fields ).filter( ( name ) => !skip.includes( name ) );
  return (
    <div className="rule-list">
      {rules.map( ( rule, index ) => {
        const change = ( next: Rule ) => { onChange( rules.map( ( item, at ) => ( at === index ? next : item ) ) ); };
        return (
          // A rule has no identity of its own besides its place in the list.
          <div className="rule" key={`rule-${String( index )}`}>
            {names.map( ( name ) => {
              const field = fields[ name ];
              return field === undefined ? null : (
                <FieldInput key={name} name={name} field={field} value={rule[ name ]} disabled={disabled}
                  onChange={( value ) => { change( withField( rule, name, value ) ); }} />
              );
            } )}
            {nested?.( rule, change )}
            <button type="button" disabled={disabled} onClick={() => { onChange( rules.filter( ( _item, at ) => at !== index ) ); }}>
              {t( 'settings.rules.remove' )}
            </button>
          </div>
        );
      } )}
      <button type="button" disabled={disabled} onClick={() => { onChange( [ ...rules, {} ] ); }}>{addLabel}</button>
    </div>
  );
}

// --- rules keyed by label ---------------------------------------------------

interface ByLabelProps extends EditorProps {
  readonly renderRules: ( rules: readonly Rule[], change: ( rules: Rule[] ) => void ) => React.ReactNode;
}

// { label: [ rule, ... ] }: one section per label, and a picker to add one.
function RulesByLabel( { value, onChange, schema, disabled, renderRules }: ByLabelProps ) {
  const groups = isRecord( value ) ? value : {};
  const labels = schema.placeholders[ '<label>' ] ?? [];
  const unused = labels.filter( ( label ) => !( label in groups ) );
  const addId = useId();
  const set = ( label: string, rules: Rule[] | undefined ) => {
    onChange( rules === undefined || rules.length === 0 ? without( groups, label ) : { ...groups, [ label ]: rules } );
  };
  return (
    <div className="rules-by-label">
      {Object.entries( groups ).map( ( [ label, rules ] ) => (
        <fieldset key={label} className="rule-group">
          <legend>{t( 'settings.rules.for_label', { label } )}</legend>
          {renderRules( rulesOf( rules ), ( next ) => { set( label, next ); } )}
        </fieldset>
      ) )}
      {unused.length > 0 && (
        <span className="rule-add-label">
          <label htmlFor={addId}>{t( 'settings.rules.add_label' )}</label>
          <select id={addId} value="" disabled={disabled}
            onChange={( event ) => { if ( event.target.value !== '' ) { set( event.target.value, [ {} ] ); } }}>
            <option value="">{t( 'settings.rules.pick' )}</option>
            {unused.map( ( label ) => <option key={label} value={label}>{label}</option> )}
          </select>
        </span>
      )}
    </div>
  );
}

export function SuppressionEditor( props: EditorProps ) {
  const fields = props.schema.rules[ 'suppression' ] ?? {};
  return (
    <RulesByLabel {...props} renderRules={( rules, change ) => (
      <RuleList rules={rules} fields={fields} onChange={change} disabled={props.disabled} addLabel={t( 'settings.rules.add' )} />
    )} />
  );
}

export function PromotionEditor( props: EditorProps ) {
  const fields = props.schema.rules[ 'promotion' ] ?? {};
  const requirement = props.schema.rules[ 'promotion_requirement' ] ?? {};
  return (
    <RulesByLabel {...props} renderRules={( rules, change ) => (
      <RuleList rules={rules} fields={fields} onChange={change} disabled={props.disabled} skip={[ 'requires' ]}
        addLabel={t( 'settings.rules.add' )}
        nested={( rule, changeRule ) => (
          <fieldset className="rule-requirements">
            <legend>{t( 'settings.rules.requirements' )}</legend>
            <RuleList rules={rulesOf( rule[ 'requires' ] )} fields={requirement} disabled={props.disabled}
              addLabel={t( 'settings.rules.add_requirement' )}
              onChange={( next ) => { changeRule( withField( rule, 'requires', next.length === 0 ? undefined : next ) ); }} />
          </fieldset>
        )} />
    )} />
  );
}

// --- refused pairs: [ 'label<-suppressor', ... ] ------------------------------

const PAIR = '<-';

export function RefusedPairsEditor( { value, onChange, schema, disabled }: EditorProps ) {
  const pairs = Array.isArray( value ) ? value.map( display ) : [];
  const labels = schema.placeholders[ '<label>' ] ?? [];
  const split = ( pair: string ): [ string, string ] => {
    const [ label = '', suppressor = '' ] = pair.split( PAIR );
    return [ label, suppressor ];
  };
  const set = ( index: number, label: string, suppressor: string ) => {
    onChange( pairs.map( ( pair, at ) => ( at === index ? `${label}${PAIR}${suppressor}` : pair ) ) );
  };
  const picker = ( id: string, current: string, pick: ( value: string ) => void ) => (
    <select id={id} value={current} disabled={disabled} onChange={( event ) => { pick( event.target.value ); }}>
      <option value="">{t( 'settings.rules.pick' )}</option>
      {labels.map( ( label ) => <option key={label} value={label}>{label}</option> )}
    </select>
  );
  return (
    <div className="rule-list">
      {pairs.map( ( pair, index ) => {
        const [ label, suppressor ] = split( pair );
        const id = `pair-${String( index )}`;
        return (
          <div className="rule" key={id}>
            <span className="rule-field">
              <label htmlFor={`${id}-label`}>{t( 'settings.pairs.label' )}</label>
              {picker( `${id}-label`, label, ( next ) => { set( index, next, suppressor ); } )}
            </span>
            <span className="rule-field">
              <label htmlFor={`${id}-by`}>{t( 'settings.pairs.suppressor' )}</label>
              {picker( `${id}-by`, suppressor, ( next ) => { set( index, label, next ); } )}
            </span>
            <button type="button" disabled={disabled} onClick={() => { onChange( pairs.filter( ( _item, at ) => at !== index ) ); }}>
              {t( 'settings.rules.remove' )}
            </button>
          </div>
        );
      } )}
      <button type="button" disabled={disabled} onClick={() => { onChange( [ ...pairs, PAIR ] ); }}>{t( 'settings.pairs.add' )}</button>
    </div>
  );
}

// --- profile map: { glob: profile } or None ---------------------------------------

export function ProfileMapEditor( { value, onChange, context, disabled, entry }: EditorProps ) {
  const map = isRecord( value ) ? Object.entries( value ).map( ( [ pattern, profile ] ) => [ pattern, display( profile ) ] as const ) : [];
  const backend = display( context.valueOf( 'detector_backend.selected' ) );
  const prefix = `detector_backend.${backend}.profiles.variants.`;
  const profiles = [ ...new Set( context.paths.filter( ( path ) => path.startsWith( prefix ) )
    .map( ( path ) => path.slice( prefix.length ).split( '.' )[ 0 ] ?? '' ) ) ].filter( ( name ) => name !== '' ).sort();
  const write = ( rows: ReadonlyArray<readonly [ string, string ]> ) => {
    onChange( rows.length === 0 && entry.nullable === true ? null : Object.fromEntries( rows ) );
  };
  return (
    <div className="rule-list">
      {map.map( ( [ pattern, profile ], index ) => {
        const id = `pattern-${String( index )}`;
        const change = ( next: readonly [ string, string ] ) => { write( map.map( ( row, at ) => ( at === index ? next : row ) ) ); };
        return (
          <div className="rule" key={id}>
            <span className="rule-field">
              <label htmlFor={`${id}-glob`}>{t( 'settings.profile_map.pattern' )}</label>
              <input id={`${id}-glob`} type="text" spellCheck={false} value={pattern} disabled={disabled}
                onChange={( event ) => { change( [ event.target.value, profile ] ); }} />
            </span>
            <span className="rule-field">
              <label htmlFor={`${id}-profile`}>{t( 'settings.profile_map.profile' )}</label>
              <select id={`${id}-profile`} value={profile} disabled={disabled} onChange={( event ) => { change( [ pattern, event.target.value ] ); }}>
                <option value="">{t( 'settings.rules.pick' )}</option>
                {profiles.map( ( name ) => <option key={name} value={name}>{name}</option> )}
              </select>
            </span>
            <button type="button" disabled={disabled} onClick={() => { write( map.filter( ( _row, at ) => at !== index ) ); }}>
              {t( 'settings.rules.remove' )}
            </button>
          </div>
        );
      } )}
      <button type="button" disabled={disabled} onClick={() => { write( [ ...map, [ '', profiles[ 0 ] ?? '' ] ] ); }}>
        {t( 'settings.profile_map.add' )}
      </button>
    </div>
  );
}

// --- geometry action: one action, or one per bound ------------------------------

export function GeometryActionEditor( { value, onChange, schema, disabled, inputId }: EditorProps ) {
  const actions = schema.geometry_actions;
  const bound = ( name: 'min' | 'max' ): string => ( isRecord( value ) ? display( value[ name ] ) : display( value ) );
  const picker = ( name: 'min' | 'max', label: string ) => (
    <span className="rule-field">
      <label htmlFor={`${inputId}-${name}`}>{label}</label>
      <select id={name === 'min' ? inputId : `${inputId}-${name}`} value={bound( name )} disabled={disabled}
        onChange={( event ) => { onChange( { min: bound( 'min' ), max: bound( 'max' ), [ name ]: event.target.value } ); }}>
        {actions.map( ( action ) => <option key={action} value={action}>{action}</option> )}
      </select>
    </span>
  );
  return (
    <span className="editor-geometry">
      {picker( 'min', t( 'settings.geometry.min' ) )}
      {picker( 'max', t( 'settings.geometry.max' ) )}
    </span>
  );
}

// --- the style slot -----------------------------------------------------------

// Visual editors for censor, window and frame styles register here by the
// setting kind they edit (the what-to-hide screen registers censor_style).
// A kind with none summarizes the style; the row still offers reset and
// shows what a change costs.
const registered = new Map<string, ComponentType<EditorProps>>();

export function registerStyleEditor( kind: string, editor: ComponentType<EditorProps> ): void {
  registered.set( kind, editor );
}

function styleSummary( style: Rule ): string {
  const type = display( style[ 'type' ] );
  const rest = Object.entries( style ).filter( ( [ name ] ) => name !== 'type' )
    .map( ( [ name, part ] ) => t( 'settings.style.field', { name, value: display( part ) } ) );
  return [ type, ...rest ].join( ', ' );
}

export function StyleSlot( props: EditorProps ) {
  const editor = registered.get( props.entry.kind );
  if ( editor !== undefined ) {
    return createElement( editor, props );
  }
  const styles = Array.isArray( props.value ) ? rulesOf( props.value ) : isRecord( props.value ) ? [ props.value ] : [];
  return (
    <span className="style-slot">
      <span className="editor-hint">{t( 'settings.style.count', { count: styles.length } )}</span>
      <ul id={props.inputId} className="style-list">
        {styles.map( ( style ) => <li key={styleSummary( style )}><code>{styleSummary( style )}</code></li> )}
      </ul>
      <span className="editor-hint">{t( 'settings.style.edit_elsewhere' )}</span>
      <button type="button" disabled>{t( 'settings.style.edit' )}</button>
    </span>
  );
}
