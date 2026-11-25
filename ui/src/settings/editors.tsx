// One editor per kind of value (model.ts EDITOR_FOR_KIND). Each shows a
// value and reports an edited one; nothing is checked or written here. The
// row runs the change past the engine first and shows what it costs.
import { useState, type ComponentType } from 'react';
import { t } from '../i18n/format';
import type { EditorId, Schema, SettingSchema } from './model';
import { GeometryActionEditor, ProfileMapEditor, PromotionEditor, RefusedPairsEditor, StyleSlot, SuppressionEditor } from './structured';
import { noneMeaning } from './text';

// What an editor may need beyond its own value: the setting paths the
// config holds (to list profiles) and any setting's current value.
export interface EditorContext {
  readonly paths: readonly string[];
  readonly valueOf: ( path: string ) => unknown;
}

export interface EditorProps {
  readonly entry: SettingSchema;
  readonly schema: Schema;
  readonly context: EditorContext;
  readonly value: unknown;
  readonly onChange: ( value: unknown ) => void;
  readonly inputId: string;
  readonly disabled: boolean;
}

function same( a: unknown, b: unknown ): boolean {
  return JSON.stringify( a ) === JSON.stringify( b );
}

function display( value: unknown ): string {
  return typeof value === 'string' ? value : JSON.stringify( value );
}

// Text being typed, which may not parse yet, kept until the value it
// stands for changes from outside (a reset, another instance).
function useDraftText( value: unknown, format: ( value: unknown ) => string ): [ string, ( text: string ) => void ] {
  const [ seen, setSeen ] = useState( value );
  const [ text, setText ] = useState( () => format( value ) );
  if ( !same( seen, value ) ) {
    setSeen( value );
    setText( format( value ) );
  }
  return [ text, setText ];
}

// A nullable setting: a "None" switch beside its editor, worded by what
// None means for it.
function NoneSwitch( { entry, value, onChange, disabled, fallback }: EditorProps & { fallback: unknown } ) {
  const meaning = noneMeaning( entry );
  if ( entry.nullable !== true ) {
    return null;
  }
  return (
    <label className="editor-none">
      <input
        type="checkbox"
        checked={value === null}
        disabled={disabled}
        onChange={( event ) => { onChange( event.target.checked ? null : fallback ); }}
      />
      {meaning ?? t( 'settings.editor.none' )}
    </label>
  );
}

function Toggle( { value, onChange, inputId, disabled }: EditorProps ) {
  return (
    <input
      id={inputId}
      type="checkbox"
      className="editor-toggle"
      checked={Boolean( value )}
      disabled={disabled}
      onChange={( event ) => { onChange( event.target.checked ); }}
    />
  );
}

function NumberEditor( props: EditorProps ) {
  const { entry, value, onChange, inputId, disabled } = props;
  const [ text, setText ] = useDraftText( value, ( current ) => ( typeof current === 'number' ? String( current ) : '' ) );
  const isNone = value === null;
  return (
    <span className="editor-number">
      {!isNone && (
        <input
          id={inputId}
          type="number"
          inputMode={entry.kind === 'int' ? 'numeric' : 'decimal'}
          // catalog-ignore: the HTML step token for any decimal, not text a user sees
          step={entry.kind === 'int' ? 1 : 'any'}
          min={entry.minimum}
          max={entry.maximum}
          value={text}
          disabled={disabled}
          onChange={( event ) => {
            setText( event.target.value );
            const parsed = Number( event.target.value );
            if ( event.target.value.trim() !== '' && Number.isFinite( parsed ) ) {
              onChange( entry.kind === 'int' ? Math.trunc( parsed ) : parsed );
            }
          }}
        />
      )}
      <NoneSwitch {...props} fallback={entry.minimum ?? 0} />
    </span>
  );
}

// Choices of 0 and 1 (gpu_enabled) read as a switch.
function isBinary( choices: readonly unknown[] ): boolean {
  return choices.length === 2 && choices.includes( 0 ) && choices.includes( 1 );
}

function ChoiceEditor( props: EditorProps ) {
  const { entry, value, onChange, inputId, disabled } = props;
  const choices = entry.choices ?? [];
  if ( isBinary( choices ) ) {
    return <Toggle {...props} value={value === 1 || value === true} onChange={( on ) => { onChange( on === true ? 1 : 0 ); }} />;
  }
  const index = choices.findIndex( ( choice ) => same( choice, value ) );
  return (
    <span className="editor-choice">
      {value !== null && (
        <select
          id={inputId}
          value={index}
          disabled={disabled}
          onChange={( event ) => { onChange( choices[ Number( event.target.value ) ] ); }}
        >
          {index < 0 && <option value={-1}>{display( value )}</option>}
          {choices.map( ( choice, position ) => (
            <option key={display( choice )} value={position}>{display( choice )}</option>
          ) )}
        </select>
      )}
      <NoneSwitch {...props} fallback={choices[ 0 ] ?? ''} />
    </span>
  );
}

function TextEditor( props: EditorProps ) {
  const { value, onChange, inputId, disabled } = props;
  return (
    <span className="editor-text">
      {value !== null && (
        <input
          id={inputId}
          type="text"
          spellCheck={false}
          value={typeof value === 'string' ? value : display( value )}
          disabled={disabled}
          onChange={( event ) => { onChange( event.target.value ); }}
        />
      )}
      <NoneSwitch {...props} fallback="" />
    </span>
  );
}

function PathsEditor( props: EditorProps ) {
  const { value, onChange, inputId, disabled } = props;
  const lines = Array.isArray( value ) ? value.map( display ) : [];
  return (
    <span className="editor-paths">
      {value !== null && (
        <textarea
          id={inputId}
          rows={Math.max( 2, lines.length + 1 )}
          spellCheck={false}
          value={lines.join( '\n' )}
          disabled={disabled}
          aria-describedby={`${inputId}-hint`}
          onChange={( event ) => { onChange( event.target.value.split( '\n' ).map( ( line ) => line.trim() ).filter( ( line ) => line !== '' ) ); }}
        />
      )}
      <span id={`${inputId}-hint`} className="editor-hint">{t( 'settings.editor.one_per_line' )}</span>
      <NoneSwitch {...props} fallback={[]} />
    </span>
  );
}

function SizesEditor( props: EditorProps ) {
  const { value, onChange, inputId, disabled } = props;
  const [ text, setText ] = useDraftText( value, ( current ) => ( Array.isArray( current ) ? current.join( ', ' ) : '' ) );
  return (
    <span className="editor-sizes">
      {value !== null && (
        <input
          id={inputId}
          type="text"
          inputMode="numeric"
          value={text}
          disabled={disabled}
          aria-describedby={`${inputId}-hint`}
          onChange={( event ) => {
            setText( event.target.value );
            const parts = event.target.value.split( /[\s,]+/ ).filter( ( part ) => part !== '' );
            const sizes = parts.map( Number );
            if ( sizes.every( ( size ) => Number.isInteger( size ) ) ) {
              onChange( sizes );
            }
          }}
        />
      )}
      <span id={`${inputId}-hint`} className="editor-hint">{t( 'settings.editor.sizes_hint' )}</span>
      <NoneSwitch {...props} fallback={[]} />
    </span>
  );
}

function LabelsEditor( props: EditorProps ) {
  const { entry, schema, value, onChange, disabled } = props;
  const all = ( entry.choices ?? schema.placeholders[ '<label>' ] ?? [] ).map( display );
  const chosen = new Set( Array.isArray( value ) ? value.map( display ) : [] );
  return (
    <span className="editor-labels">
      {value !== null && (
        <span role="group" aria-labelledby={`${props.inputId}-label`} className="label-chips">
          {all.map( ( label ) => (
            <label key={label} className="label-chip">
              <input
                type="checkbox"
                checked={chosen.has( label )}
                disabled={disabled}
                onChange={( event ) => {
                  const next = new Set( chosen );
                  if ( event.target.checked ) {
                    next.add( label );
                  } else {
                    next.delete( label );
                  }
                  onChange( all.filter( ( item ) => next.has( item ) ) );
                }}
              />
              <code>{label}</code>
            </label>
          ) )}
        </span>
      )}
      <NoneSwitch {...props} fallback={[]} />
    </span>
  );
}

function ColorEditor( { value, onChange, inputId, disabled }: EditorProps ) {
  const parts = Array.isArray( value ) && value.length === 3 ? value.map( Number ) : [ 0, 0, 0 ];
  return (
    <span className="editor-color" role="group" aria-labelledby={`${inputId}-label`}>
      {parts.map( ( part, index ) => (
        <input
          // A color is three fixed positions, so the position is its identity.
          key={`part-${String( index )}`}
          id={index === 0 ? inputId : undefined}
          type="number"
          min={0}
          max={255}
          step={1}
          value={part}
          disabled={disabled}
          aria-label={t( 'settings.editor.color_part', { position: index + 1 } )}
          onChange={( event ) => {
            const next = [ ...parts ];
            next[ index ] = Math.trunc( Number( event.target.value ) );
            onChange( next );
          }}
        />
      ) )}
    </span>
  );
}

export const EDITORS: Readonly<Record<EditorId, ComponentType<EditorProps>>> = {
  toggle: Toggle,
  number: NumberEditor,
  choice: ChoiceEditor,
  text: TextEditor,
  path: TextEditor,
  paths: PathsEditor,
  labels: LabelsEditor,
  color: ColorEditor,
  sizes: SizesEditor,
  style_slot: StyleSlot,
  geometry_action: GeometryActionEditor,
  suppression_rules: SuppressionEditor,
  promotion_rules: PromotionEditor,
  refused_pairs: RefusedPairsEditor,
  profile_map: ProfileMapEditor,
};
