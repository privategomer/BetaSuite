// The feedback.* commands the player uses, typed. Shapes are the engine's
// (protocol/v1/commands/feedback.*.schema.json). The player names the
// result by its path; the engine finds its item and keys.
import { engineRequest } from '../engine/client';

export type Verdict = 'missed' | 'too_much';

export interface Keys {
  readonly detection: string;
  readonly censor: string;
  readonly encode: string;
}

export interface Mark {
  readonly id: number;
  // Seconds into this result, as the player shows it.
  readonly time: number;
  // Seconds into the source: what is stored.
  readonly at_seconds: number;
  readonly label: string | null;
  readonly verdict: Verdict;
  readonly keys: Keys;
  readonly preview: boolean;
  readonly created_at: number;
}

export interface Result {
  readonly item: string;
  readonly keys: Keys;
  readonly preview: boolean;
  readonly offset: number;
}

export interface Marks {
  // Null when the file is not a result (a source, another file).
  readonly result: Result | null;
  readonly records: readonly Mark[];
  // Every label a mark can be about, in the engine's order.
  readonly labels: readonly string[];
}

export async function listMarks( path: string ): Promise<Marks> {
  return await engineRequest( 'feedback.list', { path } ) as Marks;
}

export async function addMark( path: string, time: number, verdict: Verdict, label: string | null ): Promise<Mark> {
  return await engineRequest( 'feedback.add', { path, time, verdict, label } ) as Mark;
}

export async function removeMark( id: number ): Promise<void> {
  await engineRequest( 'feedback.remove', { id } );
}
