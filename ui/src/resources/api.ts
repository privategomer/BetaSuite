// The engine's resource, governor, performance, Optimize and storage
// commands, typed. Shapes follow protocol/v1 (hw.detect, resources.watch,
// governor.*, perf.*, storage.*, config.pending, job.restart_with_changes).
import { engineRequest } from '../engine/client';

export interface Reason {
  readonly code: string;
  readonly params: Readonly<Record<string, unknown>>;
}

export interface GovernorState {
  readonly worker_cap: number | null;
  readonly priority: 'normal' | 'low';
  readonly pace: number;
  readonly reasons: readonly Reason[];
  readonly guard: boolean;
  readonly override: { readonly scope: 'run' | 'session'; readonly memory_guard: boolean } | null;
  readonly throttled: boolean;
  readonly planned: number | null;
  readonly running: number | null;
  readonly enabled: boolean;
  // A run is attached; false while the Performance page shows what a run would meet.
  readonly active: boolean;
  // Set while a render runs narrower because memory was short when it started; no override lifts it.
  readonly memory_start_cap?: { readonly workers: number; readonly wanted: number; readonly available_mb: number | null } | null;
  readonly time: number;
  readonly history?: readonly ( GovernorState & { readonly time: number } )[];
}

export interface Sample {
  readonly time: number;
  readonly memory_pressure: number | null;
  readonly memory_available_mb: number | null;
  readonly memory_total_mb: number | null;
  readonly on_battery: boolean | null;
  readonly battery_percent: number | null;
  readonly cpu_percent: number | null;
  readonly other_cpu_percent: number | null;
  readonly own_cpu_percent: number | null;
  readonly input_idle_seconds: number | null;
  readonly gpu_percent: number | null;
  readonly vram_used_mb: number | null;
  readonly vram_total_mb: number | null;
  readonly own_rss_mb: number | null;
  readonly cache_free_bytes: number | null;
  readonly outputs_free_bytes: number | null;
}

export interface Disk {
  readonly kind: string;
  readonly volume: number;
  readonly free_bytes: number;
  readonly total_bytes: number;
}

export interface SafeSetting {
  readonly setting: string;
  readonly value: unknown;
  readonly current: unknown;
  readonly code: string;
  readonly params: Readonly<Record<string, unknown>>;
}

export interface Machine {
  readonly os: { readonly platform: string; readonly release: string | null };
  readonly cpu: { readonly model: string | null; readonly threads: number | null };
  readonly memory: { readonly total_mb: number | null; readonly available_mb: number | null; readonly swap_mb: number | null };
  readonly gpu: { readonly name: string | null; readonly driver: string | null; readonly vram_mb: number | null } | null;
  readonly cuda_usable: boolean;
  readonly battery: boolean;
  readonly disks: readonly Disk[];
  readonly safe_settings: readonly SafeSetting[];
}

export interface Band {
  readonly band: 'sd' | 'hd' | 'fhd' | 'qhd' | 'uhd';
  readonly runs: number;
  readonly detected_runs: number;
  readonly detect_speed: number | null;
  readonly render_fps: number | null;
  readonly total_speed: number | null;
}

export interface Nudge {
  readonly code: string;
  readonly params: Readonly<Record<string, unknown>>;
  readonly action: { readonly kind: 'set' | 'optimize'; readonly values?: Readonly<Record<string, unknown>>; readonly stages?: readonly string[] | null };
}

export interface Evidence {
  readonly runs: number;
  readonly measured: number;
  readonly bands: readonly Band[];
  readonly memory_capped: { readonly runs: number; readonly of: number };
  readonly optimized: { readonly at: number; readonly applied: boolean } | null;
  readonly nudges: readonly Nudge[];
}

export interface PlanStep {
  readonly stage: string;
  readonly run: boolean;
  readonly narrow: boolean;
  readonly code: string;
}

export interface ProposalRow {
  readonly setting: string;
  readonly value: unknown;
  readonly current: unknown;
}

export interface Category {
  readonly id: string;
  readonly kind: string;
  readonly policy: 'safe' | 'costly' | 'resume' | 'shared' | 'kept';
  readonly bytes: number;
  readonly files: number;
}

export interface StorageUsage {
  readonly categories: readonly Category[];
  readonly disks: readonly Disk[];
  readonly cache_source: 'variable' | 'setting' | 'default';
  readonly busy: boolean;
}

export interface PendingChange {
  readonly id: string;
  readonly values: Readonly<Record<string, unknown>>;
  readonly resets: readonly string[];
  readonly at: number;
}

export interface SettingsChange {
  readonly values?: Readonly<Record<string, unknown>>;
  readonly resets?: readonly string[];
}

export const detectMachine = async () => await engineRequest( 'hw.detect' ) as Machine;
export const watchResources = async ( on: boolean ) =>
  await engineRequest( 'resources.watch', { on } ) as { watching: boolean; sample: Sample | null };
export const governorState = async () => await engineRequest( 'governor.get' ) as GovernorState;
export const fullSpeed = async ( scope: 'run' | 'session', memoryGuard = true ) =>
  await engineRequest( 'governor.override', { scope, memory_guard: memoryGuard } ) as GovernorState;
export const clearFullSpeed = async () => await engineRequest( 'governor.override_clear' ) as GovernorState;
export const enableGovernor = async ( on: boolean ) =>
  await engineRequest( 'governor.enable', { on } ) as GovernorState & { saved: 'now' | 'after_jobs' };
export const evidence = async () => await engineRequest( 'perf.evidence' ) as Evidence;
export const optimize = async ( options: { full?: boolean; stages?: readonly string[]; force?: boolean } ) =>
  await engineRequest( 'perf.optimize', { ...options } ) as { plan: readonly PlanStep[]; narrow: readonly string[] };
export const cancelOptimize = async () => await engineRequest( 'perf.optimize_cancel' ) as { cancelled: boolean };
export const applyOptimize = async () => await engineRequest( 'perf.optimize_apply' ) as { applied: readonly string[] };
export const storageUsage = async () => await engineRequest( 'storage.usage' ) as StorageUsage;
export const cleanStorage = async ( categories: readonly string[], confirm: boolean ) =>
  await engineRequest( 'storage.clean', { categories, confirm } ) as
    { freed: readonly { id: string; bytes: number; files: number }[]; trashed: boolean };
export const pendingChanges = async () => await engineRequest( 'config.pending' ) as { pending: readonly PendingChange[] };
export const discardPending = async ( id?: string ) =>
  await engineRequest( 'config.pending_discard', id === undefined ? {} : { id } ) as { discarded: number };

// A change met config.busy: keep it for when the jobs are done.
export async function changeAfterJobs( change: SettingsChange ): Promise<void> {
  if ( change.values !== undefined && Object.keys( change.values ).length > 0 ) {
    await engineRequest( 'config.set', { values: change.values, when: 'after_jobs' } );
  }
  if ( change.resets !== undefined && change.resets.length > 0 ) {
    await engineRequest( 'config.reset', { paths: change.resets, when: 'after_jobs' } );
  }
}

// A change met config.busy: pause and apply, or cancel and start again with it.
export const restartWithChange = async ( change: SettingsChange, mode: 'resume' | 'requeue' ) =>
  await engineRequest( 'job.restart_with_changes', { mode, values: change.values ?? {}, resets: change.resets ?? [] } ) as
    { mode: string; jobs: readonly string[] };
