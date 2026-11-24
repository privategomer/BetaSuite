// Lint for the desktop UI. The catalog check (scripts/check-catalog.mjs)
// covers user-facing text; this covers code quality and accessibility.
import js from '@eslint/js';
import { defineConfig } from 'eslint/config';
import jsxA11y from 'eslint-plugin-jsx-a11y-x';
import reactHooks from 'eslint-plugin-react-hooks';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default defineConfig(
  { ignores: [ 'dist/', 'src-tauri/', 'node_modules/', 'scripts/catalog-fixtures/' ] },
  {
    files: [ 'src/**/*.{ts,tsx}', 'vite.config.ts' ],
    extends: [
      js.configs.recommended,
      tseslint.configs.strictTypeChecked,
      reactHooks.configs.flat[ 'recommended-latest' ],
      jsxA11y.configs.strict,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    rules: {
      '@typescript-eslint/restrict-template-expressions': [ 'error', { allowNumber: true } ],
    },
  },
  {
    files: [ '**/*.{js,mjs}' ],
    extends: [ js.configs.recommended ],
    languageOptions: { globals: globals.node },
  },
);
