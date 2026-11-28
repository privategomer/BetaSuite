import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// No dev server: the app always loads the built page through its own
// scheme, so there is no listening socket and file:// video behaves in
// development exactly as it does in a release build.
export default defineConfig( {
  plugins: [ react() ],
  clearScreen: false,
  envPrefix: [],
  build: {
    target: 'es2022',
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
} );
