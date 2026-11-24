// The theme follows the system unless --theme asked for light or dark.
export function applyTheme( theme: 'light' | 'dark' | null ): void {
  if ( theme === null ) {
    delete document.documentElement.dataset.theme;
  } else {
    document.documentElement.dataset.theme = theme;
  }
}
