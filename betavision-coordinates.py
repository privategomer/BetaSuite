"""
betavision-coordinates.py

Tiny standalone diagnostic tool: continuously prints the mouse cursor's
current screen coordinates. Used while tuning betaconfig.py's
vision_cap_left/vision_cap_top/vision_cap_width/vision_cap_height (the
screen region BetaVision captures) - run this, hover the mouse over the
corners of the region you want to capture, and read off the printed
x/y values. Windows-only (win32gui).
"""
import win32gui, win32ui


def get_cursor_position():
    """
    Returns the current mouse cursor position as an (x, y) pixel pair,
    in screen coordinates.
    """
    _flags, _hcursor, (cursor_x, cursor_y) = win32gui.GetCursorInfo()
    return cursor_x, cursor_y


def main():
    """Entry point: print the cursor position in a tight loop, forever."""
    while True:
        cursor_x, cursor_y = get_cursor_position()
        print( 'x: %4d, y: %4d'%( cursor_x, cursor_y ) )


if __name__ == '__main__':
    main()
