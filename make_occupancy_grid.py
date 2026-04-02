#!/usr/bin/env python3
"""
Generate and interactively edit a 2-D occupancy grid from a point cloud.
Occupied cells are black; free cells are white.

The cloud is projected onto the XY plane; per-cell density is pushed through
a sigmoid (S-curve) that is normalised to the per-cloud maximum, so clouds
with very different point counts behave consistently.

  High density  → occupied (black)
  Sparse density → free    (white)
  Zero density   → occupied (black, unscanned / unknown region)

Usage:
    python3 make_occupancy_grid.py <file.pcd> [options]

Editor controls:
    Left-click / drag    paint
    Middle-click drag    pan
    Scroll wheel         zoom in / out
    P / B                switch tool: Paint / Bucket
    O / F                cell type: Occupied / Free
    [ / ]                decrease / increase brush radius
    Ctrl+Z               undo
    K                    soften (with confirmation)
    R                    reset view to full extent
    S                    save PNG and quit
    Q                    quit without saving
"""

import argparse
import os
from collections import deque, namedtuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_C = dict(
    fig_bg      = '#EEF2F7',   # sidebar / figure background
    main_bg     = '#FFFFFF',   # canvas background
    hdr_bg      = '#DCE8F5',   # section-header strip
    hdr_accent  = '#1565C0',   # section-header left bar
    hdr_text    = '#0D2B5E',   # section-header label
    radio_bg    = '#EEF2F7',   # radio group background
    radio_idle  = '#AEBFCC',   # unselected radio circle
    radio_sel   = '#1565C0',   # selected radio circle
    radio_edge  = '#607D8B',
    divider     = '#B0C4D8',   # separator lines
    txt_light   = '#FFFFFF',
    # buttons (idle, hover)
    soften = ('#E65100', '#BF360C'),
    undo   = ('#546E7A', '#37474F'),
    save   = ('#2E7D32', '#1B5E20'),
    quit   = ('#B71C1C', '#7F0000'),
    go     = ('#1565C0', '#0D47A1'),
    slider = '#90CAF9',
)


# ---------------------------------------------------------------------------
# PCD I/O  (same reader as visualize_levels.py)
# ---------------------------------------------------------------------------

def parse_pcd_header(f):
    header = {}
    while True:
        line = f.readline().decode("utf-8", errors="replace").strip()
        if line.startswith("#") or not line:
            continue
        key, *values = line.split()
        header[key] = values
        if key == "DATA":
            break
    return header


def read_pcd(path):
    with open(path, "rb") as f:
        header = parse_pcd_header(f)
        raw = f.read()

    fields = header["FIELDS"]
    sizes  = list(map(int, header["SIZE"]))
    types  = header["TYPE"]
    counts = list(map(int, header["COUNT"]))
    n_pts  = int(header["POINTS"][0])

    if header["DATA"][0] != "binary":
        raise ValueError(f"Only binary PCD supported, got: {header['DATA'][0]}")

    dtype_list = []
    for field, size, typ, count in zip(fields, sizes, types, counts):
        np_type = {"F": "f", "I": "i", "U": "u"}[typ] + str(size)
        dtype_list.append((field, np_type, count) if count > 1 else (field, np_type))

    pts = np.frombuffer(raw, dtype=np.dtype(dtype_list), count=n_pts)
    x = pts["x"].astype(np.float64)
    y = pts["y"].astype(np.float64)
    z = pts["z"].astype(np.float64)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[mask], y[mask], z[mask]


# ---------------------------------------------------------------------------
# Plane fitting & alignment  (same as visualize_levels.py)
# ---------------------------------------------------------------------------

def fit_plane_ransac(x, y, z, n_iter=500, threshold=0.05, subsample=80_000):
    pts = np.stack([x, y, z], axis=1)
    rng = np.random.default_rng(42)
    if len(pts) > subsample:
        pts_sub = pts[rng.choice(len(pts), subsample, replace=False)]
    else:
        pts_sub = pts

    best_n_inliers, best_normal, best_d = 0, np.array([0.0, 0.0, 1.0]), 0.0
    for _ in range(n_iter):
        s = rng.choice(len(pts_sub), 3, replace=False)
        p1, p2, p3 = pts_sub[s]
        normal = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-9:
            continue
        normal /= norm_n
        d = normal @ p1
        n_inliers = np.sum(np.abs(pts_sub @ normal - d) < threshold)
        if n_inliers > best_n_inliers:
            best_n_inliers, best_normal, best_d = n_inliers, normal, d

    inliers = pts[np.abs(pts @ best_normal - best_d) < threshold]
    print(f"  RANSAC inliers: {len(inliers):,} / {len(pts):,} "
          f"({100 * len(inliers) / len(pts):.1f}%)")

    centroid = inliers.mean(axis=0)
    _, _, Vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, centroid


def align_to_plane(x, y, z, normal, centroid):
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(normal, target))
    pts = np.stack([x, y, z], axis=1) - centroid

    if sin_a < 1e-9:
        if cos_a < 0:
            pts[:, 2] = -pts[:, 2]
        return pts[:, 0], pts[:, 1], pts[:, 2]

    axis /= sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = cos_a * np.eye(3) + sin_a * K + (1 - cos_a) * np.outer(axis, axis)
    pts = pts @ R.T
    return pts[:, 0], pts[:, 1], pts[:, 2]


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def build_occupancy_grid(x, y, resolution, steepness, midpoint):
    """
    Project points onto the XY plane and classify each cell.

        occ_prob = sigmoid(steepness × (norm_density − midpoint))

    Zero-return cells → occupied (unknown / unscanned region).

    Returns
    -------
    grid  : uint8 (nrows × ncols),  0 = free,  1 = occupied
    x_min, y_min : world coords of the grid origin (bottom-left corner)
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    ncols = int(np.ceil((x_max - x_min) / resolution)) + 1
    nrows = int(np.ceil((y_max - y_min) / resolution)) + 1

    col_idx = np.clip(((x - x_min) / resolution).astype(int), 0, ncols - 1)
    row_idx = np.clip(((y - y_min) / resolution).astype(int), 0, nrows - 1)

    counts = np.zeros((nrows, ncols), dtype=np.float64)
    np.add.at(counts, (row_idx, col_idx), 1)

    max_count = counts.max()
    if max_count == 0:
        return np.ones((nrows, ncols), dtype=np.uint8), x_min, y_min

    occ_prob = np.ones((nrows, ncols), dtype=np.float64)
    has_pts = counts > 0
    t = counts[has_pts] / max_count
    occ_prob[has_pts] = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))

    return (occ_prob >= 0.5).astype(np.uint8), x_min, y_min


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def bresenham_line(r0, c0, r1, c1):
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc


def flood_fill(grid, row, col, value):
    nrows, ncols = grid.shape
    if not (0 <= row < nrows and 0 <= col < ncols):
        return
    original = int(grid[row, col])
    if original == value:
        return
    grid[row, col] = value
    queue = deque([(row, col)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < nrows and 0 <= nc < ncols
                    and int(grid[nr, nc]) == original):
                grid[nr, nc] = value
                queue.append((nr, nc))


# ---------------------------------------------------------------------------
# Interactive editor
# ---------------------------------------------------------------------------

_TOOL_LABELS  = ('Paint  [P]', 'Bucket [B]')
_CELL_LABELS  = ('Occupied [O]', 'Free     [F]')
_SHAPE_LABELS = ('Square', 'Circle')

_UndoState = namedtuple('_UndoState', ['grid', 'x_min', 'y_min', 'res'])


class OccupancyGridEditor:
    """
    Matplotlib-based paint editor for a binary occupancy grid.

    Grid  : 0 = free (white),  1 = occupied (black)
    Display: imshow origin='lower'  (Y increases upward)
    PNG out: vertically flipped; occupied → black (0), free → white (255)
    """

    _MAX_UNDO = 20

    def __init__(self, x, y, steepness, midpoint,
                 grid, x_min, y_min, resolution, output_path):
        self._orig_x    = x
        self._orig_y    = y
        self._steepness = steepness
        self._midpoint  = midpoint

        self.grid  = grid.copy()
        self.x_min = x_min
        self.y_min = y_min
        self.res   = resolution
        self.out   = output_path

        self.tool        = 'paint'
        self.cell_type   = 'occupied'
        self.brush_shape = 'square'
        self.brush_r     = 0
        self.painting    = False
        self.last_rc     = None
        self.panning     = False
        self._pan_x0 = self._pan_y0 = 0
        self._pan_xlim0 = self._pan_ylim0 = None
        self._last_evt   = None
        self._cursor_patch = None
        self._cursor_shape = None
        self._dialog_open  = False
        self._undo_stack   = []   # list of _UndoState

        nrows, ncols = grid.shape
        x_max = x_min + ncols * resolution
        y_max = y_min + nrows * resolution
        self._full_xlim = (x_min, x_max)
        self._full_ylim = (y_min, y_max)

        import matplotlib
        matplotlib.rcParams['toolbar'] = 'None'

        # ---- Figure -------------------------------------------------------
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.patch.set_facecolor(_C['fig_bg'])
        self.fig.canvas.manager.set_window_title('Occupancy Grid Editor')

        SL, SW = 0.012, 0.152   # sidebar left / width

        # Vertical separator between sidebar and canvas
        from matplotlib.lines import Line2D
        self.fig.add_artist(
            Line2D([SL + SW + 0.006, SL + SW + 0.006], [0.01, 0.99],
                   transform=self.fig.transFigure,
                   color=_C['divider'], linewidth=1.5, zorder=50))

        # ---- Sidebar widgets (top → bottom) --------------------------------
        self._make_hdr(SL, SW, 0.93, 'Tool')
        ax_tool = self._sidebar_ax(SL, SW, 0.82, 0.11)
        self.radio_tool = mwidgets.RadioButtons(ax_tool, _TOOL_LABELS, active=0)
        self._style_radio(self.radio_tool, ax_tool)
        self.radio_tool.on_clicked(self._on_radio_tool)

        self._make_hdr(SL, SW, 0.78, 'Cell')
        ax_cell = self._sidebar_ax(SL, SW, 0.67, 0.11)
        self.radio_cell = mwidgets.RadioButtons(ax_cell, _CELL_LABELS, active=0)
        self._style_radio(self.radio_cell, ax_cell)
        self.radio_cell.on_clicked(self._on_radio_cell)

        self._make_hdr(SL, SW, 0.63, 'Shape')
        ax_shape = self._sidebar_ax(SL, SW, 0.52, 0.11)
        self.radio_shape = mwidgets.RadioButtons(ax_shape, _SHAPE_LABELS, active=0)
        self._style_radio(self.radio_shape, ax_shape)
        self.radio_shape.on_clicked(self._on_radio_shape)

        # Brush radius slider
        self._make_hdr(SL, SW, 0.48, 'Brush Size')
        ax_sl = self._sidebar_ax(SL + 0.01, SW - 0.04, 0.43, 0.04)
        self._brush_slider = mwidgets.Slider(
            ax_sl, '', 0, 20, valinit=0, valstep=1,
            color=_C['slider'])
        ax_sl.set_facecolor(_C['fig_bg'])
        try:
            self._brush_slider.vline.set_color(_C['hdr_accent'])
            self._brush_slider.vline.set_linewidth(2)
        except AttributeError:
            pass
        self._brush_slider.label.set_fontsize(8)
        self._brush_slider.label.set_color(_C['hdr_text'])
        self._brush_slider.valtext.set_fontsize(8)
        self._brush_slider.on_changed(self._on_brush_slider)

        # Cell size header + text input + Go button
        self._make_hdr(SL, SW, 0.39, 'Cell size (m)')
        ax_res = self._sidebar_ax(SL, SW * 0.64, 0.34, 0.04)
        ax_res.set_facecolor('#FFFFFF')
        for sp in ax_res.spines.values():
            sp.set_color(_C['divider']); sp.set_linewidth(1)
        self._res_box = mwidgets.TextBox(ax_res, '', initial=f'{resolution:.3f}')
        self._res_box.on_submit(lambda _: self._recalculate(None))

        self._btn_go = self._make_btn(
            SL + SW * 0.66, 0.34, SW * 0.34, 0.04,
            'Go', _C['go'][0], _C['go'][1], fontsize=8)
        self._btn_go.on_clicked(self._recalculate)

        # Divider line
        from matplotlib.lines import Line2D as L2
        self.fig.add_artist(L2(
            [SL, SL + SW], [0.325, 0.325],
            transform=self.fig.transFigure,
            color=_C['divider'], linewidth=1, zorder=50))

        # Action buttons
        self._btn_soften = self._make_btn(
            SL, 0.25, SW, 0.07,
            'Soften  [K]', _C['soften'][0], _C['soften'][1])
        self._btn_soften.on_clicked(self._do_soften)

        self._btn_undo = self._make_btn(
            SL, 0.16, SW, 0.07,
            'Undo  [Ctrl+Z]', _C['undo'][0], _C['undo'][1])
        self._btn_undo.on_clicked(self._undo)

        self._btn_save = self._make_btn(
            SL, 0.07, SW, 0.07,
            'Save & Quit  [S]', _C['save'][0], _C['save'][1])
        self._btn_save.on_clicked(self._save_and_quit)

        self._btn_quit = self._make_btn(
            SL, 0.01, SW, 0.05,
            'Quit  [Q]', _C['quit'][0], _C['quit'][1])
        self._btn_quit.on_clicked(lambda _: plt.close(self.fig))

        # ---- Main image ---------------------------------------------------
        self.ax = self.fig.add_axes([0.20, 0.05, 0.77, 0.90])
        self.ax.set_facecolor(_C['main_bg'])
        cmap = ListedColormap(['white', 'black'])
        self.im = self.ax.imshow(
            self.grid, cmap=cmap, vmin=0, vmax=1,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            interpolation='none',
        )
        self.ax.set_xlabel('X (m)', labelpad=4)
        self.ax.set_ylabel('Y (m)', labelpad=4)
        for sp in self.ax.spines.values():
            sp.set_color(_C['divider'])
        self._update_title()

        # ---- Events -------------------------------------------------------
        c = self.fig.canvas.mpl_connect
        c('button_press_event',   self._on_press)
        c('button_release_event', self._on_release)
        c('motion_notify_event',  self._on_motion)
        c('scroll_event',         self._on_scroll)
        c('key_press_event',      self._on_key)
        c('axes_leave_event',     self._on_axes_leave)

    # -----------------------------------------------------------------------
    # Widget factory helpers
    # -----------------------------------------------------------------------

    def _sidebar_ax(self, left, width, bottom, height):
        """Create a plain sidebar axes with the sidebar background colour."""
        ax = self.fig.add_axes([left, bottom, width, height])
        ax.set_facecolor(_C['fig_bg'])
        ax.set_navigate(False)
        return ax

    def _make_hdr(self, left, width, bottom, label):
        """Section header: accent bar + bold label on a tinted strip."""
        ax = self.fig.add_axes([left, bottom, width, 0.035])
        ax.set_navigate(False)
        ax.set_facecolor(_C['hdr_bg'])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_patch(mpatches.Rectangle(
            (0, 0), 0.040, 1.0,
            facecolor=_C['hdr_accent'], edgecolor='none',
            transform=ax.transAxes, zorder=1))
        ax.text(0.09, 0.5, label.upper(),
                ha='left', va='center', fontsize=7.5,
                fontweight='bold', color=_C['hdr_text'],
                transform=ax.transAxes, zorder=2)

    def _make_btn(self, left, bottom, width, height,
                  label, color, hover, fontsize=9):
        """
        Rounded-corner button backed by a FancyBboxPatch.
        Hover effect via axes_enter / axes_leave events.
        """
        ax = self.fig.add_axes([left, bottom, width, height])
        btn = mwidgets.Button(ax, '')
        ax.patch.set_alpha(0)           # hide default rect
        for sp in ax.spines.values():
            sp.set_visible(False)

        bg = mpatches.FancyBboxPatch(
            (0.05, 0.15), 0.9, 0.7,
            boxstyle='round,pad=0.05',
            facecolor=color, edgecolor='none',
            transform=ax.transAxes, zorder=0, clip_on=False)
        ax.add_patch(bg)

        lbl = ax.text(0.5, 0.5, label,
                      ha='center', va='center',
                      fontsize=fontsize, color=_C['txt_light'],
                      fontweight='medium',
                      transform=ax.transAxes, zorder=1)

        # Capture by value via default args
        def _enter(ev, _ax=ax, _bg=bg, _h=hover):
            if ev.inaxes is _ax:
                _bg.set_facecolor(_h)
                self.fig.canvas.draw_idle()

        def _leave(ev, _ax=ax, _bg=bg, _c=color):
            if ev.inaxes is _ax:
                _bg.set_facecolor(_c)
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('axes_enter_event', _enter)
        self.fig.canvas.mpl_connect('axes_leave_event', _leave)

        btn._btn_bg  = bg
        btn._btn_lbl = lbl
        return btn

    def _style_radio(self, radio, ax):
        """Apply palette styling to a freshly created RadioButtons widget."""
        ax.set_facecolor(_C['radio_bg'])
        for sp in ax.spines.values():
            sp.set_color(_C['divider']); sp.set_linewidth(0.8)

        def _repaint(_lbl=None, _r=radio):
            sel = _r.value_selected
            if hasattr(_r, 'circles'):
                for circle, txt in zip(_r.circles, _r.labels):
                    is_sel = txt.get_text() == sel
                    circle.set_facecolor(_C['radio_sel'] if is_sel else _C['radio_idle'])
                    circle.set_edgecolor(_C['hdr_accent'] if is_sel else _C['radio_edge'])
                    circle.set_linewidth(1.8)
                    txt.set_fontsize(8.5)
                    txt.set_color(_C['hdr_text'])
            else:
                facecolors = []
                edgecolors = []
                for txt in _r.labels:
                    is_sel = txt.get_text() == sel
                    facecolors.append(_C['radio_sel'] if is_sel else _C['radio_idle'])
                    edgecolors.append(_C['hdr_accent'] if is_sel else _C['radio_edge'])
                    txt.set_fontsize(8.5)
                    txt.set_color(_C['hdr_text'])
                if hasattr(_r, '_buttons'):
                    _r._buttons.set_facecolor(facecolors)
                    _r._buttons.set_edgecolor(edgecolors)
                    _r._buttons.set_linewidth(1.8)

        _repaint()
        radio.on_clicked(lambda _: _repaint())

    # -----------------------------------------------------------------------
    # Undo
    # -----------------------------------------------------------------------

    def _save_undo(self):
        state = _UndoState(self.grid.copy(), self.x_min, self.y_min, self.res)
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._refresh_undo_btn()

    def _refresh_undo_btn(self):
        n = len(self._undo_stack)
        txt = f'Undo ({n})  [Ctrl+Z]' if n else 'Undo  [Ctrl+Z]'
        self._btn_undo._btn_lbl.set_text(txt)
        # Dim when stack is empty
        col = _C['undo'][0] if n else '#9E9E9E'
        self._btn_undo._btn_bg.set_facecolor(col)
        self.fig.canvas.draw_idle()

    def _undo(self, _e=None):
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        res_changed = (state.res != self.res or
                       state.x_min != self.x_min or
                       state.y_min != self.y_min)
        self.grid  = state.grid
        self.x_min = state.x_min
        self.y_min = state.y_min
        self.res   = state.res
        nrows, ncols = self.grid.shape
        x_max = self.x_min + ncols * self.res
        y_max = self.y_min + nrows * self.res
        self.im.set_data(self.grid)
        if res_changed:
            self.im.set_extent([self.x_min, x_max, self.y_min, y_max])
            self.ax.set_xlim(self.x_min, x_max)
            self.ax.set_ylim(self.y_min, y_max)
            self._full_xlim = (self.x_min, x_max)
            self._full_ylim = (self.y_min, y_max)
            self._update_title()
            try:
                self._res_box.set_val(f'{self.res:.3f}')
            except Exception:
                pass
            if self._cursor_patch is not None:
                self._cursor_patch.remove()
                self._cursor_patch = None
                self._cursor_shape = None
        self._refresh_undo_btn()
        self.fig.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------

    def _update_title(self):
        nrows, ncols = self.grid.shape
        self.ax.set_title(
            f'Occupancy Grid  —  {nrows}\u00d7{ncols} cells  '
            f'({nrows * self.res:.1f}\u2009m \u00d7 {ncols * self.res:.1f}\u2009m)  '
            f'@  {self.res:.3f} m/cell',
            pad=6, fontsize=10, color=_C['hdr_text'])

    # -----------------------------------------------------------------------
    # Soften action
    # -----------------------------------------------------------------------

    def _do_soften(self, _e=None):
        """
        Three-pass morphological clean-up (4-connected cross kernel):
          close ×2 → fills holes / joins nearby walls
          open  ×2 → removes thin spurs & isolated cells
          close ×1 → repairs over-erosion at edges
        """
        self._save_undo()
        try:
            import scipy.ndimage as ndi
        except ImportError:
            print('scipy required for Soften  (pip install scipy).')
            return
        s = ndi.generate_binary_structure(2, 1)
        g = self.grid.astype(bool)
        g = ndi.binary_closing(g, structure=s, iterations=2)
        g = ndi.binary_opening(g, structure=s, iterations=2)
        g = ndi.binary_closing(g, structure=s, iterations=1)
        self.grid = g.astype(np.uint8)
        self.im.set_data(self.grid)
        self.fig.canvas.draw_idle()
        print('Map softened.')

    # -----------------------------------------------------------------------
    # Brush
    # -----------------------------------------------------------------------

    def _on_brush_slider(self, val):
        self.brush_r = int(round(val))
        if self._last_evt is not None:
            self._update_cursor(self._last_evt)
            self.fig.canvas.draw_idle()

    def _circle_mask(self, cr, cc):
        nrows, ncols = self.grid.shape
        r = self.brush_r
        r0, r1 = max(0, cr - r), min(nrows, cr + r + 1)
        c0, c1 = max(0, cc - r), min(ncols, cc + r + 1)
        RR, CC = np.mgrid[r0:r1, c0:c1]
        mask = (RR - cr) ** 2 + (CC - cc) ** 2 <= r * r
        return RR[mask], CC[mask]

    def _apply_brush(self, row, col, value):
        if self.brush_shape == 'circle':
            rows, cols = self._circle_mask(row, col)
            self.grid[rows, cols] = value
        else:
            nrows, ncols = self.grid.shape
            r0, r1 = max(0, row - self.brush_r), min(nrows, row + self.brush_r + 1)
            c0, c1 = max(0, col - self.brush_r), min(ncols, col + self.brush_r + 1)
            self.grid[r0:r1, c0:c1] = value
        self.im.set_data(self.grid)

    def _paint_segment(self, r0, c0, r1, c1, value):
        for r, c in bresenham_line(r0, c0, r1, c1):
            self._apply_brush(r, c, value)

    # -----------------------------------------------------------------------
    # Brush cursor overlay
    # -----------------------------------------------------------------------

    def _update_cursor(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            if self._cursor_patch is not None:
                self._cursor_patch.set_visible(False)
            return

        col = int((event.xdata - self.x_min) / self.res)
        row = int((event.ydata - self.y_min) / self.res)
        nrows, ncols = self.grid.shape
        col = max(0, min(ncols - 1, col))
        row = max(0, min(nrows - 1, row))

        cx = self.x_min + (col + 0.5) * self.res
        cy = self.y_min + (row + 0.5) * self.res
        half = (self.brush_r + 0.5) * self.res

        kw = dict(fill=False, edgecolor='#00E5FF', linewidth=1.8,
                  linestyle='--', zorder=6)

        if self._cursor_patch is None or self._cursor_shape != self.brush_shape:
            if self._cursor_patch is not None:
                self._cursor_patch.remove()
            if self.brush_shape == 'circle':
                self._cursor_patch = mpatches.Circle((cx, cy), half, **kw)
            else:
                self._cursor_patch = mpatches.Rectangle(
                    (cx - half, cy - half), 2 * half, 2 * half, **kw)
            self.ax.add_patch(self._cursor_patch)
            self._cursor_shape = self.brush_shape
        else:
            if self.brush_shape == 'circle':
                self._cursor_patch.center = (cx, cy)
                self._cursor_patch.set_radius(half)
            else:
                self._cursor_patch.set_xy((cx - half, cy - half))
                self._cursor_patch.set_width(2 * half)
                self._cursor_patch.set_height(2 * half)
            self._cursor_patch.set_visible(True)

    # -----------------------------------------------------------------------
    # Radio callbacks & setters
    # -----------------------------------------------------------------------

    def _on_radio_tool(self, label):
        self.tool = 'paint' if label == _TOOL_LABELS[0] else 'bucket'

    def _on_radio_cell(self, label):
        self.cell_type = 'occupied' if label == _CELL_LABELS[0] else 'free'

    def _on_radio_shape(self, label):
        self.brush_shape = 'square' if label == _SHAPE_LABELS[0] else 'circle'
        if self._cursor_patch is not None:
            self._cursor_patch.remove()
            self._cursor_patch = None
            self._cursor_shape = None

    def _set_tool(self, tool):
        self.tool = tool
        try:
            self.radio_tool.set_active(0 if tool == 'paint' else 1)
        except Exception:
            pass

    def _set_cell_type(self, cell_type):
        self.cell_type = cell_type
        try:
            self.radio_cell.set_active(0 if cell_type == 'occupied' else 1)
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Coordinate helper
    # -----------------------------------------------------------------------

    def _event_to_rc(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None, None
        col = int((event.xdata - self.x_min) / self.res)
        row = int((event.ydata - self.y_min) / self.res)
        nrows, ncols = self.grid.shape
        if not (0 <= row < nrows and 0 <= col < ncols):
            return None, None
        return row, col

    # -----------------------------------------------------------------------
    # Recalculate
    # -----------------------------------------------------------------------

    def _recalculate(self, _e):
        try:
            new_res = float(self._res_box.text.strip())
            if new_res <= 0:
                raise ValueError
        except ValueError:
            print('Invalid resolution — enter a positive number.')
            return

        self._save_undo()   # allow undoing a recalculation
        print(f'Recalculating at {new_res:.4f} m/cell ...')
        grid, x_min, y_min = build_occupancy_grid(
            self._orig_x, self._orig_y,
            resolution=new_res,
            steepness=self._steepness,
            midpoint=self._midpoint,
        )
        nrows, ncols = grid.shape
        x_max, y_max = x_min + ncols * new_res, y_min + nrows * new_res

        self.grid  = grid
        self.x_min = x_min
        self.y_min = y_min
        self.res   = new_res
        self._full_xlim = (x_min, x_max)
        self._full_ylim = (y_min, y_max)

        self.im.set_data(grid)
        self.im.set_extent([x_min, x_max, y_min, y_max])
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self._update_title()

        if self._cursor_patch is not None:
            self._cursor_patch.remove()
            self._cursor_patch = None
            self._cursor_shape = None

        self.fig.canvas.draw_idle()
        print(f'  Done: {nrows}\xd7{ncols}  occupied={grid.mean() * 100:.1f}%')

    # -----------------------------------------------------------------------
    # Mouse / keyboard events
    # -----------------------------------------------------------------------

    def _on_press(self, event):
        if self._dialog_open:
            return
        if event.button == 2:
            self.panning    = True
            self._pan_x0    = event.x
            self._pan_y0    = event.y
            self._pan_xlim0 = list(self.ax.get_xlim())
            self._pan_ylim0 = list(self.ax.get_ylim())
            return
        if event.button != 1 or event.inaxes is not self.ax:
            return
        row, col = self._event_to_rc(event)
        if row is None:
            return
        self._save_undo()
        value = 1 if self.cell_type == 'occupied' else 0
        if self.tool == 'bucket':
            flood_fill(self.grid, row, col, value)
            self.im.set_data(self.grid)
            self.fig.canvas.draw_idle()
        else:
            self.painting = True
            self._apply_brush(row, col, value)
            self.last_rc = (row, col)
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if event.button == 2:
            self.panning = False
            return
        self.painting = False
        self.last_rc  = None

    def _on_motion(self, event):
        self._last_evt = event
        if self.panning:
            self._do_pan(event)
            return
        self._update_cursor(event)
        if self.painting and event.inaxes is self.ax:
            row, col = self._event_to_rc(event)
            if row is not None:
                value = 1 if self.cell_type == 'occupied' else 0
                if self.last_rc is not None:
                    self._paint_segment(self.last_rc[0], self.last_rc[1],
                                        row, col, value)
                else:
                    self._apply_brush(row, col, value)
                self.last_rc = (row, col)
        self.fig.canvas.draw_idle()

    def _on_axes_leave(self, event):
        if event.inaxes is self.ax and self._cursor_patch is not None:
            self._cursor_patch.set_visible(False)
            self.fig.canvas.draw_idle()

    def _do_pan(self, event):
        bbox = self.ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return
        dx = (event.x - self._pan_x0) * (self._pan_xlim0[1] - self._pan_xlim0[0]) / bbox.width
        dy = (event.y - self._pan_y0) * (self._pan_ylim0[1] - self._pan_ylim0[0]) / bbox.height
        self.ax.set_xlim([self._pan_xlim0[0] - dx, self._pan_xlim0[1] - dx])
        self.ax.set_ylim([self._pan_ylim0[0] - dy, self._pan_ylim0[1] - dy])
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 1.0 / 1.25 if event.button == 'up' else 1.25
        cx, cy = event.xdata, event.ydata
        self.ax.set_xlim([cx + (x - cx) * factor for x in self.ax.get_xlim()])
        self.ax.set_ylim([cy + (y - cy) * factor for y in self.ax.get_ylim()])
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if self._dialog_open:
            return
        k = event.key
        if   k == 'p':       self._set_tool('paint')
        elif k == 'b':       self._set_tool('bucket')
        elif k == 'o':       self._set_cell_type('occupied')
        elif k == 'f':       self._set_cell_type('free')
        elif k == 'k':       self._do_soften(None)
        elif k == 'ctrl+z':  self._undo()
        elif k == 'r':
            self.ax.set_xlim(self._full_xlim)
            self.ax.set_ylim(self._full_ylim)
            self.fig.canvas.draw_idle()
        elif k in ('[', 'bracketleft'):
            self._brush_slider.set_val(max(0, self.brush_r - 1))
        elif k in (']', 'bracketright'):
            self._brush_slider.set_val(self.brush_r + 1)
        elif k == 's':       self._save_and_quit(None)
        elif k == 'q':       plt.close(self.fig)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def _save_and_quit(self, _e):
        nrows, ncols = self.grid.shape
        img    = np.flipud(self.grid)
        pixels = ((1 - img) * 255).astype(np.uint8)
        try:
            from PIL import Image
            Image.fromarray(pixels, mode='L').save(self.out)
        except ImportError:
            plt.imsave(self.out, img.astype(float), cmap='gray_r', vmin=0, vmax=1)
        print(f'Saved {nrows}\xd7{ncols} occupancy grid \u2192 {self.out}')
        plt.close(self.fig)

    def run(self):
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Build a 2-D occupancy grid from a point cloud, then edit it '
            'interactively before saving as a PNG.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('pcd',
                        help='Path to the input .pcd file (binary format)')
    parser.add_argument('-o', '--output', default=None, metavar='FILE',
                        help='Output PNG path (default: <pcd_basename>.png)')
    parser.add_argument('-r', '--resolution', type=float, default=0.25,
                        metavar='M', help='Initial grid cell size in metres')
    parser.add_argument('--z_min', type=float, default=None, metavar='M',
                        help='Discard points with Z below this value')
    parser.add_argument('--z_max', type=float, default=None, metavar='M',
                        help='Discard points with Z above this value')
    parser.add_argument('--steepness', type=float, default=10.0, metavar='K',
                        help='Sigmoid steepness: higher = sharper transition')
    parser.add_argument('--midpoint', type=float, default=0.3, metavar='T',
                        help='Sigmoid midpoint as fraction of max cell count (0-1)')
    parser.add_argument('--fit_plane', action='store_true',
                        help='RANSAC-fit a ground plane and re-align before gridding')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.pcd)[0] + '.png'

    print(f'Reading {args.pcd} ...')
    x, y, z = read_pcd(args.pcd)
    print(f'Loaded {len(x):,} valid points')

    if args.fit_plane:
        print('Fitting ground plane (RANSAC) ...')
        normal, centroid = fit_plane_ransac(x, y, z)
        angle_deg = np.degrees(np.arccos(np.clip(float(normal @ [0, 0, 1]), -1, 1)))
        print(f'  Plane normal : {normal}')
        print(f'  Tilt from Z  : {angle_deg:.2f} deg')
        x, y, z = align_to_plane(x, y, z, normal, centroid)
        print('  Cloud re-aligned.')

    print(f'  X: [{x.min():.2f}, {x.max():.2f}]  '
          f'Y: [{y.min():.2f}, {y.max():.2f}]  '
          f'Z: [{z.min():.2f}, {z.max():.2f}]')

    if args.z_min is not None or args.z_max is not None:
        mask = np.ones(len(z), dtype=bool)
        if args.z_min is not None:
            mask &= z >= args.z_min
        if args.z_max is not None:
            mask &= z <= args.z_max
        x, y, z = x[mask], y[mask], z[mask]
        print(f'  After Z filter [{args.z_min}, {args.z_max}]: {len(x):,} points remain')

    print(f'Building occupancy grid  '
          f'(resolution={args.resolution} m,  '
          f'steepness={args.steepness},  midpoint={args.midpoint}) ...')
    grid, x_min, y_min = build_occupancy_grid(
        x, y,
        resolution=args.resolution,
        steepness=args.steepness,
        midpoint=args.midpoint,
    )
    nrows, ncols = grid.shape
    print(f'  Grid size   : {nrows}\xd7{ncols}  '
          f'({nrows * args.resolution:.1f} m \xd7 {ncols * args.resolution:.1f} m)')
    print(f'  Occupied    : {grid.mean() * 100:.1f}%')
    print(f'  Output path : {args.output}')

    print('Opening editor ...')
    editor = OccupancyGridEditor(
        x, y,
        steepness=args.steepness,
        midpoint=args.midpoint,
        grid=grid,
        x_min=x_min,
        y_min=y_min,
        resolution=args.resolution,
        output_path=args.output,
    )
    editor.run()


if __name__ == '__main__':
    main()
