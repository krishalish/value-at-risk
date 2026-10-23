"""Charts for the README: black background, orange accent, JetBrains Mono."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats

BG, TEXT, MUTED, GRID = '#000000', '#f4f1ee', '#8c8985', '#262321'
ORANGE, ORANGE_LIGHT, BLUE, PURPLE, RED = '#d97757', '#f0a58b', '#8fc2ff', '#d4b8ff', '#ff9a8f'

FONT_DIR = Path(__file__).resolve().parent.parent / 'assets' / 'fonts'


def use_theme():
    """Apply the dark theme; falls back to the default monospace font if JetBrains Mono is missing."""
    family = 'monospace'
    for f in FONT_DIR.glob('*.ttf'):
        font_manager.fontManager.addfont(str(f))
        family = 'JetBrains Mono'
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'axes.edgecolor': '#3a2a23', 'axes.labelcolor': MUTED, 'text.color': TEXT,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.grid': True, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.family': family, 'font.size': 9.5, 'axes.titlesize': 11, 'axes.titleweight': 600,
        'axes.titlelocation': 'left', 'legend.frameon': False, 'legend.fontsize': 8.5,
    })


def _title(ax, text):
    # monospace font, so '  ' lines the text up exactly after an orange '>' prompt
    kw = dict(transform=ax.transAxes, fontsize=11, fontweight=600, va='bottom', ha='left')
    ax.text(0, 1.04, '>', color=ORANGE, **kw)
    ax.text(0, 1.04, '  ' + text, color=TEXT, **kw)


def _pct(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))


def plot_backtest(forecasts: pd.DataFrame, path: str, conf: float = 0.99,
                  shade=('2025-04-02', '2025-04-11')):
    """Daily portfolio returns against minus the VaR forecasts, with breaches marked."""
    use_theme()
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=160)
    r = forecasts['return'] * 100
    ax.bar(r.index, r.values, width=1.0, color=np.where(r < 0, '#5a3426', '#3a3936'), linewidth=0)
    series = [('historical', ORANGE, 'historical'), ('normal', BLUE, 'normal'), ('garch', PURPLE, 'GARCH(1,1)')]
    for m, colour, label in series:
        ax.plot(forecasts.index, -forecasts[f'{m}_{conf}'] * 100, color=colour, lw=1.3, label=f'{label} VaR')
    breaches = forecasts[-forecasts['return'] > forecasts[f'normal_{conf}']]
    ax.scatter(breaches.index, breaches['return'] * 100, s=26, color=RED, zorder=5,
               label='breach of normal VaR')
    # keep the axis on the loss side; label any day too big to fit instead of squashing the chart
    top = 8
    ax.set_ylim(min(-12, (-forecasts[f'garch_{conf}'] * 100).min() - 1), top)
    for day, val in r[r > top].items():
        ax.annotate(f'+{val:.1f}% ({day:%-d %b}) ↑', (day, top), xytext=(6, -14), textcoords='offset points',
                    color=MUTED, fontsize=8)
    if shade:
        ax.axvspan(pd.Timestamp(shade[0]), pd.Timestamp(shade[1]), color=ORANGE, alpha=0.10, lw=0)
        ax.annotate('tariff sell-off', (pd.Timestamp(shade[0]), top), xytext=(-6, -14), textcoords='offset points',
                    color=ORANGE_LIGHT, fontsize=8.5, ha='right')
    _title(ax, f'2025 out-of-sample backtest · {conf:.0%} one-day VaR')
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    _pct(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tail(port_returns: pd.Series, path: str, conf: float = 0.99):
    """Left tail of the portfolio return distribution against a fitted normal."""
    use_theme()
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=160)
    r = port_returns.values * 100
    mu, sd = r.mean(), r.std(ddof=1)
    bins = np.linspace(r.min(), r.max(), 120)
    ax.hist(r, bins=bins, density=True, color='#3a3936', label='actual daily returns')
    x = np.linspace(r.min(), r.max(), 600)
    ax.plot(x, stats.norm.pdf(x, mu, sd), color=BLUE, lw=1.4, label='normal with same mean and sd')
    hist_var, norm_var = -np.quantile(r, 1 - conf), stats.norm.ppf(conf) * sd - mu
    ax.axvline(-hist_var, color=ORANGE, lw=1.3, label=f'historical {conf:.0%} VaR  {hist_var:.2f}%')
    ax.axvline(-norm_var, color=BLUE, lw=1.3, ls='--', label=f'normal {conf:.0%} VaR  {norm_var:.2f}%')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, None)
    ax.set_xlim(r.min() - 0.5, 6)
    ax.set_ylabel('density (log scale)')
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    _title(ax, f'fat left tail · equal-weight portfolio {port_returns.index[0].year}–{port_returns.index[-1].year}')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_garch_vol(port_returns: pd.Series, path: str):
    """GARCH(1,1) conditional volatility (annualised) next to absolute returns."""
    from arch import arch_model
    use_theme()
    fit = arch_model(port_returns.values * 100, mean='Constant', vol='Garch', p=1, q=1).fit(disp='off')
    vol = pd.Series(fit.conditional_volatility * np.sqrt(252), index=port_returns.index)
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=160)
    ax.bar(port_returns.index, np.abs(port_returns.values) * 100 * np.sqrt(252), width=1.5,
           color='#3a3936', linewidth=0, label='|daily return|, annualised')
    ax.plot(vol.index, vol.values, color=ORANGE, lw=1.3, label='GARCH(1,1) volatility, annualised')
    _title(ax, 'volatility clustering')
    _pct(ax)
    ax.set_ylim(0, vol.max() * 1.6)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
