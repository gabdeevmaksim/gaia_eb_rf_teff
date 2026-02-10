"""
Visualization functions for eclipsing binary analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import polars as pl
from typing import Union, List, Optional, Tuple

# Astronomical visualization imports
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp

# Cache management
from data.cache_manager import CacheManager, compute_histogram_data, compute_sky_plot_data


def setup_plot_style():
    """Set up consistent plotting style for all visualizations."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set consistent parameters
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100
    })


def plot_histograms(
    data: pl.DataFrame, 
    columns: List[str],
    bins: Union[int, str] = 'auto',
    figsize: Tuple[int, int] = (15, 5),
    title_prefix: str = "Distribution of",
    use_cache: bool = True
) -> plt.Figure:
    """
    Plot histograms for specified columns.
    
    Parameters
    ----------
    data : pl.DataFrame
        The data to plot
    columns : list of str
        Column names to plot histograms for
    bins : int or str, default 'auto'
        Number of bins or binning strategy
    figsize : tuple, default (15, 5)
        Figure size (width, height)
    title_prefix : str, default "Distribution of"
        Prefix for subplot titles
    use_cache : bool, default True
        Whether to use cached results if available
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    setup_plot_style()
    
    # Initialize cache manager
    cache = CacheManager() if use_cache else None
    
    # Try to get cached histogram data
    bins_val = 50 if bins == 'auto' else bins
    hist_data = None
    if cache:
        hist_data = cache.get('histograms', data, columns=columns, bins=bins_val)
    
    # Compute if not cached
    if hist_data is None:
        print("🔄 Computing histogram data...")
        hist_data = compute_histogram_data(data, columns, bins=bins_val)
        if cache:
            cache.set('histograms', data, hist_data, columns=columns, bins=bins_val)
    
    # Create plots
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Handle single column case
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        if col in hist_data:
            hist_info = hist_data[col]
            
            # Plot histogram using pre-computed data
            bin_centers = (hist_info['bin_edges'][:-1] + hist_info['bin_edges'][1:]) / 2
            axes[i].bar(bin_centers, hist_info['hist'], 
                       width=np.diff(hist_info['bin_edges']), 
                       alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Formatting
            axes[i].set_xlabel(col.replace('_', ' ').title())
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{title_prefix} {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistics text from cached data
            stats_text = (f'n = {hist_info["count"]:,}/{hist_info["total_count"]:,}\n'
                         f'μ = {hist_info["mean"]:.1f}\n'
                         f'σ = {hist_info["std"]:.1f}')
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_temperature_photometry_histograms(data: pl.DataFrame) -> plt.Figure:
    """
    Plot histograms specifically for temperature and photometry data.
    
    Parameters
    ----------
    data : pl.DataFrame
        The eclipsing binary catalog data
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Find temperature and photometry columns
    temp_cols = [col for col in data.columns if 'teff' in col.lower()]
    phot_g_cols = [col for col in data.columns if 'phot_g' in col.lower() or col.lower() == 'phot_g_mean_mag']
    
    # Use the most relevant columns
    columns_to_plot = []
    if temp_cols:
        columns_to_plot.append(temp_cols[0])  # Use first teff column
    if phot_g_cols:
        columns_to_plot.append(phot_g_cols[0])  # Use first phot_g column
    
    if not columns_to_plot:
        raise ValueError("No temperature (teff) or photometry (phot_g) columns found in data")
    
    return plot_histograms(
        data, 
        columns_to_plot, 
        bins=50,
        figsize=(15, 5),
        title_prefix="Distribution of"
    )


def plot_sky_distribution(
    data: pl.DataFrame,
    ra_col: str = 'ra',
    dec_col: str = 'dec',
    l_col: str = 'l', 
    b_col: str = 'b',
    projection: str = 'mollweide',
    figsize: Tuple[int, int] = (15, 8)
) -> plt.Figure:
    """
    Plot sky distribution of objects in both equatorial and galactic coordinates.
    
    Parameters
    ----------
    data : pl.DataFrame
        The data containing coordinates
    ra_col, dec_col : str
        Column names for RA and Dec (degrees)
    l_col, b_col : str  
        Column names for galactic longitude and latitude (degrees)
    projection : str, default 'mollweide'
        Map projection ('mollweide', 'aitoff', 'hammer')
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   subplot_kw={'projection': projection})
    
    # Extract coordinates (filter to keep only valid coordinate pairs)
    if ra_col in data.columns and dec_col in data.columns:
        # Keep only rows where both RA and Dec are not null
        valid_eq_data = data.filter(
            (pl.col(ra_col).is_not_null()) & 
            (pl.col(dec_col).is_not_null())
        )
        ra = valid_eq_data[ra_col].to_numpy()
        dec = valid_eq_data[dec_col].to_numpy()
        
        # Convert RA from 0-360 to -180 to 180 for mollweide
        ra_plot = np.where(ra > 180, ra - 360, ra)
        
        ax1.scatter(np.radians(ra_plot), np.radians(dec), 
                   s=0.1, alpha=0.6, c='blue')
        ax1.set_title(f'Sky Distribution - Equatorial Coordinates\n({len(ra):,} objects)')
        ax1.set_xlabel('Right Ascension')
        ax1.set_ylabel('Declination')
        ax1.grid(True, alpha=0.3)
    
    # Galactic coordinates
    if l_col in data.columns and b_col in data.columns:
        # Keep only rows where both l and b are not null
        valid_gal_data = data.filter(
            (pl.col(l_col).is_not_null()) & 
            (pl.col(b_col).is_not_null())
        )
        l = valid_gal_data[l_col].to_numpy() 
        b = valid_gal_data[b_col].to_numpy()
        
        # Convert longitude from 0-360 to -180 to 180
        l_plot = np.where(l > 180, l - 360, l)
        
        ax2.scatter(np.radians(l_plot), np.radians(b), 
                   s=0.1, alpha=0.6, c='red')
        ax2.set_title(f'Sky Distribution - Galactic Coordinates\n({len(l):,} objects)')
        ax2.set_xlabel('Galactic Longitude')
        ax2.set_ylabel('Galactic Latitude')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mollweide_sky_map(
    data: pl.DataFrame,
    coord_system: str = 'galactic',
    value_col: Optional[str] = None,
    title: str = "Sky Distribution",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a Mollweide projection sky map using HEALPix.
    
    Parameters
    ----------
    data : pl.DataFrame
        The data with coordinates
    coord_system : str, default 'galactic'
        'galactic' or 'equatorial'
    value_col : str, optional
        Column to use for color-coding points
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    setup_plot_style()
    
    if coord_system == 'galactic':
        lon_col, lat_col = 'l', 'b'
        coord_label = 'Galactic'
    else:
        lon_col, lat_col = 'ra', 'dec' 
        coord_label = 'Equatorial'
    
    if lon_col not in data.columns or lat_col not in data.columns:
        raise ValueError(f"Columns {lon_col} and {lat_col} not found in data")
    
    # Filter data to include only rows with valid coordinates and values
    if value_col and value_col in data.columns:
        # Keep only rows where all required columns are not null
        valid_data = data.filter(
            (pl.col(lon_col).is_not_null()) & 
            (pl.col(lat_col).is_not_null()) & 
            (pl.col(value_col).is_not_null())
        )
        lon = valid_data[lon_col].to_numpy()
        lat = valid_data[lat_col].to_numpy()
        values = valid_data[value_col].to_numpy()
    else:
        # Keep only rows where coordinates are not null
        valid_data = data.filter(
            (pl.col(lon_col).is_not_null()) & 
            (pl.col(lat_col).is_not_null())
        )
        lon = valid_data[lon_col].to_numpy()
        lat = valid_data[lat_col].to_numpy()
        values = None
    
    # Create figure with mollweide projection
    fig = plt.figure(figsize=figsize)
    
    # Convert to radians and adjust longitude range
    lon_rad = np.radians(np.where(lon > 180, lon - 360, lon))
    lat_rad = np.radians(lat)
    
    # Create subplot with mollweide projection
    ax = fig.add_subplot(111, projection='mollweide')
    
    if values is not None:
        scatter = ax.scatter(lon_rad, lat_rad, c=values, s=0.5, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label=value_col.replace('_', ' ').title())
        ax.set_title(f'{title} - {coord_label} Coordinates\n({len(lon):,} objects with {value_col})')
    else:
        ax.scatter(lon_rad, lat_rad, s=0.5, alpha=0.6, c='blue')
        ax.set_title(f'{title} - {coord_label} Coordinates\n({len(lon):,} objects)')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mollweide_sky_map_cached(
    data: pl.DataFrame,
    coord_system: str = 'galactic',
    value_col: Optional[str] = None,
    title: str = "Sky Distribution",
    figsize: Tuple[int, int] = (12, 6),
    sample_size: int = 100000,
    use_cache: bool = True
) -> plt.Figure:
    """
    Create a Mollweide projection sky map with caching for faster repeated use.
    
    Parameters
    ----------
    data : pl.DataFrame
        The data with coordinates
    coord_system : str, default 'galactic'
        'galactic' or 'equatorial'
    value_col : str, optional
        Column to use for color-coding points
    title : str
        Plot title
    figsize : tuple
        Figure size
    sample_size : int, default 100000
        Subsample large datasets for faster plotting
    use_cache : bool, default True
        Whether to use cached results if available
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    setup_plot_style()
    
    # Initialize cache manager
    cache = CacheManager() if use_cache else None
    
    # Try to get cached plot data
    plot_data = None
    if cache:
        plot_data = cache.get('sky_plot', data, coord_system=coord_system, 
                             value_col=value_col, sample_size=sample_size)
    
    # Compute if not cached
    if plot_data is None:
        print("🔄 Computing sky plot data...")
        plot_data = compute_sky_plot_data(data, coord_system, value_col, sample_size)
        if cache:
            cache.set('sky_plot', data, plot_data, coord_system=coord_system, 
                     value_col=value_col, sample_size=sample_size)
    
    # Create figure with mollweide projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='mollweide')
    
    # Convert to radians and adjust longitude range
    lon_rad = np.radians(np.where(plot_data['lon'] > 180, plot_data['lon'] - 360, plot_data['lon']))
    lat_rad = np.radians(plot_data['lat'])
    
    # Plot
    if 'values' in plot_data:
        scatter = ax.scatter(lon_rad, lat_rad, c=plot_data['values'], 
                           s=0.5, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label=plot_data['value_col'].replace('_', ' ').title())
        ax.set_title(f'{title} - {coord_system.title()} Coordinates\n'
                    f'({plot_data["count"]:,} objects with {plot_data["value_col"]})')
    else:
        ax.scatter(lon_rad, lat_rad, s=0.5, alpha=0.6, c='blue')
        ax.set_title(f'{title} - {coord_system.title()} Coordinates\n'
                    f'({plot_data["count"]:,} objects)')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def get_cache_info():
    """Get information about cached analysis results."""
    cache = CacheManager()
    return cache.get_cache_info()


def clear_cache(operation: str = None):
    """
    Clear analysis cache.
    
    Parameters
    ----------
    operation : str, optional
        If provided, only clear cache for this operation.
        If None, clear all cache.
    """
    cache = CacheManager()
    cache.invalidate(operation)
