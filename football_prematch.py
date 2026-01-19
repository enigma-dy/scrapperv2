"""
Football Prematch Under 1.5 Goals Predictor
============================================

Predicts Under 1.5 goals for pre-match football in low-scoring leagues.
Uses ensemble ML models with calibration for reliable probability estimates.

Usage: python football_prematch.py

Output CSVs (with timestamp, e.g., 2026-01-18_14-00):
- data/prematch_under15_YYYY-MM-DD_HH-MM.csv         - Predictions
- data/prematch_under15_YYYY-MM-DD_HH-MM.result.csv  - Results template
- data/prematch_team_stats.csv                        - Team statistics
- data/prematch_historical.csv                        - Historical match data
"""

import asyncio
import pandas as pd
import numpy as np
import time
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

warnings.filterwarnings('ignore')

# ML imports
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️ ML libraries not found. Install with: pip install xgboost lightgbm scikit-learn")

# Sofascore imports
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match
from sofascore_wrapper.team import Team
from sofascore_wrapper.league import League

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate timestamp for unique filenames
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
FILENAME_PREFIX = f"prematch_under15_{RUN_TIMESTAMP}"

# Time between API requests to avoid rate limiting
REQUEST_DELAY = 0.5

# Minimum requirements
MIN_TRAINING_SAMPLES = 30
MIN_TEAM_MATCHES = 5

# Statistical fallbacks
DEFAULT_UNDER_1_5_RATE = 0.28
DEFAULT_UNDER_2_5_RATE = 0.50

# Default low-scoring leagues (fallback if dynamic discovery fails)
# Ranked by Under 2.5 Goals % - Top 30 leagues
DEFAULT_LOW_SCORING_LEAGUES = {
    # Top Tier (70%+ Under 2.5)
    268: "Iran Azadegan League",          # 78.92%
    68: "Iran Pro League",                 # 74.18%
    879: "Tanzania Premier League",        # 73.75%
    378: "Nigeria Premier League",         # 72.50%
    288: "South Africa PSL",              # 71.26%
    67: "Tunisia Ligue 1",                 # 71.09%
    544: "Kenya Premier League",           # 70.59%
    141: "Albania Super Liga",             # 70.53%
    66: "Algeria Ligue 1",                 # 69.17%
    156: "Egypt Premier League",           # 66.86%
    
    # High Tier (60-70% Under 2.5)
    177: "Croatia 2. HNL",                 # 64.71%
    679: "Spain Primera RFEF Gr. 2",       # 64.23%
    202: "Italy Serie C Gr. B",            # 63.20%
    395: "South Africa First Division",    # 62.91%
    64: "Morocco Botola Pro",              # 62.80%
    233: "Bosnia-Herzegovina Premier Liga", # 62.10%
    200: "Italy Serie C Gr. A",            # 61.93%
    186: "Greece Super League 2",          # 61.88%
    
    # Medium-High Tier (55-60% Under 2.5)
    338: "Ukraine Persha Liga",            # 59.69%
    240: "Bulgaria Second League",         # 59.29%
    58: "Portugal Segunda",                # 58.84%
    23: "Italy Serie A",                   # 57.08%
    240: "Montenegro CFL",                 # 56.84%
    367: "Malta Premier League",           # 56.40%
    194: "Hungary NB II",                  # 55.83%
    455: "Bahrain Premier League",         # 55.56%
    155: "Belgium First Division A",       # 55.45%
}

# Threshold for low-scoring classification (goals per game)
LOW_SCORING_THRESHOLD = 2.1

# Known football leagues to scan for low-scoring detection
# These are leagues with historically high Under 2.5% rates
FOOTBALL_LEAGUES_TO_SCAN = [
    # Iran (Top 2 in Under 2.5)
    268,  # Iran Azadegan League (78.92%)
    68,   # Iran Pro League (74.18%)
    
    # Africa (High Under 2.5 rates)
    879,  # Tanzania Premier League (73.75%)
    378,  # Nigeria Premier League (72.50%)
    288,  # South Africa PSL (71.26%)
    67,   # Tunisia Ligue 1 (71.09%)
    544,  # Kenya Premier League (70.59%)
    66,   # Algeria Ligue 1 (69.17%)
    156,  # Egypt Premier League (66.86%)
    395,  # South Africa First Division (62.91%)
    64,   # Morocco Botola Pro (62.80%)
    554,  # Uganda Premier League (58.95%)
    
    # Balkans/Eastern Europe
    141,  # Albania Super Liga (70.53%)
    177,  # Croatia 2. HNL (64.71%)
    233,  # Bosnia-Herzegovina Premier Liga (62.10%)
    186,  # Greece Super League 2 (61.88%)
    338,  # Ukraine Persha Liga (59.69%)
    240,  # Bulgaria Second League (59.29%)
    240,  # Montenegro CFL (56.84%)
    
    # Southern Europe/Italy
    200,  # Italy Serie C Gr. A (61.93%)
    202,  # Italy Serie C Gr. B (63.20%)
    23,   # Italy Serie A (57.08%)
    5,    # Italy Serie B (53.51%)
    201,  # Italy Serie C Gr. C (50.73%)
    
    # Spain (Multiple lower tiers)
    679,  # Spain Primera RFEF Gr. 2 (64.23%)
    678,  # Spain Primera RFEF Gr. 1 (56.58%)
    8,    # Spain Primera (53.59%)
    54,   # Spain Segunda (50.58%)
    
    # Portugal
    58,   # Portugal Segunda (58.84%)
    238,  # Portugal Primeira (46.91%)
    
    # Other Europe
    194,  # Hungary NB II (55.83%)
    155,  # Belgium First Division A (55.45%)
    253,  # Romania Liga I (54.67%)
    164,  # Scotland Championship (54.13%)
    153,  # Czech Rep. First League (53.95%)
    240,  # Bulgaria First League (53.95%)
    354,  # Cyprus Div 1 (53.73%)
    338,  # Ukraine Premier League (52.92%)
    71,   # Turkey Super Lig (52.67%)
    164,  # Scotland Premiership (52.65%)
    218,  # Serbia SuperLiga (52.50%)
    182,  # France Ligue 2 (51.71%)
    
    # Middle East
    455,  # Bahrain Premier League (55.56%)
    203,  # UAE Arabian Gulf League (51.74%)
    
    # Other
    367,  # Malta Premier League (56.40%)
]


async def discover_low_scoring_leagues(api: 'SofascoreAPI', 
                                        threshold: float = LOW_SCORING_THRESHOLD,
                                        leagues_to_scan: List[int] = None) -> Dict[int, str]:
    """
    Dynamically discover low-scoring leagues by calculating goals per game.
    
    Uses the League.get_info() and League.standings() methods to calculate
    the average goals per game for each league and filters those below threshold.
    
    Args:
        api: SofascoreAPI instance
        threshold: Goals per game threshold (default 2.1)
        leagues_to_scan: List of league IDs to scan (uses default if None)
    
    Returns:
        Dictionary of {league_id: league_name} for low-scoring leagues
    """
    if leagues_to_scan is None:
        leagues_to_scan = FOOTBALL_LEAGUES_TO_SCAN
    
    low_scoring = {}
    print(f"\n🔍 Discovering low-scoring leagues (threshold: {threshold} goals/game)...")
    
    for league_id in leagues_to_scan:
        try:
            league = League(api, league_id)
            
            # Get current season
            seasons = await league.get_seasons()
            if not seasons:
                continue
            
            current_season = seasons[0]
            season_id = current_season.get('id')
            season_name = current_season.get('name', 'Unknown')
            
            await asyncio.sleep(REQUEST_DELAY)
            
            # Get league info for goals count
            try:
                info = await league.get_info(season_id)
                total_goals = info.get('goals', 0)
                home_wins = info.get('homeTeamWins', 0)
                away_wins = info.get('awayTeamWins', 0)
                draws = info.get('draws', 0)
                total_matches = home_wins + away_wins + draws
                
                if total_matches > 0:
                    goals_per_game = total_goals / total_matches
                    
                    # Get league name from standings or info
                    league_name = season_name.split(' ')[0] if ' ' in season_name else season_name
                    try:
                        standings = await league.standings(season_id)
                        if standings and isinstance(standings, list) and len(standings) > 0:
                            league_name = standings[0].get('tournament', {}).get('uniqueTournament', {}).get('name', league_name)
                    except:
                        pass
                    
                    if goals_per_game < threshold and goals_per_game > 0:
                        low_scoring[league_id] = league_name
                        print(f"   ✓ {league_name}: {goals_per_game:.2f} goals/game ({total_matches} matches)")
                    else:
                        print(f"   ✗ {league_name}: {goals_per_game:.2f} goals/game (above threshold)")
                        
            except Exception as e:
                # If get_info fails, try calculating from standings
                try:
                    standings = await league.standings(season_id)
                    if standings and isinstance(standings, list):
                        total_goals = 0
                        total_matches = 0
                        league_name = "Unknown"
                        
                        for standing in standings:
                            if 'rows' in standing:
                                for row in standing.get('rows', []):
                                    total_goals += row.get('scoresFor', 0) + row.get('scoresAgainst', 0)
                                    total_matches += row.get('matches', 0)
                                league_name = standing.get('tournament', {}).get('uniqueTournament', {}).get('name', 'Unknown')
                        
                        # Divide by 2 since each goal is counted twice (for and against)
                        if total_matches > 0:
                            goals_per_game = (total_goals / 2) / (total_matches / 2)
                            
                            if goals_per_game < threshold and goals_per_game > 0:
                                low_scoring[league_id] = league_name
                                print(f"   ✓ {league_name}: {goals_per_game:.2f} goals/game (from standings)")
                except:
                    pass
                    
            await asyncio.sleep(REQUEST_DELAY)
            
        except Exception as e:
            continue
    
    if not low_scoring:
        print("   ⚠️ No low-scoring leagues found, using defaults")
        return DEFAULT_LOW_SCORING_LEAGUES
    
    print(f"\n✓ Found {len(low_scoring)} low-scoring leagues")
    return low_scoring


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_get(data: dict, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data


def parse_stats(stats_response) -> Dict[str, Any]:
    """Parse match statistics into flat dict."""
    flat = {}
    
    stats_list = []
    if isinstance(stats_response, dict):
        stats_list = stats_response.get('statistics', [])
    elif isinstance(stats_response, list):
        stats_list = stats_response
    
    if not stats_list:
        return flat
    
    # Get ALL period stats, or 1ST if not available
    period_stats = None
    for period in ['ALL', '1ST', '2ND']:
        period_stats = next((s for s in stats_list if s.get('period') == period), None)
        if period_stats:
            break
    
    if not period_stats:
        return flat
    
    for group in period_stats.get('groups', []):
        for item in group.get('statisticsItems', []):
            name = item.get('name', '').lower().replace(' ', '_').replace('-', '_')
            flat[f"home_{name}"] = item.get('homeValue', item.get('home'))
            flat[f"away_{name}"] = item.get('awayValue', item.get('away'))
    
    return flat


# ============================================================================
# DATA COLLECTION
# ============================================================================

class PrematchDataCollector:
    """Collects prematch data for low-scoring leagues."""
    
    def __init__(self):
        self.api = SofascoreAPI()
        self.team_stats_cache = {}
        self.historical_matches = []
    
    async def close(self):
        await self.api.close()
    
    async def get_upcoming_matches(self, league_id: int, league_name: str) -> List[Dict]:
        """Get upcoming matches (not started) for a league."""
        upcoming = []
        
        try:
            league = League(self.api, league_id)
            
            # Get current season
            seasons = await league.get_seasons()
            if not seasons:
                print(f"  ⚠️ No seasons found for {league_name}")
                return []
            
            current_season = seasons[0]  # Most recent season
            season_id = current_season.get('id')
            
            await asyncio.sleep(REQUEST_DELAY)
            
            # Get fixtures for current round (status code 0 = not started)
            # Try multiple rounds to find upcoming matches
            for round_offset in range(0, 5):
                try:
                    fixtures = await league.next_fixtures()
                    if fixtures:
                        for event in fixtures if isinstance(fixtures, list) else fixtures.get('events', []):
                            status_code = safe_get(event, 'status', 'code', default=100)
                            if status_code == 0:  # Not started
                                upcoming.append({
                                    'event': event,
                                    'league_id': league_id,
                                    'league_name': league_name,
                                    'season_id': season_id,
                                })
                    break
                except Exception as e:
                    print(f"    Error getting fixtures: {e}")
                    break
            
            await asyncio.sleep(REQUEST_DELAY)
            
        except Exception as e:
            print(f"  ⚠️ Error getting matches for {league_name}: {e}")
        
        return upcoming
    
    async def get_team_historical_matches(self, team_id: int, team_name: str, 
                                          num_matches: int = 15) -> List[Dict]:
        """Collect historical finished matches for a team."""
        matches = []
        
        try:
            team = Team(self.api, team_id=team_id)
            events_response = await team.last_fixtures()
            events = events_response if isinstance(events_response, list) else events_response.get('events', [])
            
            for event in events[:num_matches]:
                status = safe_get(event, 'status', 'type', default='')
                if status != 'finished':
                    continue
                
                home_score = safe_get(event, 'homeScore', 'current', default=0) or 0
                away_score = safe_get(event, 'awayScore', 'current', default=0) or 0
                total_goals = home_score + away_score
                
                is_home = safe_get(event, 'homeTeam', 'id') == team_id
                
                match_data = {
                    'match_id': event.get('id'),
                    'team_id': team_id,
                    'team_name': team_name,
                    'is_home': is_home,
                    'opponent_id': safe_get(event, 'awayTeam', 'id') if is_home else safe_get(event, 'homeTeam', 'id'),
                    'opponent_name': safe_get(event, 'awayTeam', 'name') if is_home else safe_get(event, 'homeTeam', 'name'),
                    'goals_scored': home_score if is_home else away_score,
                    'goals_conceded': away_score if is_home else home_score,
                    'total_goals': total_goals,
                    'under_1_5': 1 if total_goals < 2 else 0,
                    'under_2_5': 1 if total_goals < 3 else 0,
                    'clean_sheet': 1 if (away_score == 0 if is_home else home_score == 0) else 0,
                    'failed_to_score': 1 if (home_score == 0 if is_home else away_score == 0) else 0,
                    'tournament': safe_get(event, 'tournament', 'name'),
                }
                matches.append(match_data)
                
        except Exception as e:
            print(f"    ⚠️ Error fetching history for {team_name}: {e}")
        
        return matches
    
    def calculate_team_stats(self, matches: List[Dict]) -> Dict[str, float]:
        """Calculate team statistics from historical matches."""
        if not matches or len(matches) < MIN_TEAM_MATCHES:
            return {
                'goals_scored_avg': 1.3,
                'goals_conceded_avg': 1.3,
                'clean_sheet_pct': 0.25,
                'failed_to_score_pct': 0.25,
                'under_1_5_pct': DEFAULT_UNDER_1_5_RATE,
                'under_2_5_pct': DEFAULT_UNDER_2_5_RATE,
                'matches_count': len(matches) if matches else 0,
            }
        
        df = pd.DataFrame(matches)
        
        return {
            'goals_scored_avg': df['goals_scored'].mean(),
            'goals_conceded_avg': df['goals_conceded'].mean(),
            'clean_sheet_pct': df['clean_sheet'].mean(),
            'failed_to_score_pct': df['failed_to_score'].mean(),
            'under_1_5_pct': df['under_1_5'].mean(),
            'under_2_5_pct': df['under_2_5'].mean(),
            'matches_count': len(df),
        }
    
    async def get_h2h_data(self, match_id: int) -> Dict:
        """Get head-to-head data for a match."""
        h2h_stats = {
            'h2h_matches': 0,
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0,
            'h2h_avg_goals': 0,
            'h2h_under_1_5_pct': DEFAULT_UNDER_1_5_RATE,
        }
        
        try:
            match = Match(self.api, match_id=match_id)
            h2h = await match.h2h()
            
            if not h2h:
                return h2h_stats
            
            team_duel = h2h.get('teamDuel', {})
            h2h_stats['h2h_home_wins'] = team_duel.get('homeWins', 0)
            h2h_stats['h2h_away_wins'] = team_duel.get('awayWins', 0)
            h2h_stats['h2h_draws'] = team_duel.get('draws', 0)
            
            # Get H2H events for goal analysis
            events = h2h.get('events', [])
            if events:
                total_goals = 0
                under_1_5_count = 0
                
                for event in events[:10]:  # Last 10 H2H matches
                    home_score = safe_get(event, 'homeScore', 'current', default=0) or 0
                    away_score = safe_get(event, 'awayScore', 'current', default=0) or 0
                    match_goals = home_score + away_score
                    total_goals += match_goals
                    
                    if match_goals < 2:
                        under_1_5_count += 1
                
                n = len(events[:10])
                h2h_stats['h2h_matches'] = n
                h2h_stats['h2h_avg_goals'] = total_goals / n if n > 0 else 0
                h2h_stats['h2h_under_1_5_pct'] = under_1_5_count / n if n > 0 else DEFAULT_UNDER_1_5_RATE
            
        except Exception as e:
            pass  # Use defaults
        
        return h2h_stats
    
    async def collect_all_data(self, low_scoring_leagues: Dict[int, str] = None) -> Tuple[List[Dict], Dict, pd.DataFrame]:
        """Collect all data for prediction.
        
        Args:
            low_scoring_leagues: Dictionary of {league_id: league_name} to collect from.
                                 Uses DEFAULT_LOW_SCORING_LEAGUES if None.
        """
        if low_scoring_leagues is None:
            low_scoring_leagues = DEFAULT_LOW_SCORING_LEAGUES
            
        print("\n" + "="*70)
        print("STEP 2: COLLECTING UPCOMING MATCHES FROM LOW-SCORING LEAGUES")
        print("="*70)
        
        all_upcoming = []
        
        for league_id, league_name in low_scoring_leagues.items():
            print(f"\n  📋 {league_name}...")
            upcoming = await self.get_upcoming_matches(league_id, league_name)
            print(f"     Found {len(upcoming)} upcoming matches")
            all_upcoming.extend(upcoming)
            await asyncio.sleep(REQUEST_DELAY)
        
        print(f"\n✓ Total upcoming matches: {len(all_upcoming)}")
        
        if not all_upcoming:
            return [], {}, pd.DataFrame()
        
        # Collect team data
        print("\n" + "="*70)
        print("STEP 2: COLLECTING TEAM HISTORICAL DATA")
        print("="*70)
        
        team_stats = {}
        processed_teams = set()
        
        for idx, match_info in enumerate(all_upcoming):
            event = match_info['event']
            home_id = safe_get(event, 'homeTeam', 'id')
            home_name = safe_get(event, 'homeTeam', 'name', default='Unknown')
            away_id = safe_get(event, 'awayTeam', 'id')
            away_name = safe_get(event, 'awayTeam', 'name', default='Unknown')
            
            # Home team
            if home_id and home_id not in processed_teams:
                print(f"  [{len(processed_teams)+1}] {home_name}...")
                matches = await self.get_team_historical_matches(home_id, home_name)
                self.historical_matches.extend(matches)
                team_stats[home_id] = self.calculate_team_stats(matches)
                team_stats[home_id]['team_name'] = home_name
                processed_teams.add(home_id)
                await asyncio.sleep(REQUEST_DELAY)
            
            # Away team
            if away_id and away_id not in processed_teams:
                print(f"  [{len(processed_teams)+1}] {away_name}...")
                matches = await self.get_team_historical_matches(away_id, away_name)
                self.historical_matches.extend(matches)
                team_stats[away_id] = self.calculate_team_stats(matches)
                team_stats[away_id]['team_name'] = away_name
                processed_teams.add(away_id)
                await asyncio.sleep(REQUEST_DELAY)
        
        print(f"\n✓ Collected stats for {len(team_stats)} teams")
        
        # Collect H2H data
        print("\n" + "="*70)
        print("STEP 3: COLLECTING HEAD-TO-HEAD DATA")
        print("="*70)
        
        for idx, match_info in enumerate(all_upcoming):
            event = match_info['event']
            match_id = event.get('id')
            home_name = safe_get(event, 'homeTeam', 'name', default='Unknown')
            away_name = safe_get(event, 'awayTeam', 'name', default='Unknown')
            
            print(f"  [{idx+1}/{len(all_upcoming)}] {home_name} vs {away_name}...")
            h2h = await self.get_h2h_data(match_id)
            match_info['h2h'] = h2h
            await asyncio.sleep(REQUEST_DELAY)
        
        # Create historical DataFrame
        df_historical = pd.DataFrame(self.historical_matches)
        if not df_historical.empty:
            df_historical = df_historical.drop_duplicates(subset=['match_id'])
        
        return all_upcoming, team_stats, df_historical


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_prediction_features(match_info: Dict, team_stats: Dict) -> Dict[str, float]:
    """Create features for prediction."""
    event = match_info['event']
    home_id = safe_get(event, 'homeTeam', 'id')
    away_id = safe_get(event, 'awayTeam', 'id')
    h2h = match_info.get('h2h', {})
    
    # Default stats
    default_stats = {
        'goals_scored_avg': 1.3,
        'goals_conceded_avg': 1.3,
        'clean_sheet_pct': 0.25,
        'failed_to_score_pct': 0.25,
        'under_1_5_pct': DEFAULT_UNDER_1_5_RATE,
        'under_2_5_pct': DEFAULT_UNDER_2_5_RATE,
        'matches_count': 0,
    }
    
    home_stats = team_stats.get(home_id, default_stats)
    away_stats = team_stats.get(away_id, default_stats)
    
    # Expected goals
    home_expected_score = (home_stats.get('goals_scored_avg', 1.3) + 
                           away_stats.get('goals_conceded_avg', 1.3)) / 2
    away_expected_score = (away_stats.get('goals_scored_avg', 1.3) + 
                           home_stats.get('goals_conceded_avg', 1.3)) / 2
    expected_total = home_expected_score + away_expected_score
    
    # Combined probabilities
    combined_under_1_5 = (home_stats.get('under_1_5_pct', DEFAULT_UNDER_1_5_RATE) + 
                          away_stats.get('under_1_5_pct', DEFAULT_UNDER_1_5_RATE)) / 2
    
    both_cs_prob = (home_stats.get('clean_sheet_pct', 0.25) * 
                    away_stats.get('clean_sheet_pct', 0.25))
    
    one_fts_prob = (home_stats.get('failed_to_score_pct', 0.25) + 
                    away_stats.get('failed_to_score_pct', 0.25)) / 2
    
    # H2H influence
    h2h_under_1_5 = h2h.get('h2h_under_1_5_pct', DEFAULT_UNDER_1_5_RATE)
    h2h_matches = h2h.get('h2h_matches', 0)
    
    # Blend H2H with team stats (H2H gets more weight if more matches)
    h2h_weight = min(h2h_matches / 10, 0.3)  # Max 30% H2H influence
    blended_under_1_5 = combined_under_1_5 * (1 - h2h_weight) + h2h_under_1_5 * h2h_weight
    
    # Low-scoring league boost
    league_boost = 1.1  # 10% boost for being in low-scoring league
    adjusted_under = min(blended_under_1_5 * league_boost, 0.85)
    
    features = {
        'home_goals_scored_avg': home_stats.get('goals_scored_avg', 1.3),
        'home_goals_conceded_avg': home_stats.get('goals_conceded_avg', 1.3),
        'away_goals_scored_avg': away_stats.get('goals_scored_avg', 1.3),
        'away_goals_conceded_avg': away_stats.get('goals_conceded_avg', 1.3),
        'expected_total_goals': expected_total,
        'home_clean_sheet_pct': home_stats.get('clean_sheet_pct', 0.25),
        'away_clean_sheet_pct': away_stats.get('clean_sheet_pct', 0.25),
        'both_cs_prob': both_cs_prob,
        'home_fts_pct': home_stats.get('failed_to_score_pct', 0.25),
        'away_fts_pct': away_stats.get('failed_to_score_pct', 0.25),
        'one_fts_prob': one_fts_prob,
        'combined_under_1_5': combined_under_1_5,
        'h2h_under_1_5_pct': h2h_under_1_5,
        'h2h_avg_goals': h2h.get('h2h_avg_goals', 2.6),
        'h2h_matches': h2h_matches,
        'adjusted_under_1_5': adjusted_under,
        'home_historical_matches': home_stats.get('matches_count', 0),
        'away_historical_matches': away_stats.get('matches_count', 0),
    }
    
    return features


# ============================================================================
# ML MODEL
# ============================================================================

class CalibratedUnder15Model:
    """Calibrated ensemble model for Under 1.5 predictions."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_accuracy = 0
        self.training_samples = 0
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the model on historical data."""
        print("\n--- Training Calibrated Under 1.5 Model ---")
        
        if len(df) < MIN_TRAINING_SAMPLES:
            print(f"⚠️ Insufficient data: {len(df)} samples (need {MIN_TRAINING_SAMPLES}+)")
            print("   Model will use statistical fallback for predictions.")
            return {}
        
        # Features and target
        self.feature_names = ['goals_scored', 'goals_conceded', 'total_goals']
        existing = [c for c in self.feature_names if c in df.columns]
        
        if not existing:
            print("❌ No valid features found")
            return {}
        
        X = df[existing].fillna(0)
        y = (df['total_goals'] < 2).astype(int)
        
        print(f"Training on {len(df)} matches with {len(existing)} features")
        print(f"Target distribution: Under 1.5 = {y.sum()} ({y.mean():.1%}), Over 1.5 = {len(y)-y.sum()}")
        
        if len(y.unique()) < 2:
            print("❌ Need both classes for training")
            return {}
        
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Base models
        xgb = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / max(1, len(y_train[y_train==1])),
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        
        lgb = LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            class_weight='balanced', verbose=-1
        )
        
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=5,
            class_weight='balanced', random_state=42
        )
        
        # Ensemble
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf)],
            voting='soft'
        )
        
        # Calibrate
        self.model = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
        
        try:
            self.model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"⚠️ Calibration failed, using uncalibrated: {e}")
            ensemble.fit(X_train_scaled, y_train)
            self.model = ensemble
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)
        proba_std = np.std(y_proba)
        
        if proba_std < 0.05:
            print(f"⚠️ WARNING: Model outputs have low variance ({proba_std:.3f})")
            self.is_trained = False
            return {}
        
        self.training_accuracy = accuracy
        self.training_samples = len(df)
        self.feature_names = existing
        self.is_trained = True
        
        print(f"✓ Accuracy: {accuracy:.1%}")
        print(f"✓ Brier Score: {brier:.4f}")
        
        return {'accuracy': accuracy, 'brier_score': brier, 'samples': len(df)}
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, str]:
        """Predict Under 1.5 probability."""
        if not self.is_trained:
            base_prob = features.get('adjusted_under_1_5', DEFAULT_UNDER_1_5_RATE)
            return base_prob, 'STATISTICAL'
        
        # Create feature vector
        X = pd.DataFrame([{k: features.get(k, 0) for k in self.feature_names}])
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)[0, 1]
        
        # Blend with statistical estimate
        stat_estimate = features.get('adjusted_under_1_5', DEFAULT_UNDER_1_5_RATE)
        
        if self.training_samples >= 100:
            model_weight = 0.7
        elif self.training_samples >= 50:
            model_weight = 0.5
        else:
            model_weight = 0.3
        
        blended_proba = proba * model_weight + stat_estimate * (1 - model_weight)
        
        # Confidence
        if self.training_samples >= 100 and self.training_accuracy >= 0.65:
            if blended_proba >= 0.7:
                confidence = 'HIGH'
            elif blended_proba >= 0.5:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
        else:
            confidence = 'LIMITED_DATA'
        
        return blended_proba, confidence


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def make_predictions(upcoming_matches: List[Dict], team_stats: Dict,
                     model: CalibratedUnder15Model) -> pd.DataFrame:
    """Make predictions for all upcoming matches."""
    print("\n" + "="*70)
    print("STEP 4: MAKING PREDICTIONS")
    print("="*70)
    
    predictions = []
    
    for match_info in upcoming_matches:
        event = match_info['event']
        features = create_prediction_features(match_info, team_stats)
        proba, confidence = model.predict(features)
        
        # Kickoff time
        start_ts = event.get('startTimestamp', 0)
        kickoff = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M') if start_ts else 'Unknown'
        
        pred = {
            'match_id': event.get('id'),
            'tournament': match_info.get('league_name'),
            'home_team': safe_get(event, 'homeTeam', 'name', default='Unknown'),
            'away_team': safe_get(event, 'awayTeam', 'name', default='Unknown'),
            'kickoff_time': kickoff,
            'under_1_5_probability': round(proba, 4),
            'confidence': confidence,
            'expected_total_goals': round(features.get('expected_total_goals', 2.6), 2),
            'home_goals_avg': round(features.get('home_goals_scored_avg', 0), 2),
            'away_goals_avg': round(features.get('away_goals_scored_avg', 0), 2),
            'h2h_under_pct': round(features.get('h2h_under_1_5_pct', 0), 2),
            'home_historical_matches': features.get('home_historical_matches', 0),
            'away_historical_matches': features.get('away_historical_matches', 0),
        }
        
        # Recommendation
        if proba >= 0.6 and confidence in ['HIGH', 'MEDIUM']:
            pred['recommendation'] = 'UNDER 1.5 Goals'
        elif proba >= 0.5 and confidence == 'HIGH':
            pred['recommendation'] = 'LEAN UNDER 1.5'
        elif proba <= 0.35:
            pred['recommendation'] = 'OVER 1.5 Goals'
        else:
            pred['recommendation'] = 'NO STRONG SIGNAL'
        
        predictions.append(pred)
    
    df_pred = pd.DataFrame(predictions)
    
    if not df_pred.empty:
        df_pred = df_pred.sort_values('under_1_5_probability', ascending=False)
        
        # Save predictions
        predictions_file = f"{OUTPUT_DIR}/{FILENAME_PREFIX}.csv"
        df_pred.to_csv(predictions_file, index=False)
        print(f"\n✓ Saved {len(df_pred)} predictions to {predictions_file}")
        
        # Create results template
        df_results = df_pred[['match_id', 'tournament', 'home_team', 'away_team', 
                               'kickoff_time', 'under_1_5_probability', 'confidence', 
                               'recommendation']].copy()
        df_results['actual_home_score'] = ''
        df_results['actual_away_score'] = ''
        df_results['actual_total_goals'] = ''
        df_results['was_under_1_5'] = ''
        df_results['prediction_correct'] = ''
        df_results['prediction_timestamp'] = RUN_TIMESTAMP
        
        results_file = f"{OUTPUT_DIR}/{FILENAME_PREFIX}.result.csv"
        df_results.to_csv(results_file, index=False)
        print(f"✓ Created results template: {results_file}")
    
    # Display predictions
    print("\n" + "-"*70)
    print("PREDICTIONS - Matches sorted by Under 1.5 probability:")
    print("-"*70)
    
    for _, row in df_pred.iterrows():
        prob = row['under_1_5_probability']
        conf = row['confidence']
        rec = row['recommendation']
        exp_goals = row['expected_total_goals']
        
        print(f"\n  {row['home_team']} vs {row['away_team']}")
        print(f"    Tournament: {row['tournament']} | Kickoff: {row['kickoff_time']}")
        print(f"    Expected Goals: {exp_goals} | Under 1.5 Prob: {prob:.1%}")
        print(f"    Confidence: {conf} | >>> {rec}")
    
    return df_pred


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution."""
    print("\n" + "="*70)
    print("  FOOTBALL PREMATCH UNDER 1.5 GOALS PREDICTOR")
    print("  Dynamic Low-Scoring League Discovery + Ensemble ML")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Initialize API for league discovery
    discovery_api = SofascoreAPI()
    
    try:
        # STEP 1: Dynamically discover low-scoring leagues
        print("\n" + "="*70)
        print("STEP 1: DISCOVERING LOW-SCORING LEAGUES")
        print("="*70)
        
        low_scoring_leagues = await discover_low_scoring_leagues(
            discovery_api, 
            threshold=LOW_SCORING_THRESHOLD
        )
        
        print(f"\n🎯 Targeting {len(low_scoring_leagues)} low-scoring leagues for predictions")
        
    finally:
        await discovery_api.close()
    
    # STEP 2+: Collect match data
    collector = PrematchDataCollector()
    try:
        upcoming, team_stats, df_historical = await collector.collect_all_data(low_scoring_leagues)
    finally:
        await collector.close()
    
    if not upcoming:
        print("\n❌ No upcoming matches found. Try again later.")
        return
    
    # Save team stats
    if team_stats:
        df_stats = pd.DataFrame.from_dict(team_stats, orient='index')
        df_stats.to_csv(f"{OUTPUT_DIR}/prematch_team_stats.csv", index=True)
        print(f"\n✓ Saved team stats")
    
    # Save historical data
    if not df_historical.empty:
        df_historical.to_csv(f"{OUTPUT_DIR}/prematch_historical.csv", index=False)
        print(f"✓ Saved {len(df_historical)} historical matches")
    
    # Train model
    model = CalibratedUnder15Model()
    if not df_historical.empty:
        model.train(df_historical)
    else:
        print("\n⚠️ No historical data. Using statistical predictions only.")
    
    # Make predictions
    predictions = make_predictions(upcoming, team_stats, model)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"\nGenerated files in '{OUTPUT_DIR}/' folder:")
    print(f"  • {FILENAME_PREFIX}.csv           - {len(predictions)} predictions")
    print(f"  • {FILENAME_PREFIX}.result.csv    - Results template")
    print(f"  • prematch_team_stats.csv         - Team statistics")
    print(f"  • prematch_historical.csv         - Historical matches")
    
    if not predictions.empty:
        high_conf = predictions[predictions['confidence'] == 'HIGH']
        under_recs = predictions[predictions['recommendation'].str.contains('UNDER', na=False)]
        print(f"\n📊 High confidence predictions: {len(high_conf)}")
        print(f"🎯 Under 1.5 recommendations: {len(under_recs)}")
        print(f"\n💡 TIP: Fill in {FILENAME_PREFIX}.result.csv after matches end to track accuracy!")


def run():
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()