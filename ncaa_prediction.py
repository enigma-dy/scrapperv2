"""
NCAA Basketball Over/Under Prediction System
=============================================

Predicts whether NCAA basketball games will go Over or Under the bookmaker's
total points line using team statistics, H2H data, and ensemble ML models.

Usage: python ncaa_prediction.py

Output CSVs (with timestamp):
- data/ncaa_predictions_YYYY-MM-DD_HH-MM.csv    - Predictions with picks
- data/ncaa_predictions_YYYY-MM-DD_HH-MM.result.csv - Results template
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
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️ ML libraries not found. Install with: pip install xgboost lightgbm scikit-learn")

# Sofascore imports
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.basketball import Basketball
from sofascore_wrapper.match import Match
from sofascore_wrapper.team import Team
from sofascore_wrapper.league import League
from sofascore_wrapper.search import Search

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate timestamp for unique filenames
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
FILENAME_PREFIX = f"ncaa_predictions_{RUN_TIMESTAMP}"

# Time between API requests to avoid rate limiting
REQUEST_DELAY = 0.5

# Minimum requirements
MIN_TRAINING_SAMPLES = 30
MIN_TEAM_MATCHES = 4

# Statistical fallbacks
DEFAULT_AVG_TOTAL = 145.0  # Average NCAA game total
DEFAULT_AVG_SCORED = 72.5
DEFAULT_LINE = 145.0

# NCAA Basketball Leagues (discovered from SofaScore)
NCAA_LEAGUES = {
    132: "NCAA Men's Basketball",
    133: "NCAA Women's Basketball",
    # Will discover more via search
}


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


def parse_odds_line(odds_data: Dict) -> Optional[float]:
    """Extract Over/Under line from odds data."""
    try:
        if not odds_data:
            return None
        
        # Look for totals market in featured odds
        featured = odds_data.get('featured', {})
        
        # Check for totals/over-under markets
        for market_key, market in featured.items():
            if isinstance(market, dict):
                market_name = market.get('marketName', '').lower()
                if 'total' in market_name or 'over' in market_name:
                    choices = market.get('choices', [])
                    for choice in choices:
                        name = choice.get('name', '').lower()
                        if 'over' in name or 'under' in name:
                            # Extract line from name like "Over 145.5"
                            parts = name.split()
                            for part in parts:
                                try:
                                    return float(part)
                                except ValueError:
                                    continue
        return None
    except Exception:
        return None


# ============================================================================
# DATA COLLECTION
# ============================================================================

class NCAADataCollector:
    """Collects NCAA basketball data for prediction."""
    
    def __init__(self):
        self.api = SofascoreAPI()
        self.basketball = Basketball(self.api)
        self.team_stats_cache = {}
        self.historical_matches = []
    
    async def close(self):
        await self.api.close()
    
    async def search_ncaa_leagues(self) -> List[Dict]:
        """Search for NCAA basketball leagues/tournaments."""
        leagues = []
        try:
            search = Search(self.api, "NCAA", 0)
            results = await search.search_leagues(sport="basketball")
            
            if results:
                print("\n📋 Found NCAA Basketball Leagues:")
                for item in results:
                    if isinstance(item, dict):
                        entity = item.get('entity', item)
                        name = entity.get('name', 'Unknown')
                        league_id = entity.get('id', 0)
                        if 'ncaa' in name.lower() or 'college' in name.lower():
                            print(f"  • {name} (ID: {league_id})")
                            leagues.append(entity)
        except Exception as e:
            print(f"⚠️ Search error: {e}")
        
        return leagues
    
    async def get_basketball_games(self, date: str = None) -> List[Dict]:
        """Get basketball games for a specific date."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        all_games = []
        try:
            games = await self.basketball.games_by_date(sport="basketball", date=date)
            
            if games and 'events' in games:
                for event in games['events']:
                    tournament = event.get('tournament', {})
                    unique_tournament = tournament.get('uniqueTournament', {})
                    category = unique_tournament.get('category', {})
                    
                    # Filter for NCAA/US College basketball
                    category_name = category.get('name', '').lower()
                    tournament_name = unique_tournament.get('name', '').lower()
                    
                    is_ncaa = any(term in tournament_name for term in ['ncaa', 'college', 'ncaab'])
                    is_usa = category_name in ['usa', 'united states', 'us']
                    
                    if is_ncaa or (is_usa and 'college' in tournament_name):
                        all_games.append(event)
            
            await asyncio.sleep(REQUEST_DELAY)
            
        except Exception as e:
            print(f"⚠️ Error getting games: {e}")
        
        return all_games
    
    async def get_upcoming_matches(self) -> List[Dict]:
        """Get upcoming NCAA basketball matches."""
        upcoming = []
        
        # Try today and tomorrow
        for day_offset in range(2):
            date = (datetime.now() + timedelta(days=day_offset)).strftime('%Y-%m-%d')
            print(f"\n  Checking games for {date}...")
            
            games = await self.get_basketball_games(date)
            
            for event in games:
                status_code = safe_get(event, 'status', 'code', default=100)
                # Status 0 = not started
                if status_code == 0:
                    upcoming.append({
                        'event': event,
                        'date': date,
                        'tournament': safe_get(event, 'tournament', 'uniqueTournament', 'name'),
                    })
        
        return upcoming
    
    async def get_team_last_matches(self, team_id: int, team_name: str,
                                     num_matches: int = 8) -> List[Dict]:
        """Get team's last N matches with scores, wins, and over/under data."""
        matches = []
        
        try:
            team = Team(self.api, team_id=team_id)
            fixtures = await team.last_fixtures()
            
            events = fixtures if isinstance(fixtures, list) else fixtures.get('events', [])
            
            for event in events[:num_matches * 2]:  # Get more to filter finished
                status = safe_get(event, 'status', 'type', default='')
                if status != 'finished':
                    continue
                
                home_score = safe_get(event, 'homeScore', 'current', default=0) or 0
                away_score = safe_get(event, 'awayScore', 'current', default=0) or 0
                total_points = home_score + away_score
                
                is_home = safe_get(event, 'homeTeam', 'id') == team_id
                team_score = home_score if is_home else away_score
                opp_score = away_score if is_home else home_score
                
                # Determine win/loss
                if team_score > opp_score:
                    result = 'W'
                elif team_score < opp_score:
                    result = 'L'
                else:
                    result = 'D'
                
                match_data = {
                    'match_id': event.get('id'),
                    'team_id': team_id,
                    'team_name': team_name,
                    'is_home': is_home,
                    'points_scored': team_score,
                    'points_conceded': opp_score,
                    'total_points': total_points,
                    'opponent_name': safe_get(event, 'awayTeam', 'name') if is_home 
                                    else safe_get(event, 'homeTeam', 'name'),
                    'result': result,
                    'is_win': 1 if result == 'W' else 0,
                    'is_over_145': 1 if total_points > 145 else 0,
                    'match_index': len(matches),  # 0 = most recent
                }
                matches.append(match_data)
                
                if len(matches) >= num_matches:
                    break
                    
        except Exception as e:
            print(f"    ⚠️ Error fetching history for {team_name}: {e}")
        
        return matches
    
    async def get_team_form(self, team_id: int) -> Dict:
        """Fetch team form data (pregame form, rating)."""
        form_data = {
            'form_rating': 0.0,
            'form_string': '',
            'recent_form_score': 0.0,
        }
        
        try:
            team = Team(self.api, team_id=team_id)
            team_info = await team.get_team()
            
            if team_info:
                # Extract pregame form if available
                pregame = team_info.get('pregameForm', {})
                form_data['form_rating'] = pregame.get('avgRating', 0.0)
                form_data['form_string'] = pregame.get('form', '')
                
                # Calculate form score from W/D/L string
                form_str = form_data['form_string']
                if form_str:
                    score = 0
                    for char in form_str[-5:]:  # Last 5 games
                        if char == 'W':
                            score += 1
                        elif char == 'L':
                            score -= 1
                    form_data['recent_form_score'] = score / 5.0  # Normalize to -1 to 1
                    
        except Exception as e:
            pass  # Use defaults
        
        return form_data
    
    def calculate_team_stats(self, matches: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive team statistics from recent matches."""
        default_stats = {
            'avg_scored': DEFAULT_AVG_SCORED,
            'avg_conceded': DEFAULT_AVG_SCORED,
            'avg_total': DEFAULT_AVG_TOTAL,
            'matches_count': 0,
            'max_scored': DEFAULT_AVG_SCORED + 15,
            'min_scored': DEFAULT_AVG_SCORED - 15,
            'win_rate': 0.5,
            'momentum_score': 0.0,
            'scoring_trend': 0.0,
            'conceding_trend': 0.0,
            'over_rate': 0.5,
            'home_scoring_avg': DEFAULT_AVG_SCORED,
            'away_scoring_avg': DEFAULT_AVG_SCORED,
            'std_dev_scored': 10.0,
            'std_dev_total': 15.0,
            'last_3_avg_total': DEFAULT_AVG_TOTAL,
            'streak': 0,
        }
        
        if not matches or len(matches) < MIN_TEAM_MATCHES:
            default_stats['matches_count'] = len(matches) if matches else 0
            return default_stats
        
        df = pd.DataFrame(matches)
        
        # Basic averages
        avg_scored = df['points_scored'].mean()
        avg_conceded = df['points_conceded'].mean()
        avg_total = df['total_points'].mean()
        
        # Win rate
        win_rate = df['is_win'].mean() if 'is_win' in df.columns else 0.5
        
        # Momentum score: weighted recent form (-1 to +1)
        # More recent games weighted higher
        momentum = 0.0
        if 'is_win' in df.columns:
            weights = [0.3, 0.25, 0.2, 0.15, 0.07, 0.02, 0.01, 0.0][:len(df)]
            weighted_wins = sum(w * (1 if win else -1) 
                               for w, win in zip(weights, df['is_win'].values))
            momentum = max(-1, min(1, weighted_wins))
        
        # Scoring trend (slope) - positive means improving
        scoring_trend = 0.0
        conceding_trend = 0.0
        if len(df) >= 3:
            # Reverse since index 0 is most recent
            x = np.arange(len(df))[::-1]  # [N-1, N-2, ..., 0]
            try:
                scoring_trend = np.polyfit(x, df['points_scored'].values, 1)[0]
                conceding_trend = np.polyfit(x, df['points_conceded'].values, 1)[0]
            except:
                pass
        
        # Over rate (games > 145 total)
        over_rate = df['is_over_145'].mean() if 'is_over_145' in df.columns else 0.5
        
        # Home/Away scoring averages
        home_matches = df[df['is_home'] == True] if 'is_home' in df.columns else pd.DataFrame()
        away_matches = df[df['is_home'] == False] if 'is_home' in df.columns else pd.DataFrame()
        home_scoring_avg = home_matches['points_scored'].mean() if len(home_matches) > 0 else avg_scored
        away_scoring_avg = away_matches['points_scored'].mean() if len(away_matches) > 0 else avg_scored
        
        # Consistency measures (standard deviation)
        std_dev_scored = df['points_scored'].std() if len(df) > 1 else 10.0
        std_dev_total = df['total_points'].std() if len(df) > 1 else 15.0
        
        # Last 3 games average
        last_3_avg = df['total_points'].head(3).mean() if len(df) >= 3 else avg_total
        
        # Calculate streak (consecutive wins or losses)
        streak = 0
        if 'result' in df.columns:
            first_result = df['result'].iloc[0] if len(df) > 0 else None
            for r in df['result'].values:
                if r == first_result and first_result in ['W', 'L']:
                    streak += 1 if first_result == 'W' else -1
                else:
                    break
        
        return {
            'avg_scored': avg_scored,
            'avg_conceded': avg_conceded,
            'avg_total': avg_total,
            'matches_count': len(df),
            'max_scored': df['points_scored'].max(),
            'min_scored': df['points_scored'].min(),
            'win_rate': win_rate,
            'momentum_score': momentum,
            'scoring_trend': scoring_trend,
            'conceding_trend': conceding_trend,
            'over_rate': over_rate,
            'home_scoring_avg': home_scoring_avg if not pd.isna(home_scoring_avg) else avg_scored,
            'away_scoring_avg': away_scoring_avg if not pd.isna(away_scoring_avg) else avg_scored,
            'std_dev_scored': std_dev_scored if not pd.isna(std_dev_scored) else 10.0,
            'std_dev_total': std_dev_total if not pd.isna(std_dev_total) else 15.0,
            'last_3_avg_total': last_3_avg if not pd.isna(last_3_avg) else avg_total,
            'streak': streak,
        }
    
    async def get_h2h_data(self, match_id: int) -> Dict:
        """Get head-to-head data for a match."""
        h2h_stats = {
            'h2h_matches': 0,
            'h2h_avg_total': DEFAULT_AVG_TOTAL,
            'h2h_over_tendency': 0.5,
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
        }
        
        try:
            match = Match(self.api, match_id=match_id)
            h2h = await match.h2h()
            
            if not h2h:
                return h2h_stats
            
            team_duel = h2h.get('teamDuel', {})
            h2h_stats['h2h_home_wins'] = team_duel.get('homeWins', 0)
            h2h_stats['h2h_away_wins'] = team_duel.get('awayWins', 0)
            
            # Calculate from H2H events if available
            events = h2h.get('events', [])
            if events:
                totals = []
                for event in events[:10]:
                    home_score = safe_get(event, 'homeScore', 'current', default=0) or 0
                    away_score = safe_get(event, 'awayScore', 'current', default=0) or 0
                    totals.append(home_score + away_score)
                
                if totals:
                    h2h_stats['h2h_matches'] = len(totals)
                    h2h_stats['h2h_avg_total'] = sum(totals) / len(totals)
                    # Calculate over tendency (using typical line of 145)
                    h2h_stats['h2h_over_tendency'] = sum(1 for t in totals if t > 145) / len(totals)
                    
        except Exception as e:
            pass  # Use defaults
        
        return h2h_stats
    
    async def get_bookmaker_line(self, match_id: int) -> Optional[float]:
        """Get bookmaker Over/Under line for a match."""
        try:
            match = Match(self.api, match_id=match_id)
            odds = await match.featured_odds()
            
            if odds:
                line = parse_odds_line(odds)
                if line:
                    return line
                    
        except Exception as e:
            pass
        
        return None
    
    async def collect_all_data(self) -> Tuple[List[Dict], Dict, pd.DataFrame]:
        """Collect all data for prediction."""
        print("\n" + "="*70)
        print("STEP 1: SEARCHING FOR NCAA BASKETBALL MATCHES")
        print("="*70)
        
        # Search for leagues first
        await self.search_ncaa_leagues()
        
        # Get upcoming matches
        upcoming = await self.get_upcoming_matches()
        
        print(f"\n✓ Found {len(upcoming)} upcoming NCAA matches")
        
        if not upcoming:
            # Try broader basketball search
            print("\n  Trying broader basketball search...")
            for day_offset in range(2):
                date = (datetime.now() + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                games = await self.get_basketball_games(date)
                for event in games:
                    status_code = safe_get(event, 'status', 'code', default=100)
                    if status_code == 0:
                        upcoming.append({
                            'event': event,
                            'date': date,
                            'tournament': safe_get(event, 'tournament', 'uniqueTournament', 'name'),
                        })
            print(f"  Found {len(upcoming)} total basketball matches")
        
        if not upcoming:
            return [], {}, pd.DataFrame()
        
        # Collect team data
        print("\n" + "="*70)
        print("STEP 2: COLLECTING TEAM STATISTICS")
        print("="*70)
        
        team_stats = {}
        processed_teams = set()
        
        for idx, match_info in enumerate(upcoming):
            event = match_info['event']
            home_id = safe_get(event, 'homeTeam', 'id')
            home_name = safe_get(event, 'homeTeam', 'name', default='Unknown')
            away_id = safe_get(event, 'awayTeam', 'id')
            away_name = safe_get(event, 'awayTeam', 'name', default='Unknown')
            
            # Home team
            if home_id and home_id not in processed_teams:
                print(f"  [{len(processed_teams)+1}] {home_name}...")
                matches = await self.get_team_last_matches(home_id, home_name)
                self.historical_matches.extend(matches)
                team_stats[home_id] = self.calculate_team_stats(matches)
                team_stats[home_id]['team_name'] = home_name
                processed_teams.add(home_id)
                await asyncio.sleep(REQUEST_DELAY)
            
            # Away team
            if away_id and away_id not in processed_teams:
                print(f"  [{len(processed_teams)+1}] {away_name}...")
                matches = await self.get_team_last_matches(away_id, away_name)
                self.historical_matches.extend(matches)
                team_stats[away_id] = self.calculate_team_stats(matches)
                team_stats[away_id]['team_name'] = away_name
                processed_teams.add(away_id)
                await asyncio.sleep(REQUEST_DELAY)
        
        print(f"\n✓ Collected stats for {len(team_stats)} teams")
        
        # Collect H2H and odds
        print("\n" + "="*70)
        print("STEP 3: COLLECTING H2H DATA & BOOKMAKER LINES")
        print("="*70)
        
        for idx, match_info in enumerate(upcoming):
            event = match_info['event']
            match_id = event.get('id')
            home_name = safe_get(event, 'homeTeam', 'name', default='Unknown')
            away_name = safe_get(event, 'awayTeam', 'name', default='Unknown')
            
            print(f"  [{idx+1}/{len(upcoming)}] {home_name} vs {away_name}...")
            
            # Get H2H
            h2h = await self.get_h2h_data(match_id)
            match_info['h2h'] = h2h
            await asyncio.sleep(REQUEST_DELAY)
            
            # Get bookmaker line
            line = await self.get_bookmaker_line(match_id)
            match_info['bookie_line'] = line
            await asyncio.sleep(REQUEST_DELAY)
        
        # Create historical DataFrame
        df_historical = pd.DataFrame(self.historical_matches)
        if not df_historical.empty:
            df_historical = df_historical.drop_duplicates(subset=['match_id'])
        
        return upcoming, team_stats, df_historical


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_prediction_features(match_info: Dict, team_stats: Dict) -> Dict[str, float]:
    """Create comprehensive features for Over/Under prediction."""
    event = match_info['event']
    home_id = safe_get(event, 'homeTeam', 'id')
    away_id = safe_get(event, 'awayTeam', 'id')
    h2h = match_info.get('h2h', {})
    bookie_line = match_info.get('bookie_line')
    
    # Default stats with all new metrics
    default_stats = {
        'avg_scored': DEFAULT_AVG_SCORED,
        'avg_conceded': DEFAULT_AVG_SCORED,
        'avg_total': DEFAULT_AVG_TOTAL,
        'matches_count': 0,
        'win_rate': 0.5,
        'momentum_score': 0.0,
        'scoring_trend': 0.0,
        'conceding_trend': 0.0,
        'over_rate': 0.5,
        'home_scoring_avg': DEFAULT_AVG_SCORED,
        'away_scoring_avg': DEFAULT_AVG_SCORED,
        'std_dev_scored': 10.0,
        'std_dev_total': 15.0,
        'last_3_avg_total': DEFAULT_AVG_TOTAL,
        'streak': 0,
    }
    
    home_stats = team_stats.get(home_id, default_stats)
    away_stats = team_stats.get(away_id, default_stats)
    
    # Expected total calculation
    # Method 1: Average of each team's typical total
    expected_total_1 = (home_stats.get('avg_total', DEFAULT_AVG_TOTAL) + 
                        away_stats.get('avg_total', DEFAULT_AVG_TOTAL)) / 2
    
    # Method 2: Home offense vs Away defense + Away offense vs Home defense
    expected_total_2 = (home_stats.get('avg_scored', DEFAULT_AVG_SCORED) + 
                        away_stats.get('avg_conceded', DEFAULT_AVG_SCORED) +
                        away_stats.get('avg_scored', DEFAULT_AVG_SCORED) + 
                        home_stats.get('avg_conceded', DEFAULT_AVG_SCORED)) / 2
    
    # Method 3: Last 3 games average (recent form)
    expected_total_3 = (home_stats.get('last_3_avg_total', DEFAULT_AVG_TOTAL) + 
                        away_stats.get('last_3_avg_total', DEFAULT_AVG_TOTAL)) / 2
    
    # Blend methods with weights
    expected_total = (expected_total_1 * 0.3 + expected_total_2 * 0.4 + expected_total_3 * 0.3)
    
    # H2H influence (if available)
    h2h_weight = min(h2h.get('h2h_matches', 0) / 10, 0.3)
    if h2h.get('h2h_matches', 0) >= 3:
        expected_total = expected_total * (1 - h2h_weight) + h2h.get('h2h_avg_total', expected_total) * h2h_weight
    
    # Use bookie line if available, otherwise estimate
    line = bookie_line if bookie_line else expected_total
    
    # Pace calculation (average total points per game)
    home_pace = home_stats.get('avg_total', DEFAULT_AVG_TOTAL)
    away_pace = away_stats.get('avg_total', DEFAULT_AVG_TOTAL)
    combined_pace = (home_pace + away_pace) / 2
    
    # Combined over rates
    combined_over_rate = (home_stats.get('over_rate', 0.5) + away_stats.get('over_rate', 0.5)) / 2
    
    features = {
        # Basic Team stats
        'home_avg_scored': home_stats.get('avg_scored', DEFAULT_AVG_SCORED),
        'home_avg_conceded': home_stats.get('avg_conceded', DEFAULT_AVG_SCORED),
        'home_avg_total': home_stats.get('avg_total', DEFAULT_AVG_TOTAL),
        'away_avg_scored': away_stats.get('avg_scored', DEFAULT_AVG_SCORED),
        'away_avg_conceded': away_stats.get('avg_conceded', DEFAULT_AVG_SCORED),
        'away_avg_total': away_stats.get('avg_total', DEFAULT_AVG_TOTAL),
        
        # Combined metrics
        'combined_avg_total': (home_stats.get('avg_total', DEFAULT_AVG_TOTAL) + 
                               away_stats.get('avg_total', DEFAULT_AVG_TOTAL)) / 2,
        'combined_avg_scored': (home_stats.get('avg_scored', DEFAULT_AVG_SCORED) + 
                                away_stats.get('avg_scored', DEFAULT_AVG_SCORED)),
        
        # NEW: Recent Performance / Win Rate
        'home_win_rate': home_stats.get('win_rate', 0.5),
        'away_win_rate': away_stats.get('win_rate', 0.5),
        
        # NEW: Momentum (weighted recent form)
        'home_momentum': home_stats.get('momentum_score', 0.0),
        'away_momentum': away_stats.get('momentum_score', 0.0),
        'momentum_diff': home_stats.get('momentum_score', 0.0) - away_stats.get('momentum_score', 0.0),
        
        # NEW: Scoring/Conceding Trends
        'home_scoring_trend': home_stats.get('scoring_trend', 0.0),
        'away_scoring_trend': away_stats.get('scoring_trend', 0.0),
        'home_conceding_trend': home_stats.get('conceding_trend', 0.0),
        'away_conceding_trend': away_stats.get('conceding_trend', 0.0),
        
        # NEW: Pace & Scoring Style
        'home_pace': home_pace,
        'away_pace': away_pace,
        'combined_pace': combined_pace,
        'home_over_rate': home_stats.get('over_rate', 0.5),
        'away_over_rate': away_stats.get('over_rate', 0.5),
        'combined_over_rate': combined_over_rate,
        
        # NEW: Consistency (lower std = more predictable)
        'home_std_dev_scored': home_stats.get('std_dev_scored', 10.0),
        'away_std_dev_scored': away_stats.get('std_dev_scored', 10.0),
        'home_std_dev_total': home_stats.get('std_dev_total', 15.0),
        'away_std_dev_total': away_stats.get('std_dev_total', 15.0),
        
        # NEW: Recent form (last 3 games)
        'home_last_3_avg': home_stats.get('last_3_avg_total', DEFAULT_AVG_TOTAL),
        'away_last_3_avg': away_stats.get('last_3_avg_total', DEFAULT_AVG_TOTAL),
        
        # NEW: Streaks
        'home_streak': home_stats.get('streak', 0),
        'away_streak': away_stats.get('streak', 0),
        
        # H2H
        'h2h_avg_total': h2h.get('h2h_avg_total', DEFAULT_AVG_TOTAL),
        'h2h_over_tendency': h2h.get('h2h_over_tendency', 0.5),
        'h2h_matches': h2h.get('h2h_matches', 0),
        
        # Expected and line
        'expected_total': expected_total,
        'bookie_line': line,
        'has_real_line': 1 if bookie_line else 0,
        
        # Data quality
        'home_matches_count': home_stats.get('matches_count', 0),
        'away_matches_count': away_stats.get('matches_count', 0),
        'total_data_quality': min(home_stats.get('matches_count', 0), 
                                  away_stats.get('matches_count', 0)),
    }
    
    return features


# ============================================================================
# ML MODEL
# ============================================================================

class NCAAOverUnderModel:
    """Hybrid Ensemble model for Over/Under predictions with stacking."""
    
    def __init__(self):
        self.model = None
        self.stacking_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.mean_total = DEFAULT_AVG_TOTAL
        self.training_samples = 0
        self.feature_importance = {}
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the hybrid ensemble model on historical data."""
        print("\n--- Training Hybrid Ensemble Over/Under Model ---")
        
        if len(df) < MIN_TRAINING_SAMPLES:
            print(f"⚠️ Insufficient data: {len(df)} samples (need {MIN_TRAINING_SAMPLES}+)")
            print("   Model will use statistical predictions only.")
            if not df.empty:
                self.mean_total = df['total_points'].mean()
            return {}
        
        # Use ALL available features from historical matches
        all_features = [
            'points_scored', 'points_conceded', 'is_home', 
            'is_win', 'is_over_145', 'match_index'
        ]
        existing = [c for c in all_features if c in df.columns]
        
        if not existing or 'total_points' not in df.columns:
            print("❌ No valid features found")
            return {}
        
        X = df[existing].fillna(0)
        y = df['total_points']
        
        self.mean_total = y.mean()
        self.training_samples = len(df)
        
        print(f"Training on {len(df)} matches with {len(existing)} features")
        print(f"Features: {existing}")
        print(f"Average total points: {self.mean_total:.1f}")
        
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        except Exception as e:
            print(f"⚠️ Train/test split failed: {e}")
            return {}
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Base models for ensemble
        xgb = XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42
        )
        
        lgb = LGBMRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bynode=0.8,
            verbose=-1, random_state=42
        )
        
        rf = RandomForestRegressor(
            n_estimators=150, max_depth=6, 
            min_samples_split=5, min_samples_leaf=2,
            random_state=42
        )
        
        # Primary Ensemble (VotingRegressor)
        self.model = VotingRegressor(
            estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf)]
        )
        
        try:
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
        except Exception as e:
            print(f"⚠️ Training failed: {e}")
            return {}
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"✓ MAE: {mae:.1f} points")
        print(f"✓ RMSE: {rmse:.1f} points")
        
        # Feature importance analysis (from Random Forest)
        try:
            rf_model = rf.fit(X_train_scaled, y_train)
            importances = rf_model.feature_importances_
            self.feature_importance = dict(zip(existing, importances))
            
            # Sort and display top features
            sorted_importance = sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
            print("\n📊 Feature Importance:")
            for feat, imp in sorted_importance[:5]:
                print(f"   • {feat}: {imp:.3f}")
        except Exception as e:
            print(f"⚠️ Could not calculate feature importance: {e}")
        
        self.feature_names = existing
        
        return {
            'mae': mae, 
            'rmse': rmse, 
            'samples': len(df),
            'features': len(existing)
        }
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, str, str]:
        """Predict total points and Over/Under pick with adjusted thresholds."""
        expected = features.get('expected_total', DEFAULT_AVG_TOTAL)
        bookie_line = features.get('bookie_line', DEFAULT_AVG_TOTAL)
        
        # Use expected total as prediction base (blended from multiple methods)
        predicted_total = expected
        
        # Factor in momentum and trends for adjustment
        home_momentum = features.get('home_momentum', 0)
        away_momentum = features.get('away_momentum', 0)
        home_scoring_trend = features.get('home_scoring_trend', 0)
        away_scoring_trend = features.get('away_scoring_trend', 0)
        
        # Momentum adjustment (hot teams score more)
        momentum_adjustment = (home_momentum + away_momentum) * 1.5
        
        # Trend adjustment (improving teams score more)
        trend_adjustment = (home_scoring_trend + away_scoring_trend) * 0.5
        
        # Apply adjustments
        predicted_total += momentum_adjustment + trend_adjustment
        
        # Factor in over rates
        combined_over_rate = features.get('combined_over_rate', 0.5)
        if combined_over_rate > 0.6:
            predicted_total += 1.5  # Teams that play high scoring games
        elif combined_over_rate < 0.4:
            predicted_total -= 1.5  # Teams that play low scoring games
        
        # Calculate edge
        edge = predicted_total - bookie_line
        
        # Determine pick with ADJUSTED thresholds (1.5 instead of 2)
        if edge > 1.5:
            pick = 'OVER'
        elif edge < -1.5:
            pick = 'UNDER'
        else:
            pick = 'LEAN OVER' if edge > 0 else 'LEAN UNDER'
        
        # Enhanced confidence calculation with data quality bonus
        data_quality = features.get('total_data_quality', 
                                    min(features.get('home_matches_count', 0), 
                                        features.get('away_matches_count', 0)))
        abs_edge = abs(edge)
        
        # H2H bonus (if we have H2H data, we're more confident)
        h2h_bonus = 0.5 if features.get('h2h_matches', 0) >= 3 else 0
        
        # Consistency bonus (low std dev = more predictable)
        home_std = features.get('home_std_dev_total', 15)
        away_std = features.get('away_std_dev_total', 15)
        consistency_bonus = 0.5 if (home_std < 12 and away_std < 12) else 0
        
        # Total quality score
        quality_score = data_quality + h2h_bonus + consistency_bonus
        
        # ADJUSTED thresholds for confidence
        if abs_edge >= 4 and quality_score >= 3:
            confidence = 'HIGH'
        elif abs_edge >= 2.5 and quality_score >= 2:
            confidence = 'MEDIUM'
        elif abs_edge >= 1.5:
            confidence = 'LOW'
        else:
            confidence = 'INSUFFICIENT_EDGE'
        
        return predicted_total, pick, confidence


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def make_predictions(upcoming_matches: List[Dict], team_stats: Dict,
                     model: NCAAOverUnderModel) -> pd.DataFrame:
    """Make predictions for all upcoming matches."""
    print("\n" + "="*70)
    print("STEP 4: MAKING PREDICTIONS")
    print("="*70)
    
    predictions = []
    
    for match_info in upcoming_matches:
        event = match_info['event']
        features = create_prediction_features(match_info, team_stats)
        predicted_total, pick, confidence = model.predict(features)
        
        # Kickoff time
        start_ts = event.get('startTimestamp', 0)
        kickoff = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M') if start_ts else 'Unknown'
        
        # Calculate edge
        bookie_line = features.get('bookie_line', predicted_total)
        edge = predicted_total - bookie_line
        
        pred = {
            'match_id': event.get('id'),
            'tournament': match_info.get('tournament', 'Unknown'),
            'home_team': safe_get(event, 'homeTeam', 'name', default='Unknown'),
            'away_team': safe_get(event, 'awayTeam', 'name', default='Unknown'),
            'kickoff_time': kickoff,
            'bookie_line': round(bookie_line, 1),
            'predicted_total': round(predicted_total, 1),
            'edge': round(edge, 1),
            'pick': pick,
            'confidence': confidence,
            'home_avg_pts': round(features.get('home_avg_scored', 0), 1),
            'away_avg_pts': round(features.get('away_avg_scored', 0), 1),
            'h2h_avg_total': round(features.get('h2h_avg_total', 0), 1),
            'h2h_matches': features.get('h2h_matches', 0),
            'has_real_line': features.get('has_real_line', 0),
        }
        
        predictions.append(pred)
    
    df_pred = pd.DataFrame(predictions)
    
    if not df_pred.empty:
        # Sort by absolute edge (highest confidence first)
        df_pred['abs_edge'] = df_pred['edge'].abs()
        df_pred = df_pred.sort_values('abs_edge', ascending=False)
        df_pred = df_pred.drop(columns=['abs_edge'])
        
        # Save predictions
        predictions_file = f"{OUTPUT_DIR}/{FILENAME_PREFIX}.csv"
        df_pred.to_csv(predictions_file, index=False)
        print(f"\n✓ Saved {len(df_pred)} predictions to {predictions_file}")
        
        # Create results template
        df_results = df_pred[['match_id', 'tournament', 'home_team', 'away_team',
                              'kickoff_time', 'bookie_line', 'predicted_total',
                              'pick', 'confidence']].copy()
        df_results['actual_home_score'] = ''
        df_results['actual_away_score'] = ''
        df_results['actual_total'] = ''
        df_results['over_under_result'] = ''
        df_results['prediction_correct'] = ''
        df_results['prediction_timestamp'] = RUN_TIMESTAMP
        
        results_file = f"{OUTPUT_DIR}/{FILENAME_PREFIX}.result.csv"
        df_results.to_csv(results_file, index=False)
        print(f"✓ Created results template: {results_file}")
    
    # Display predictions
    print("\n" + "-"*70)
    print("PREDICTIONS - Sorted by Edge Magnitude:")
    print("-"*70)
    
    for _, row in df_pred.iterrows():
        edge = row['edge']
        line = row['bookie_line']
        pred_total = row['predicted_total']
        pick = row['pick']
        conf = row['confidence']
        
        # Direction indicator
        direction = "📈" if edge > 0 else "📉"
        
        print(f"\n  {row['home_team']} vs {row['away_team']}")
        print(f"    Tournament: {row['tournament']} | Kickoff: {row['kickoff_time']}")
        print(f"    Bookie Line: {line} | Predicted: {pred_total} | Edge: {edge:+.1f}")
        print(f"    {direction} >>> {pick} ({conf}) <<<")
    
    return df_pred


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution."""
    print("\n" + "="*70)
    print("  NCAA BASKETBALL OVER/UNDER PREDICTOR")
    print("  Team Stats + H2H + Ensemble ML")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Collect data
    collector = NCAADataCollector()
    try:
        upcoming, team_stats, df_historical = await collector.collect_all_data()
    finally:
        await collector.close()
    
    if not upcoming:
        print("\n❌ No upcoming NCAA basketball matches found.")
        print("   This may be because:")
        print("   • No games scheduled today/tomorrow")
        print("   • NCAA season may be off")
        print("   Try running on a day with known NCAA games.")
        return
    
    # Save team stats
    if team_stats:
        df_stats = pd.DataFrame.from_dict(team_stats, orient='index')
        df_stats.to_csv(f"{OUTPUT_DIR}/ncaa_team_stats.csv", index=True)
        print(f"\n✓ Saved team stats")
    
    # Save historical data
    if not df_historical.empty:
        df_historical.to_csv(f"{OUTPUT_DIR}/ncaa_historical.csv", index=False)
        print(f"✓ Saved {len(df_historical)} historical matches")
    
    # Train model
    model = NCAAOverUnderModel()
    if HAS_ML and not df_historical.empty:
        model.train(df_historical)
    else:
        print("\n⚠️ Using statistical predictions only (no ML training data).")
    
    # Make predictions
    predictions = make_predictions(upcoming, team_stats, model)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"\nGenerated files in '{OUTPUT_DIR}/' folder:")
    print(f"  • {FILENAME_PREFIX}.csv         - {len(predictions)} predictions")
    print(f"  • {FILENAME_PREFIX}.result.csv  - Results template")
    print(f"  • ncaa_team_stats.csv           - Team statistics")
    print(f"  • ncaa_historical.csv           - Historical matches")
    
    if not predictions.empty:
        high_conf = predictions[predictions['confidence'] == 'HIGH']
        over_picks = predictions[predictions['pick'].str.contains('OVER', na=False)]
        under_picks = predictions[predictions['pick'].str.contains('UNDER', na=False)]
        
        print(f"\n📊 High confidence picks: {len(high_conf)}")
        print(f"📈 OVER picks: {len(over_picks)}")
        print(f"📉 UNDER picks: {len(under_picks)}")
        print(f"\n💡 TIP: Fill in {FILENAME_PREFIX}.result.csv after matches to track accuracy!")


def run():
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()