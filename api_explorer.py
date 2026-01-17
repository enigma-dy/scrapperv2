import asyncio
import pandas as pd
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match

CSV_FILE = "live_matches_detailed_stats.csv"
CONCURRENCY_LIMIT = 10 

def parse_statistics(stats_response):
    """
    Parses the JSON response from .stats().
    """
    flat_data = {}
    
    # Handle list vs dict response structure
    stats_list = []
    if isinstance(stats_response, dict):
        stats_list = stats_response.get('statistics', [])
    elif isinstance(stats_response, list):
        stats_list = stats_response
        
    if not stats_list:
        return flat_data

    # Priority: "ALL" period -> "1ST" period -> "2ND" period
    period_stats = next((item for item in stats_list if item.get('period') == 'ALL'), None)
    if not period_stats:
        period_stats = next((item for item in stats_list if item.get('period') == '1ST'), None)
    if not period_stats:
        period_stats = next((item for item in stats_list if item.get('period') == '2ND'), None)
        
    if not period_stats:
        return flat_data

    # Dig into groups -> statisticsItems
    for group in period_stats.get('groups', []):
        for item in group.get('statisticsItems', []):
            # Normalize names: "Ball possession" -> "ball_possession"
            # "Total shots" -> "total_shots"
            stat_name = item.get('name', '').lower().strip().replace(' ', '_').replace('-', '_')
            
            flat_data[f"home_{stat_name}"] = item.get('home')
            flat_data[f"away_{stat_name}"] = item.get('away')

    return flat_data

async def process_single_match(api, match_summary, semaphore):
    async with semaphore:
        match_id = match_summary.get("id")
        
        row = {
            "match_id": match_id,
            "tournament": match_summary.get("tournament", {}).get("name"),
            "home_team": match_summary.get("homeTeam", {}).get("name"),
            "away_team": match_summary.get("awayTeam", {}).get("name"),
            "score": f"{match_summary.get('homeScore', {}).get('current', 0)}-{match_summary.get('awayScore', {}).get('current', 0)}",
            "minute": match_summary.get("status", {}).get("description"),
        }

        try:
            # Correct usage based on documentation:
            # Create a NEW Match instance with the ID, then call .stats()
            specific_match = Match(api, match_id=match_id)
            stats_json = await specific_match.stats()
            
            parsed_stats = parse_statistics(stats_json)
            row.update(parsed_stats)
            
        except Exception:
            pass

        return row

async def get_live_stats():
    api = SofascoreAPI()
    
    print("Fetching live match list...")
    # Use a generic match instance to get the list
    finder = Match(api) 
    live_response = await finder.live_games()
    
    football_matches = live_response.get("events", [])
    print(f"Found {len(football_matches)} live matches. Fetching details...")

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [process_single_match(api, m, sem) for m in football_matches]
    results = await asyncio.gather(*tasks)

    await api.close()

    if results:
        df = pd.DataFrame(results)

        # 1. Define the columns we WANT to see if they exist
        priority_cols = [
            'match_id', 'tournament', 'home_team', 'away_team', 'score', 'minute',
            'home_ball_possession', 'away_ball_possession',
            'home_total_shots', 'away_total_shots',
            'home_shots_on_target', 'away_shots_on_target',
            'home_big_chances', 'away_big_chances',
            'home_corner_kicks', 'away_corner_kicks'
        ]
        
        # 2. Filter priority_cols to only those that ACTUALLY exist in the data
        final_cols = [col for col in priority_cols if col in df.columns]
        
        # 3. Add any other columns found (e.g., fouls, yellow cards) at the end
        remaining_cols = [c for c in df.columns if c not in final_cols]
        final_cols.extend(remaining_cols)
        
        # Reorder DataFrame
        df = df[final_cols]
        df.to_csv(CSV_FILE, index=False)
        
        print(f"Successfully saved {len(df)} matches to {CSV_FILE}")
        
        # 4. Safe Preview
        # We only try to print columns that we know exist
        preview_targets = ['home_team', 'score', 'home_ball_possession', 'home_total_shots']
        safe_preview_cols = [c for c in preview_targets if c in df.columns]
        
        print("\nPreview:")
        print(df[safe_preview_cols].head(10).to_string())
    else:
        print("No matches found.")

if __name__ == "__main__":
    asyncio.run(get_live_stats())