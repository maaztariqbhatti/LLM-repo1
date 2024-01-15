import floodtags
from datetime import datetime

start_date  = datetime(2024, 1, 2)
end_date = datetime(2024, 1, 4)

watershed_gdf = floodtags.analyze_watersheds(start_date=start_date,
                                    end_date=end_date)