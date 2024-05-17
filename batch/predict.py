import pandas as pd
import os 
from datetime import datetime

pd.DataFrame([1,2,3]).to_csv(os.path.join('..', 'app', f'{datetime.now()}_prediction.csv'))