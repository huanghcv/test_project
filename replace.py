import pandas as pd

data = pd.read_csv('submissionV4.csv')
data.id[:30490] = data.id[:30490].str.replace('evaluation$', 'validation')
data.to_csv('submissionV6.csv', index=False)
