import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the data you provided
df = pd.read_csv('FoAI_A2_data_4.6k.csv')

# Quick look
print(df.shape)  # ~4600 rows
print(df.work_year.value_counts().sort_index())

# Use salary_in_usd for all analysis (already converted)
df = df[['work_year', 'experience_level', 'employment_type', 'job_title',
         'salary_in_usd', 'employee_residence', 'remote_ratio',
         'company_location', 'company_size']].copy()

# Clean experience_level order
df['experience_level'] = pd.Categorical(df['experience_level'],
                                        categories=['EN', 'MI', 'SE', 'EX'],
                                        ordered=True)


# Basic stats
stats = df.describe().round(2)
print("Basic Statistics (USD):")
print(stats)

# Add median for salary per year and level
print("\nMedian salary by year and experience level:")
print(df.groupby(['work_year', 'experience_level'])['salary_in_usd'].median().unstack().round(0))


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
fig = plt.figure(figsize=(20, 24))

# 1. Salary Distribution Overall
plt.subplot(4, 3, 1)
sns.histplot(df['salary_in_usd'], bins=60, kde=True)
plt.title('Distribution of Salary (USD) - All Years)')
plt.xlabel('Salary in USD')
plt.axvline(df['salary_in_usd'].median(), color='red', linestyle='--', label=f'Median: ${df["salary_in_usd"].median():,.0f}')
plt.legend()

# 2. Salary Distribution 2025 only
plt.subplot(4, 3, 2)
sns.histplot(df[df['work_year']==2025]['salary_in_usd'], bins=50, kde=True, color='green')
plt.title('Salary Distribution - 2025 Only')
plt.xlabel('Salary in USD')
plt.axvline(152000, color='red', linestyle='--')

# 3. Salary by Experience Level (Boxplot)
plt.subplot(4, 3, 3)
sns.boxplot(x='experience_level', y='salary_in_usd', data=df, order=['EN','MI','SE','EX'])
plt.title('Salary by Experience Level')
plt.ylabel('Salary (USD)')

# 4. Salary over Years (Trend)
plt.subplot(4, 3, 4)
yearly = df.groupby('work_year')['salary_in_usd'].median()
plt.plot(yearly.index, yearly.values, marker='o', linewidth=3, markersize=8)
plt.title('Median Salary Trend 2020–2025')
plt.ylabel('Median Salary (USD)')
plt.grid(True, alpha=0.3)

# 5. Salary vs Company Size
plt.subplot(4, 3, 5)
sns.boxplot(x='company_size', y='salary_in_usd', data=df, order=['S','M','L'])
plt.title('Salary by Company Size')

# 6. Remote vs Onsite vs Hybrid
plt.subplot(4, 3, 6)
sns.boxplot(x='remote_ratio', y='salary_in_usd', data=df)
plt.title('Salary by Remote Ratio')
plt.xlabel('Remote Ratio (0 = Onsite, 100 = Fully Remote)')

# 7. Top 15 Job Titles (2025)
plt.subplot(4, 3, 7)
top15 = df[df['work_year']==2025]['job_title'].value_counts().head(15)
sns.barplot(y=top15.index, x=top15.values)
plt.title('Top 15 Job Titles in 2025')
plt.xlabel('Count')

# 8. Correlation Heatmap
plt.subplot(4, 3, 8)
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Heatmap')

# 9–12: Bonus useful ones
# 9. US vs Non-US salary gap
plt.subplot(4, 3, 9)
sns.boxplot(x=df['company_location'].apply(lambda x: 'US' if x=='US' else 'Non-US'),
            y='salary_in_usd', data=df)
plt.title('US vs Rest of World Salary Gap')
plt.ylim(0, 500000)

# 10. Scatter: Salary vs Year colored by experience
plt.subplot(4, 3, 10)
sns.scatterplot(x='work_year', y='salary_in_usd', hue='experience_level',
                size='remote_ratio', data=df, alpha=0.6)
plt.title('Salary Evolution by Experience & Remote Work')

# 11. Violin plot – Experience Level (prettier than box)
plt.subplot(4, 3, 11)
sns.violinplot(x='experience_level', y='salary_in_usd', data=df[df['work_year']>=2023])
plt.title('Salary Distribution by Level (2023–2025)')

# 12. Outliers detection – Top 1% earners
plt.subplot(4, 3, 12)
top1pct = df[df['salary_in_usd'] > df['salary_in_usd'].quantile(0.99)]
sns.scatterplot(data=top1pct, x='job_title', y='salary_in_usd', hue='experience_level')
plt.xticks(rotation=90)
plt.title('Top 1% Earners (> ~$350k)')

plt.tight_layout()
plt.show()