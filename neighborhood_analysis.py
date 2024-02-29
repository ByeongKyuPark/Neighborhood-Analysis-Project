import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import trim_mean

def compute_trimmed_mean(data):
    return trim_mean(data, proportiontocut=0.05)

#Load and Clean Data
data = pd.read_excel("rollingsales_brooklyn.xls", engine='xlrd', skiprows=4)
print("Initial #rows :", len(data))

data.columns = data.columns.str.replace('\n', ' ')#ex. SALE\nDATE -> SALE DATE

data['SALE DATE'] = pd.to_datetime(data['SALE DATE'])

# blank -> NA
data = data[~data['NEIGHBORHOOD'].astype(str).str.isspace()]
data = data[~data['BUILDING CLASS CATEGORY'].astype(str).str.isspace()]

# NA -> drop
data.dropna(subset=['NEIGHBORHOOD'], inplace=True)
print("#Rows after removing rows with empty values:", len(data))

# Lower & Upper bounds
salePriceLb = 1000
landSqftLb = 1000
landSqftUb = 5000000
grossSqftLb = 150
yearBuiltLb  =1800

#trim
data = data[(data['SALE PRICE'] >= salePriceLb)]
print("#Rows after removing properties with sale prices below the lower bound:", len(data))

data = data[(data['GROSS SQUARE FEET'] >= grossSqftLb)]
print("#Rows after filtering out properties with gross square footage below the lower bound:", len(data))

data = data[(data['LAND SQUARE FEET'] >= landSqftLb) &
            (data['LAND SQUARE FEET'] <= landSqftUb)]
print("#Rows after filtering properties based on land square footage constraints:", len(data))

data = data[(data['ZIP CODE'] != 0)]
print("#Rows after eliminating entries with a ZIP CODE of 0:", len(data))

mask = ~(
    (data['YEAR BUILT'] == 0) & 
    (~data['BUILDING CLASS CATEGORY'].str.contains('GARAGE|VACANT LAND|PROPERTIES|OTHER'))
)
data = data[mask]

results = {}

neighborhoods = data['NEIGHBORHOOD'].unique()
for neighborhood in neighborhoods:
    neighborhoodData = data[data['NEIGHBORHOOD'] == neighborhood]
    
    stats = {
        "range": neighborhoodData['SALE PRICE'].max() - neighborhoodData['SALE PRICE'].min(),
        "median": neighborhoodData['SALE PRICE'].median(),
        "average": neighborhoodData['SALE PRICE'].mean(),
        "sales_count": len(neighborhoodData),
        "trimmed_mean": trim_mean(neighborhoodData['SALE PRICE'], proportiontocut=0.05),
        "std_dev": neighborhoodData['SALE PRICE'].std() if len(neighborhoodData) > 1 else "DNE (only 1 data exist)"
    }
    
    results[neighborhood] = stats

neighborhoods = list(results.keys())
trimmed_means = np.array([results[k]['trimmed_mean'] for k in results.keys()])

sortedIndices = np.argsort(trimmed_means)[::-1]
sortedNeighborhoods = [neighborhoods[i] for i in sortedIndices]
#use the same sorted indices
sortedMeans = trimmed_means[sortedIndices]

# Graphs
# Neighborhoods
# 1
plt.figure(figsize=(14, 7))
plt.bar(sortedNeighborhoods, sortedMeans, alpha=0.75, label='Trimmed Mean')

plt.xticks(rotation=90)
plt.ylabel('Trimmed Mean Sale Price')
plt.xlabel('Neighborhoods')
plt.legend()
plt.title('Pareto Chart - (Trimmed) Average Sale Price per Neighborhood')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#2
salesCounts = np.array([results[k]['sales_count'] for k in results.keys()])

# Sort based on len(neighborhoodData)
sortedIndices = np.argsort(salesCounts)[::-1]
sortedNeighborhoodsSales = [neighborhoods[i] for i in sortedIndices]
sortedCounts = salesCounts[sortedIndices]

plt.figure(figsize=(15,20))
plt.bar(sortedNeighborhoodsSales, sortedCounts, alpha=0.75, label='Number of Sales', color='green')
plt.xticks(rotation=90)
plt.ylabel('Number of Sales')
plt.title('#Sales per Neighborhood')
plt.legend()
plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
plt.show()


#3
numBuildingClasses = data['BUILDING CLASS CATEGORY'].value_counts()
sortedBuildingClasses = numBuildingClasses.index.tolist()
sortedClassesValues = numBuildingClasses.values

plt.figure(figsize=(15,10))
plt.bar(sortedBuildingClasses, sortedClassesValues,  width=0.5,alpha=0.75, color='skyblue', label='#Sales')
plt.xticks(rotation=90)
plt.ylabel('#Sales')
plt.title('#Sales per Building Class Category')
plt.legend()
plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
plt.show()

#4
expensiveNeighborhoods = data[data['NEIGHBORHOOD'] == 'DOWNTOWN-FULTON MALL     ']# 5 whitespaces after the neighborhood name in the input file
classMeans = expensiveNeighborhoods.groupby('BUILDING CLASS CATEGORY')['SALE PRICE'].mean().sort_values(ascending=False)
plt.figure(figsize=(15,10))
classMeans.plot(kind='bar', color='purple', alpha=0.75)
plt.title('Average Sale Price per Building Class in DOWNTOWN-FULTON MALL')
plt.ylabel('Average Sale Price')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#time
#1
data['MONTH'] = pd.to_datetime(data['SALE DATE']).dt.to_period('M')
monthlyAvgPrices = data.groupby('MONTH')['SALE PRICE'].mean()

plt.figure(figsize=(15, 6))
monthlyAvgPrices.plot()
plt.title('Monthly  Average Sale Price Change')
plt.ylabel('Average Sale Price')
plt.xlabel('Month')
plt.grid(True)
plt.show()

#2
    # Top 5 Neighborhoods
topNeighborhoods = sortedNeighborhoods[:5]
topNeighborhoodsData = data[data['NEIGHBORHOOD'].isin(topNeighborhoods)]

topNeighborhoodsAvg = topNeighborhoodsData.groupby(['MONTH', 'NEIGHBORHOOD'])['SALE PRICE'].apply(compute_trimmed_mean).unstack()#unstack for horizontal layout

plt.figure(figsize=(15, 6))
topNeighborhoodsAvg.plot(ax=plt.gca())#get current axis. this is for multiple subplots
plt.title('Monthly Trimmed Average Sale Price for Top 5 Neighborhoods')
plt.ylabel('Trimmed Average Sale Price')
plt.xlabel('Month')
plt.legend(title='Neighborhood')
plt.grid(True)
plt.show()

#3
    # Bottom 5 Neighborhoods
lowestNeighborhoods = sortedNeighborhoods[-5:]
lowestNeighborhoodsData = data[data['NEIGHBORHOOD'].isin(lowestNeighborhoods)]
lowestMonthlyAvg = lowestNeighborhoodsData.groupby(['MONTH', 'NEIGHBORHOOD'])['SALE PRICE'].apply(compute_trimmed_mean).unstack()#transpose
plt.figure(figsize=(15, 6))
lowestMonthlyAvg.plot(ax=plt.gca())
plt.title('Monthly Trimmed Average Sale Price for Bottom 5 Neighborhoods')
plt.ylabel('Trimmed Average Sale Price')
plt.xlabel('Month')
plt.legend(title='Neighborhood')
plt.grid(True)
plt.show()

#4
#group & accumulate
numMonthlySales = data.groupby('MONTH')['SALE PRICE'].count()

plt.figure(figsize=(15, 6))
numMonthlySales.plot(kind='bar', color='blue', alpha=0.75)
plt.title('# Sales Transactions Over the Months')
plt.ylabel('# Transactions')
plt.xlabel('Month')
plt.xticks(rotation=0)
plt.grid(True, which='both', linestyle='-', linewidth=0.35)
plt.tight_layout()
plt.show()


# export the cleaned data
data['SALE DATE'] = data['SALE DATE'].dt.strftime('%m/%d/%Y')
data.to_excel("cleaned_rollingsales_brooklyn.xlsx", index=False)

# Write the output string to a txt file
globalTrimmedMean = trim_mean(data['SALE PRICE'], proportiontocut=0.05)
output = "Overall Statistics:\n"
output += "range: " + str(data['SALE PRICE'].max() - data['SALE PRICE'].min()) + "\n"
output += "median: " + str(data['SALE PRICE'].median()) + "\n"
output += "average: " + str(int(data['SALE PRICE'].mean())) + "\n"
output += "standard deviation: " + str(int(data['SALE PRICE'].std())) + "\n"
output += "(trimmed, 0.95) mean: " + str(int(globalTrimmedMean)) + "\n"
output += "-" * 50 + "\n"

for neighborhood, stats in results.items():
    output += "Neighborhood: " + neighborhood + "\n"
    output += "range: " + str(stats["range"]) + "\n"
    output += "median: " + str(stats["median"]) + "\n"
    output += "average: " + str(stats["average"]) + "\n"
    output += "sales_count: " + str(stats["sales_count"]) + "\n"
    output += "(trimmed, 0.95) mean: " + str(stats["trimmed_mean"]) + "\n"
    output += "standard deviation: " + str(stats["std_dev"]) + "\n"
    output += "-" * 30 + "\n"

with open('parameters.txt', 'w') as file:
    file.write(output)

print("saved to 'parameters.txt'.")

